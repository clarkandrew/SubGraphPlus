"""
Information Extraction Service using Babelscape/rebel-large
Provides proper triple extraction for SubgraphRAG+ ingestion pipeline
"""

import re
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as transformers_logging

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# RULE:import-rich-logger-correctly - Use centralized rich logger
from .log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

app = FastAPI(title="REBEL IE Service", description="Information Extraction using Babelscape/rebel-large")

# Global model and tokenizer
model = None
tokenizer = None

class ExtractRequest(BaseModel):
    text: str
    max_length: int = 256
    num_beams: int = 3

class Triple(BaseModel):
    head: str
    relation: str
    tail: str
    confidence: float = 1.0

class ExtractResponse(BaseModel):
    triples: List[Triple]
    raw_output: str
    processing_time: float

def load_model():
    """Load REBEL model and tokenizer"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        logger.info("Loading Babelscape/rebel-large model...")
        try:
            MODEL_ID = "Babelscape/rebel-large"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
            logger.info("REBEL model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load REBEL model: {e}")
            raise HTTPException(500, f"Model loading failed: {e}")

def extract_triplets_from_rebel(text: str, debug: bool = False) -> List[Dict[str, str]]:
    """
    Extract triplets from REBEL model output
    
    Args:
        text: Raw model output text
        debug: Enable debug logging
        
    Returns:
        List of triplet dictionaries with 'head', 'relation', 'tail' keys
    """
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'

    if debug:
        logger.debug(f"Processing REBEL output: {text}")

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if debug:
            logger.debug(f"Token='{token}', Current='{current}', Subject='{subject}', Object='{object_}', Relation='{relation}'")

        if token == "<triplet>":
            current = 't'  # Start with subject after <triplet>
            if relation != '' and subject != '' and object_ != '':
                triplets.append({
                    'head': subject.strip(), 
                    'relation': relation.strip(),
                    'tail': object_.strip()
                })
                if debug:
                    logger.debug(f"Added triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")
            relation = ''
            subject = ''
            object_ = ''
        elif token == "<subj>":
            current = 's'  # Switch to object after <subj>
        elif token == "<obj>":
            current = 'o'  # Switch to relation after <obj>
        else:
            if current == 't':
                subject += ' ' + token if subject else token
            elif current == 's':
                object_ += ' ' + token if object_ else token
            elif current == 'o':
                relation += ' ' + token if relation else token

    # Add final triplet if exists
    if subject != '' and relation != '' and object_ != '':
        triplets.append({
            'head': subject.strip(), 
            'relation': relation.strip(),
            'tail': object_.strip()
        })
        if debug:
            logger.debug(f"Final triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")

    return triplets

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/extract", response_model=ExtractResponse)
async def extract_triples(request: ExtractRequest):
    """
    Extract triples from input text using REBEL
    
    Args:
        request: ExtractRequest with text and optional parameters
        
    Returns:
        ExtractResponse with extracted triples and metadata
    """
    import time
    start_time = time.time()
    
    try:
        # Ensure model is loaded
        if model is None or tokenizer is None:
            load_model()
        
        # Tokenize input text
        model_inputs = tokenizer(
            request.text, 
            max_length=request.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        # Generation parameters
        gen_kwargs = {
            "max_length": request.max_length,
            "length_penalty": 0,
            "num_beams": request.num_beams,
            "num_return_sequences": 1,
        }

        # Generate tokens
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            **gen_kwargs,
        )

        # Decode with special tokens
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # Extract triplets from first prediction
        raw_output = decoded_preds[0] if decoded_preds else ""
        triplet_dicts = extract_triplets_from_rebel(raw_output)

        # Convert to Triple objects
        triples = [
            Triple(
                head=t['head'],
                relation=t['relation'], 
                tail=t['tail'],
                confidence=1.0  # REBEL doesn't provide confidence scores
            )
            for t in triplet_dicts
        ]

        processing_time = time.time() - start_time

        return ExtractResponse(
            triples=triples,
            raw_output=raw_output,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in triple extraction: {e}")
        raise HTTPException(500, f"Extraction failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if model is None or tokenizer is None:
            return {"status": "unhealthy", "reason": "Model not loaded"}
        return {"status": "healthy", "model": "Babelscape/rebel-large"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Babelscape/rebel-large",
        "description": "Relation Extraction By End-to-end Language generation",
        "capabilities": ["open_schema_relation_extraction", "multilingual_support"],
        "input_format": "raw_text",
        "output_format": "head_relation_tail_triples"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 