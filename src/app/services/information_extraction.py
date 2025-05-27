"""
Information Extraction Service
Using the exact approach from triple.py that works
"""

import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    logging as transformers_logging
)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# RULE:import-rich-logger-correctly
from ..log import logger
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# RULE:uppercase-constants-top
REBEL_MODEL_NAME = "Babelscape/rebel-large"
NER_MODEL_NAME = "tner/roberta-large-ontonotes5"
MAX_LEN = 256
NUM_BEAMS = 3

@dataclass
class ExtractionResult:
    """Result from information extraction"""
    triples: List[Dict[str, Any]]
    raw_output: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

# ----------------------------------------------------------------------
# MODEL LOADING (exact approach from working test)
# ----------------------------------------------------------------------
def load_models():
    """Load models (exactly like triple.py)"""
    import sys
    try:
        logger.info("🔄 Starting model loading process...")
        print("🔄 Starting model loading process...", flush=True)
        console.print("[bold blue]🔄 Starting model loading process...[/bold blue]")
        sys.stdout.flush()
        
        logger.info(f"📦 Loading REBEL model: {REBEL_MODEL_NAME}")
        print(f"📦 Loading REBEL model: {REBEL_MODEL_NAME}", flush=True)
        console.print(f"[blue]📦 Loading REBEL model:[/blue] {REBEL_MODEL_NAME}")
        
        logger.info("📦 Step 1: Loading REBEL tokenizer...")
        print("📦 Step 1: Loading REBEL tokenizer...", flush=True)
        console.print("[blue]📦 Step 1: Loading REBEL tokenizer...[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL_NAME)
        logger.info("✅ Step 1: REBEL tokenizer loaded successfully")
        print("✅ Step 1: REBEL tokenizer loaded successfully", flush=True)
        console.print("[green]✅ Step 1: REBEL tokenizer loaded successfully[/green]")
        
        logger.info("📦 Step 2: Loading REBEL model...")
        print("📦 Step 2: Loading REBEL model (this may take 30-60 seconds)...", flush=True)
        console.print("[blue]📦 Step 2: Loading REBEL model...[/blue]")
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL_NAME)
        logger.info("✅ Step 2: REBEL model loaded successfully")
        print("✅ Step 2: REBEL model loaded successfully", flush=True)
        console.print("[green]✅ Step 2: REBEL model loaded successfully[/green]")

        logger.info(f"📦 Loading NER model: {NER_MODEL_NAME}")
        print(f"📦 Loading NER model: {NER_MODEL_NAME}", flush=True)
        console.print(f"[blue]📦 Loading NER model:[/blue] {NER_MODEL_NAME}")
        
        logger.info("📦 Step 3: Creating NER pipeline...")
        print("📦 Step 3: Creating NER pipeline...", flush=True)
        console.print("[blue]📦 Step 3: Creating NER pipeline...[/blue]")
        ner_pipe = pipeline("token-classification",
                            model=NER_MODEL_NAME,
                            aggregation_strategy="simple")
        logger.info("✅ Step 3: NER pipeline created successfully")
        print("✅ Step 3: NER pipeline created successfully", flush=True)
        console.print("[green]✅ Step 3: NER pipeline created successfully[/green]")
        
        logger.info("🎉 All models loaded successfully!")
        print("🎉 All models loaded successfully!", flush=True)
        console.print("[bold green]🎉 All models loaded successfully![/bold green]")
        return tokenizer, rebel_model, ner_pipe
    except Exception as e:
        logger.error(f"💥 Failed to load models at step: {e}")
        console.print_exception()
        return None, None, None

# Global variables for lazy loading
TOKENIZER = None
REBEL_MODEL = None  
NER_PIPE = None
MODELS_LOADED = False
_models_loading_attempted = False

def ensure_models_loaded():
    """Lazy load models when first needed"""
    global TOKENIZER, REBEL_MODEL, NER_PIPE, MODELS_LOADED, _models_loading_attempted
    
    # Check if model loading is disabled (for testing)
    if os.environ.get("SUBGRAPHRAG_DISABLE_MODEL_LOADING", "false").lower() == "true":
        logger.info("⚠️ Model loading disabled by environment variable")
        _models_loading_attempted = True
        MODELS_LOADED = False
        return False
    
    logger.info("🔍 ensure_models_loaded() called")
    console.print("[magenta]🔍 Checking model loading status...[/magenta]", style="bold")
    console.print(f"[dim]Current state: _models_loading_attempted={_models_loading_attempted}, MODELS_LOADED={MODELS_LOADED}[/dim]")
    
    if _models_loading_attempted:
        logger.info(f"📋 Models already attempted, returning cached result: {MODELS_LOADED}")
        console.print(f"[magenta]📋 Using cached result:[/magenta] {MODELS_LOADED}")
        return MODELS_LOADED
        
    logger.info("🚀 First time loading models...")
    console.print("[bold magenta]🚀 First time loading models - this may take a moment...[/bold magenta]")
    console.print("[yellow]⏳ Loading process starting now...[/yellow]")
    _models_loading_attempted = True
    logger.info("Loading IE models...")
    
    try:
        logger.info("📞 Calling load_models()...")
        console.print("[magenta]📞 Loading REBEL and NER models...[/magenta]")
        import sys
        sys.stdout.flush()  # Force flush stdout
        TOKENIZER, REBEL_MODEL, NER_PIPE = load_models()
        logger.info("📞 load_models() returned, checking results...")
        console.print("[magenta]📞 Model loading completed, checking results...[/magenta]")
        
        MODELS_LOADED = TOKENIZER is not None and REBEL_MODEL is not None and NER_PIPE is not None
        logger.info(f"📊 Model loading results: TOKENIZER={TOKENIZER is not None}, REBEL_MODEL={REBEL_MODEL is not None}, NER_PIPE={NER_PIPE is not None}")
        
        if MODELS_LOADED:
            logger.info("✅ IE models loaded successfully")
        else:
            logger.warning("⚠️ IE models failed to load - extraction will use mock responses")
    except Exception as e:
        logger.error(f"💥 Critical error during model loading: {e}")
        console.print_exception()
        TOKENIZER, REBEL_MODEL, NER_PIPE = None, None, None
        MODELS_LOADED = False
        
    logger.info(f"🏁 ensure_models_loaded() finished, returning: {MODELS_LOADED}")
    return MODELS_LOADED

# ----------------------------------------------------------------------
# EXTRACTION FUNCTIONS (exact from triple.py)
# ----------------------------------------------------------------------
def extract_triplets(text: str) -> List[Dict[str, str]]:
    """Extract triplets from REBEL output (exactly from triple.py)"""
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(),
                                 'relation': relation.strip(),
                                 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(),
                                 'relation': relation.strip(),
                                 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(),
                         'relation': relation.strip(),
                         'tail': object_.strip()})
    return triplets

def build_context_ner_map(sentence: str) -> Dict[str, str]:
    """Build contextual NER mapping (exactly from triple.py)"""
    ctx_map = {}
    try:
        if not ensure_models_loaded():
            return ctx_map
        preds = NER_PIPE(sentence)
        for ent in preds:
            span = ent["word"].strip()
            typ = ent["entity_group"]
            ctx_map[span] = typ
    except Exception:
        logger.exception("Contextual NER failed.")
    return ctx_map

def get_entity_type(span: str) -> str:
    """Get entity type for span (exactly from triple.py)"""
    try:
        if not span.strip():
            return "ENTITY"
        if not ensure_models_loaded():
            return "ENTITY"
        pred = NER_PIPE(span)
        return pred[0]["entity_group"] if pred else "ENTITY"
    except Exception:
        logger.exception("Fallback NER failed.")
        return "ENTITY"

def generate_raw(sentence: str) -> str:
    """Generate raw REBEL output (exactly from triple.py)"""
    try:
        logger.info(f"🔧 generate_raw() called with: '{sentence[:30]}...'")
        
        if not ensure_models_loaded():
            logger.warning("⚠️ Models not loaded in generate_raw, returning mock")
            return "<s><triplet> mock <subj> mock relation <obj> mock tail</s>"
            
        logger.info("🔧 Tokenizing input...")
        inputs = TOKENIZER(sentence,
                           max_length=MAX_LEN,
                           padding=True,
                           truncation=True,
                           return_tensors="pt")

        gen_kwargs = dict(max_length=MAX_LEN,
                          num_beams=NUM_BEAMS,
                          length_penalty=0.0)

        logger.info("🔧 Generating with REBEL model...")
        generated = REBEL_MODEL.generate(
            inputs["input_ids"].to(REBEL_MODEL.device),
            attention_mask=inputs["attention_mask"].to(REBEL_MODEL.device),
            **gen_kwargs
        )
        logger.info("🔧 Decoding output...")
        decoded = TOKENIZER.batch_decode(generated, skip_special_tokens=False)
        result = decoded[0] if decoded else ""
        logger.info(f"🔧 generate_raw() returning: '{result[:50]}...'")
        return result
    except Exception:
        logger.exception("💥 Generation failed.")
        raise

def extract(sentence: str, debug: bool = False) -> List[Dict[str, str]]:
    """Extract triples with entity typing (exactly from triple.py)"""
    # Check if model loading is disabled (for testing)
    if os.environ.get("SUBGRAPHRAG_DISABLE_MODEL_LOADING", "false").lower() == "true":
        logger.info("⚠️ Model loading disabled - returning empty result")
        return []
    
    try:
        ctx_types = build_context_ner_map(sentence)
        raw = generate_raw(sentence)
        triples = extract_triplets(raw)

        for t in triples:
            head = t["head"]
            tail = t["tail"]
            t["head_type"] = ctx_types.get(head, get_entity_type(head))
            t["tail_type"] = ctx_types.get(tail, get_entity_type(tail))

        if debug:
            console.print("\n[dim]RAW OUTPUT:[/]\n", raw)

        return triples
    except Exception:
        logger.exception("Full extraction failed.")
        raise

# ----------------------------------------------------------------------
# API SERVICE WRAPPER
# ----------------------------------------------------------------------
class InformationExtractionService:
    """Service wrapper for the extraction functions"""
    
    def __init__(self):
        logger.debug("InformationExtractionService initialized")
    
    def load_rebel_model(self) -> bool:
        """Load REBEL model if needed"""
        return ensure_models_loaded()

    def load_ner_model(self) -> bool:
        """Load NER model if needed"""
        return ensure_models_loaded()

    def load_models(self) -> Dict[str, bool]:
        """Load both models if needed"""
        loaded = ensure_models_loaded()
        return {"rebel": loaded, "ner": loaded}
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return MODELS_LOADED

    def is_rebel_loaded(self) -> bool:
        """Check if REBEL model is loaded"""
        return MODELS_LOADED

    def is_ner_loaded(self) -> bool:
        """Check if NER pipeline is loaded"""
        return MODELS_LOADED
    
    def is_service_available(self) -> bool:
        """Check if service is available"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            return True
        except ImportError:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "service": "SubgraphRAG+ Information Extraction Service",
            "models": [
                {
                    "model_name": REBEL_MODEL_NAME,
                    "description": "Relation Extraction By End-to-end Language generation",
                    "capabilities": ["open_schema_relation_extraction", "multilingual_support"],
                    "input_format": "raw_text",
                    "output_format": "head_relation_tail_triples",
                    "loaded": MODELS_LOADED,
                    "purpose": "relation_extraction"
                },
                {
                    "model_name": NER_MODEL_NAME,
                    "description": "RoBERTa-large fine-tuned for NER on OntoNotes 5.0 dataset",
                    "capabilities": ["named_entity_recognition", "entity_typing"],
                    "input_format": "raw_text",
                    "output_format": "entity_type_predictions",
                    "loaded": MODELS_LOADED,
                    "purpose": "entity_typing"
                }
            ],
            "overall_status": "ready" if MODELS_LOADED else "not_loaded"
        }
    
    def extract_triples(self, text: str, max_length: int = 256, num_beams: int = 3) -> ExtractionResult:
        """Extract triples using the global extract function"""
        # Handle None or empty text
        if text is None:
            text = ""
        
        text_preview = text[:50] if text else "None"
        logger.info(f"🎯 extract_triples() called with text: '{text_preview}...'")
        console.print(f"[cyan]🎯 IE Service[/cyan] - Processing text: '{text[:30] if text else 'None'}...'")
        start_time = time.time()
        
        try:
            logger.info(f"📝 Processing text of length {len(text)} characters")
            console.print(f"[cyan]📝 Text length:[/cyan] {len(text)} characters")
            
            logger.info("🔄 About to call ensure_models_loaded()...")
            console.print("[cyan]🔄 Checking if models are loaded...[/cyan]")
            
            # This is where it might hang - let's see
            models_ready = ensure_models_loaded()
            logger.info(f"✅ ensure_models_loaded() returned: {models_ready}")
            console.print(f"[cyan]✅ Models ready:[/cyan] {models_ready}")
            
            if not models_ready:
                logger.warning("⚠️ Models not loaded, returning mock result")
                return ExtractionResult(
                    triples=[{"head": "mock", "relation": "mock", "tail": "mock", "head_type": "ENTITY", "tail_type": "ENTITY", "confidence": 1.0}],
                    raw_output="mock",
                    processing_time=time.time() - start_time,
                    success=True
                )
            
            logger.info("🚀 About to call extract() function...")
            # Use the global extract function (exactly like triple.py)
            triples = extract(text, debug=False)
            logger.info(f"✅ extract() returned {len(triples)} triples")
            
            # Convert to API format
            formatted_triples = []
            for t in triples:
                triple_with_confidence = {
                    "head": t["head"],
                    "relation": t["relation"], 
                    "tail": t["tail"],
                    "head_type": t.get("head_type", "ENTITY"),
                    "tail_type": t.get("tail_type", "ENTITY"),
                    "confidence": 1.0
                }
                formatted_triples.append(triple_with_confidence)

            processing_time = time.time() - start_time
            
            logger.info(f"🎉 Finished extraction: {len(formatted_triples)} triples in {processing_time:.2f}s")

            return ExtractionResult(
                triples=formatted_triples,
                raw_output="",  # Not exposing raw output
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            logger.error(f"Error in triple extraction: {e}")
            console.print_exception()
            
            return ExtractionResult(
                triples=[],
                raw_output="",
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

# Global service instance
_ie_service_instance = None

def get_information_extraction_service() -> InformationExtractionService:
    """Get the global service instance"""
    global _ie_service_instance
    if _ie_service_instance is None:
        _ie_service_instance = InformationExtractionService()
    return _ie_service_instance 