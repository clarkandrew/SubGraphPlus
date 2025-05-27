"""
Information Extraction Service
Handles REBEL model management, text processing, and triple extraction
Separated from API layer to follow single responsibility principle
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as transformers_logging

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# RULE:import-rich-logger-correctly
from ..log import logger
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# RULE:uppercase-constants-top
REBEL_MODEL_ID = "Babelscape/rebel-large"

@dataclass
class ExtractionResult:
    """Result from information extraction"""
    triples: List[Dict[str, Any]]
    raw_output: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class InformationExtractionService:
    """
    Service for information extraction using REBEL model
    Handles model lifecycle, text processing, and triple extraction
    """
    
    def __init__(self):
        """Initialize the Information Extraction Service"""
        # RULE:debug-trace-every-step
        logger.debug("Starting InformationExtractionService initialization")
        
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        logger.debug("Finished InformationExtractionService initialization")
    
    def load_model(self) -> bool:
        """
        Load REBEL model and tokenizer
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True
            
        logger.debug("Starting REBEL model loading")
        
        try:
            logger.info("Loading Babelscape/rebel-large model...")
            self._tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL_ID)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL_ID)
            self._model_loaded = True
            logger.info("REBEL model loaded successfully")
            return True
            
        except Exception as e:
            # RULE:rich-error-handling-required
            logger.error(f"Failed to load REBEL model: {e}")
            console.print_exception()
            self._model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model_loaded and self._model is not None and self._tokenizer is not None
    
    def is_service_available(self) -> bool:
        """Check if service is available (without loading models)"""
        try:
            # Check if we can import the required dependencies
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            return True
        except ImportError:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": REBEL_MODEL_ID,
            "description": "Relation Extraction By End-to-end Language generation",
            "capabilities": ["open_schema_relation_extraction", "multilingual_support"],
            "input_format": "raw_text",
            "output_format": "head_relation_tail_triples",
            "loaded": self._model_loaded
        }
    
    def extract_triples(self, text: str, max_length: int = 256, num_beams: int = 3) -> ExtractionResult:
        """
        Extract triples from input text using REBEL
        
        Args:
            text: Input text to extract triples from
            max_length: Maximum sequence length for REBEL
            num_beams: Number of beams for beam search
            
        Returns:
            ExtractionResult with extracted triples and metadata
        """
        # RULE:debug-trace-every-step
        logger.debug("Starting triple extraction")
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.is_model_loaded():
                if not self.load_model():
                    return ExtractionResult(
                        triples=[],
                        raw_output="",
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Failed to load REBEL model"
                    )
            
            logger.debug(f"Processing text of length {len(text)} characters")
            
            # Tokenize input text
            model_inputs = self._tokenizer(
                text, 
                max_length=max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )

            # Generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "length_penalty": 0,
                "num_beams": num_beams,
                "num_return_sequences": 1,
            }

            # Generate tokens
            generated_tokens = self._model.generate(
                model_inputs["input_ids"].to(self._model.device),
                attention_mask=model_inputs["attention_mask"].to(self._model.device),
                **gen_kwargs,
            )

            # Decode with special tokens
            decoded_preds = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

            # Extract triplets from first prediction
            raw_output = decoded_preds[0] if decoded_preds else ""
            triplet_dicts = self._extract_triplets_from_rebel_output(raw_output)

            # Convert to standardized format
            triples = [
                {
                    "head": t['head'],
                    "relation": t['relation'], 
                    "tail": t['tail'],
                    "confidence": 1.0  # REBEL doesn't provide confidence scores
                }
                for t in triplet_dicts
            ]

            processing_time = time.time() - start_time
            
            logger.debug(f"Finished triple extraction: {len(triples)} triples in {processing_time:.2f}s")

            return ExtractionResult(
                triples=triples,
                raw_output=raw_output,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            # RULE:rich-error-handling-required
            logger.error(f"Error in triple extraction: {e}")
            console.print_exception()
            
            return ExtractionResult(
                triples=[],
                raw_output="",
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _extract_triplets_from_rebel_output(self, text: str, debug: bool = False) -> List[Dict[str, str]]:
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


# Global service instance for singleton pattern
_ie_service_instance = None

def get_information_extraction_service() -> InformationExtractionService:
    """
    Get the global Information Extraction Service instance
    Implements singleton pattern for model management
    """
    global _ie_service_instance
    if _ie_service_instance is None:
        _ie_service_instance = InformationExtractionService()
    return _ie_service_instance 