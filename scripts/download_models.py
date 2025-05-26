#!/usr/bin/env python3
"""
Model Download Script for SubgraphRAG+ Entity Typing
Downloads and vendors models based on config.json definitions
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def download_ontonotes_model(config: dict = None):
    """
    Download the OntoNotes-5 NER model for offline usage
    Uses configuration from config.json
    
    Args:
        config: Configuration dictionary (loaded from config.json)
    """
    if not config:
        config = load_config()
    
    ontonotes_config = config.get("models", {}).get("entity_typing", {}).get("ontonotes_ner", {})
    
    model_name = ontonotes_config.get("model", "tner/roberta-large-ontonotes5")
    target_dir = ontonotes_config.get("local_path", "models/roberta-large-ontonotes5")
    
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading OntoNotes-5 NER model: {model_name}")
        logger.info(f"Target directory: {target_dir}")
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Successfully downloaded model to {target_dir}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def download_rebel_model(config: dict = None):
    """
    Download the REBEL model for information extraction
    Uses configuration from config.json
    
    Args:
        config: Configuration dictionary (loaded from config.json)
    """
    if not config:
        config = load_config()
    
    rebel_config = config.get("models", {}).get("information_extraction", {}).get("rebel", {})
    
    model_name = rebel_config.get("model", "Babelscape/rebel-large")
    target_dir = rebel_config.get("local_path", "models/rebel-large")
    
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading REBEL model: {model_name}")
        logger.info(f"Target directory: {target_dir}")
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Successfully downloaded model to {target_dir}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def download_spacy_model(config: dict = None):
    """
    Download spaCy model for fallback NER
    Uses configuration from config.json
    """
    if not config:
        config = load_config()
    
    spacy_config = config.get("models", {}).get("entity_typing", {}).get("spacy_fallback", {})
    model_name = spacy_config.get("model", "en_core_web_sm")
    
    try:
        import subprocess
        
        logger.info(f"Downloading spaCy model: {model_name}")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", model_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Successfully downloaded spaCy model")
            return True
        else:
            logger.error(f"Failed to download spaCy model: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False

def verify_models(config: dict = None):
    """Verify that downloaded models can be loaded"""
    if not config:
        config = load_config()
    
    logger.info("Verifying model installations...")
    
    # Test OntoNotes model
    try:
        import tner
        ontonotes_config = config.get("models", {}).get("entity_typing", {}).get("ontonotes_ner", {})
        model_path = ontonotes_config.get("local_path", "models/roberta-large-ontonotes5")
        
        if Path(model_path).exists():
            logger.info(f"Testing OntoNotes model from {model_path}")
            model = tner.TransformersNER(model_path)
            test_result = model.predict(["test"])
            logger.info("✓ OntoNotes model loaded successfully")
        else:
            logger.warning(f"OntoNotes model not found at {model_path}")
    except Exception as e:
        logger.error(f"✗ OntoNotes model test failed: {e}")
    
    # Test REBEL model
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        rebel_config = config.get("models", {}).get("information_extraction", {}).get("rebel", {})
        model_path = rebel_config.get("local_path", "models/rebel-large")
        
        if Path(model_path).exists():
            logger.info(f"Testing REBEL model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            logger.info("✓ REBEL model loaded successfully")
        else:
            logger.warning(f"REBEL model not found at {model_path}")
    except Exception as e:
        logger.error(f"✗ REBEL model test failed: {e}")
    
    # Test spaCy model
    try:
        import spacy
        spacy_config = config.get("models", {}).get("entity_typing", {}).get("spacy_fallback", {})
        model_name = spacy_config.get("model", "en_core_web_sm")
        
        nlp = spacy.load(model_name)
        doc = nlp("test")
        logger.info("✓ spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"✗ spaCy model test failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download models for SubgraphRAG+ based on config.json")
    parser.add_argument("--model", choices=["ontonotes", "rebel", "spacy", "all"], default="all",
                       help="Which model to download (default: all)")
    parser.add_argument("--skip-spacy", action="store_true",
                       help="Skip downloading spaCy model")
    parser.add_argument("--verify", action="store_true",
                       help="Verify models after download")
    parser.add_argument("--config", help="Path to config.json file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_path = Path(args.config)
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return 1
    else:
        config = load_config()
    
    if not config:
        logger.error("No configuration loaded, cannot proceed")
        return 1
    
    logger.info("Using configuration:")
    logger.info(f"  OntoNotes: {config.get('models', {}).get('entity_typing', {}).get('ontonotes_ner', {}).get('model', 'N/A')}")
    logger.info(f"  REBEL: {config.get('models', {}).get('information_extraction', {}).get('rebel', {}).get('model', 'N/A')}")
    logger.info(f"  spaCy: {config.get('models', {}).get('entity_typing', {}).get('spacy_fallback', {}).get('model', 'N/A')}")
    
    success = True
    
    # Download models based on selection
    if args.model in ["ontonotes", "all"]:
        if not download_ontonotes_model(config):
            success = False
    
    if args.model in ["rebel", "all"]:
        if not download_rebel_model(config):
            success = False
    
    if args.model in ["spacy", "all"] and not args.skip_spacy:
        if not download_spacy_model(config):
            success = False
    
    # Verify models if requested
    if args.verify:
        verify_models(config)
    
    if success:
        logger.info("All requested models downloaded successfully!")
        return 0
    else:
        logger.error("Some models failed to download")
        return 1

if __name__ == "__main__":
    sys.exit(main())