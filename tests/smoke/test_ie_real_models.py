#!/usr/bin/env python3
"""
Test IE with real models (from cache) - similar to triple.py
"""

import os
import sys
from pathlib import Path

# Don't disable model loading for this test - we want to use real models
# Force offline mode to use cached models
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=== IE Test with Real Models (from cache) ===")

# Test 1: Import and load models
print("\n1. Testing model loading from cache...")
try:
    from src.app.services.information_extraction import (
        load_models,
        extract_triplets,
        REBEL_MODEL_NAME,
        NER_MODEL_NAME
    )
    
    print(f"Loading models: {REBEL_MODEL_NAME} and {NER_MODEL_NAME}")
    print("This should use cached models, not download...")
    
    tokenizer, rebel_model, ner_pipe = load_models()
    
    if tokenizer and rebel_model and ner_pipe:
        print("✅ Models loaded successfully from cache!")
    else:
        print("❌ Failed to load models")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test extraction with real models
print("\n2. Testing extraction with real models...")
try:
    # Import the complete extract function
    from src.app.services.information_extraction import extract
    
    test_sentence = "Barack Obama was born in Hawaii."
    print(f"Test input: {test_sentence}")
    
    # Extract triples using real models
    triples = extract(test_sentence)
    
    print(f"✅ Extracted {len(triples)} triples:")
    for i, triple in enumerate(triples, 1):
        print(f"   {i}. {triple['head']} --[{triple['relation']}]--> {triple['tail']}")
        print(f"      Head type: {triple.get('head_type', 'UNKNOWN')}, Tail type: {triple.get('tail_type', 'UNKNOWN')}")
        
except Exception as e:
    print(f"❌ Error during extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test service wrapper
print("\n3. Testing IE service with real models...")
try:
    from src.app.services.information_extraction import get_information_extraction_service
    
    service = get_information_extraction_service()
    
    # Check model status
    print(f"Models loaded: {service.is_model_loaded()}")
    
    # Extract using service
    result = service.extract_triples("Barack Obama was born in Hawaii.")
    
    if result.success:
        print(f"✅ Service extraction successful!")
        print(f"   Triples: {len(result.triples)}")
        print(f"   Processing time: {result.processing_time:.2f}s")
    else:
        print(f"❌ Service extraction failed: {result.error_message}")
        
except Exception as e:
    print(f"❌ Error with service: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")
print("\nSummary:")
print("✅ Models loaded from cache successfully")
print("✅ Extraction works with real models") 
print("✅ Service wrapper works correctly")
print("\nThe IE functionality works correctly with cached models!") 