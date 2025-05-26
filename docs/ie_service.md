# Information Extraction Service Documentation

## Overview

The SubgraphRAG+ Information Extraction (IE) Service provides production-grade triple extraction and entity typing capabilities. It combines REBEL relation extraction with a sophisticated schema-first entity typing system.

## Architecture

```
Raw Text → REBEL IE Service → Triples → Entity Typing → Knowledge Graph
```

### Components

1. **REBEL IE Service**: FastAPI microservice for relation extraction
2. **Entity Typing Service**: Schema-first typing with NER fallback
3. **Triple Processing Pipeline**: Centralized processing and validation

## REBEL IE Service

### Setup and Deployment

#### Docker Deployment

```bash
# Build IE service container
docker build -f Dockerfile.ie -t subgraphrag-ie .

# Run IE service
docker run -p 8003:8003 subgraphrag-ie

# Or use docker-compose
docker-compose up ie-service
```

#### Manual Setup

```bash
# Install dependencies
pip install transformers torch

# Download REBEL model (first run only)
python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('Babelscape/rebel-large')"

# Start service
uvicorn src.app.ie_service:app --host 0.0.0.0 --port 8003
```

### API Reference

#### POST /extract

Extract triples from raw text using REBEL.

**Request:**
```json
{
  "text": "Jesus was born in Bethlehem and later lived in Nazareth.",
  "max_length": 256,
  "num_beams": 3
}
```

**Response:**
```json
{
  "triples": [
    {
      "head": "Jesus",
      "relation": "place of birth", 
      "tail": "Bethlehem"
    },
    {
      "head": "Jesus",
      "relation": "residence",
      "tail": "Nazareth"
    }
  ],
  "processing_time": 0.234,
  "model_info": {
    "model": "Babelscape/rebel-large",
    "version": "1.0"
  }
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 3600
}
```

#### GET /info

Service information.

**Response:**
```json
{
  "service": "REBEL IE Service",
  "version": "1.0.0",
  "model": "Babelscape/rebel-large",
  "capabilities": ["relation_extraction", "triple_extraction"]
}
```

### Performance Characteristics

- **Latency**: ~200ms per text chunk on CPU, ~50ms on GPU
- **Throughput**: ~300 texts/minute on CPU, ~1200 texts/minute on GPU
- **Memory**: ~2GB for model, ~1GB working memory
- **Model Size**: ~1.5GB download

### Batch Processing

For high-throughput scenarios, use the batch processing utilities:

```python
from app.utils.triple_extraction import batch_process_texts

texts = [
    "Jesus was born in Bethlehem.",
    "Moses led the Israelites out of Egypt.",
    "David defeated Goliath."
]

triples = batch_process_texts(texts, "http://localhost:8003")
```

## Entity Typing Service

### Schema-First Approach

The entity typing system uses a **schema-first approach** that prioritizes existing knowledge graph data over machine learning predictions.

#### Flow

1. **Schema Lookup**: Check Neo4j for existing entity type
2. **NER Fallback**: Use transformer model if schema miss
3. **Cache Result**: Store prediction for future use

### Implementation

```python
from app.entity_typing import detect_entity_type, batch_detect_entity_types

# Single entity
entity_type = detect_entity_type("Jesus")  # Returns: "Person"

# Batch processing (more efficient)
entities = ["Jesus", "Jerusalem", "Israelites"]
types = batch_detect_entity_types(entities)
# Returns: {"Jesus": "Person", "Jerusalem": "Location", "Israelites": "Organization"}
```

### NER Models

#### Primary: BERT-based NER

- **Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Labels**: PER, ORG, LOC, GPE, MISC
- **Accuracy**: ~95% on CoNLL-03 test set
- **Performance**: ~50ms per entity on CPU

#### Fallback: spaCy NER

- **Model**: `en_core_web_sm`
- **Labels**: PERSON, ORG, GPE, LOC, MISC
- **Accuracy**: ~90% on general text
- **Performance**: ~10ms per entity on CPU

### Entity Type Hierarchy

```
Entity (root)
├── Person
├── Location  
├── Organization
├── Event
├── Concept
└── Entity (default)
```

### Configuration

#### Environment Variables

```bash
# NER Model Selection
NER_MODEL=dbmdz/bert-large-cased-finetuned-conll03-english
NER_DEVICE=-1  # -1 for CPU, 0 for GPU

# Cache Settings
ENTITY_TYPE_CACHE_SIZE=4096
ENTITY_TYPE_CACHE_TTL=3600
```

#### Model Downloads

```bash
# Download BERT NER model
python -c "from transformers import pipeline; pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Integration Examples

### Basic Ingestion Pipeline

```python
from app.utils.triple_extraction import process_rebel_output
from app.entity_typing import detect_entity_type

# Raw text
text = "Moses parted the Red Sea and led the Israelites across."

# Extract triples via REBEL
rebel_output = "<s><triplet> Moses <subj> Red Sea <obj> parted</s>"
triples = process_rebel_output(rebel_output)

# Each triple now has entity types
for triple in triples:
    print(f"({triple.head}:{triple.head_type}) --[{triple.relation}]--> ({triple.tail}:{triple.tail_type})")
    # Output: (Moses:Person) --[parted]--> (Red Sea:Location)
```

### Batch Processing with IE Service

```python
from app.utils.triple_extraction import batch_process_texts

# Process multiple texts
texts = [
    "Jesus was born in Bethlehem.",
    "Moses received the Ten Commandments on Mount Sinai.",
    "David ruled Israel from Jerusalem."
]

# Extract and type all entities
triples = batch_process_texts(texts, "http://localhost:8003")

# Results include both relations and entity types
for triple in triples:
    print(f"Source: {triple.source}")
    print(f"Triple: ({triple.head}:{triple.head_type}) --[{triple.relation}]--> ({triple.tail}:{triple.tail_type})")
    print(f"Confidence: {triple.confidence}")
```

### Schema Updates

```python
from app.entity_typing import update_entity_type_in_schema

# Persist NER predictions to schema
success = update_entity_type_in_schema("NewEntity", "Person")
if success:
    print("Schema updated successfully")
```

## Troubleshooting

### Common Issues

#### IE Service Not Starting

```bash
# Check model download
ls ~/.cache/huggingface/transformers/

# Check dependencies
pip install transformers torch

# Check port availability
lsof -i :8003
```

#### NER Model Loading Errors

```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/

# Reinstall transformers
pip uninstall transformers
pip install transformers>=4.35.0

# Test model loading
python -c "from transformers import pipeline; print(pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english'))"
```

#### Entity Typing Performance Issues

```bash
# Check cache hit rate
python -c "from app.entity_typing import detect_entity_type; print(detect_entity_type.cache_info())"

# Clear cache if needed
python -c "from app.entity_typing import detect_entity_type; detect_entity_type.cache_clear()"

# Use batch processing for better performance
python -c "from app.entity_typing import batch_detect_entity_types; print('Batch processing available')"
```

### Performance Tuning

#### GPU Acceleration

```python
# Enable GPU for NER
import os
os.environ['NER_DEVICE'] = '0'  # Use first GPU

# Verify GPU usage
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### Batch Size Optimization

```python
# Tune batch size for your hardware
from app.utils.triple_extraction import batch_process_texts

# Start with small batches
batch_size = 16
texts_batched = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

for batch in texts_batched:
    triples = batch_process_texts(batch, "http://localhost:8003")
```

#### Memory Management

```bash
# Monitor memory usage
htop

# Reduce cache size if needed
export ENTITY_TYPE_CACHE_SIZE=1024

# Use CPU-only mode to save GPU memory
export NER_DEVICE=-1
```

## Monitoring and Metrics

### Health Checks

```bash
# IE Service health
curl http://localhost:8003/health

# Entity typing test
python -c "from app.entity_typing import detect_entity_type; print(detect_entity_type('test'))"
```

### Performance Metrics

```python
import time
from app.entity_typing import detect_entity_type

# Measure entity typing performance
start = time.time()
result = detect_entity_type("Jesus")
elapsed = time.time() - start
print(f"Entity typing took {elapsed:.3f}s, result: {result}")

# Check cache statistics
print(detect_entity_type.cache_info())
```

### Logging

```python
import logging

# Enable debug logging for entity typing
logging.getLogger('app.entity_typing').setLevel(logging.DEBUG)

# Enable debug logging for IE service
logging.getLogger('app.ie_service').setLevel(logging.DEBUG)
```

## Best Practices

### Production Deployment

1. **Use Docker**: Deploy IE service in containers for isolation
2. **Load Balancing**: Run multiple IE service instances behind a load balancer
3. **Monitoring**: Set up health checks and metrics collection
4. **Caching**: Use Redis for distributed caching in multi-instance setups

### Performance Optimization

1. **Batch Processing**: Always use batch functions for multiple entities
2. **Schema First**: Populate Neo4j with known entities to reduce NER calls
3. **GPU Usage**: Use GPU for NER in high-throughput scenarios
4. **Cache Tuning**: Monitor cache hit rates and adjust sizes accordingly

### Error Handling

1. **Graceful Degradation**: System continues working even if NER fails
2. **Retry Logic**: Implement retries for transient failures
3. **Dead Letter Queue**: Handle failed extractions appropriately
4. **Monitoring**: Set up alerts for high error rates

## API Integration

### Python Client

```python
import requests

class IEServiceClient:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
    
    def extract_triples(self, text, max_length=256, num_beams=3):
        response = requests.post(
            f"{self.base_url}/extract",
            json={
                "text": text,
                "max_length": max_length,
                "num_beams": num_beams
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.status_code == 200

# Usage
client = IEServiceClient()
result = client.extract_triples("Jesus was born in Bethlehem.")
print(result["triples"])
```

### cURL Examples

```bash
# Extract triples
curl -X POST http://localhost:8003/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Moses led the Israelites out of Egypt."}'

# Health check
curl http://localhost:8003/health

# Service info
curl http://localhost:8003/info
``` 