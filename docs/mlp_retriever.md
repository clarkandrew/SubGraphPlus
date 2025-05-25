# MLP Retriever in SubgraphPlus

## Overview

SubgraphPlus incorporates the MLP (Multi-Layer Perceptron) retriever from the original SubgraphRAG repository to score and rank knowledge graph triples for relevance to natural language queries. The pre-trained MLP model (`models/mlp/mlp.pth`) was trained using the process documented in `notebooks/train_SubGraph_MLP.ipynb`.

## Architecture

### MLP Model Structure

The MLP retriever is a shallow neural network designed for efficient triple scoring:

```python
# Expected MLP architecture from original SubgraphRAG
class Retriever(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output score
        )
```

### Input Feature Vector

The MLP takes a concatenated feature vector with the following components:

1. **Query Embedding** (384 dims): Embedding of the natural language question
2. **Triple Embedding** (384 dims): Embedding of the triple (head, relation, tail)
3. **DDE Features** (variable dims): Deep Distance Embedding features

**Total Input Dimension**: `768 + DDE_feature_count`

### DDE (Deep Distance Embedding) Features

DDE features capture the graph-theoretic distance between query entities and triple entities:

- **Hop-based encoding**: For each query entity and each hop (1 to MAX_DDE_HOPS=2)
- **Head/Tail features**: Separate features for head and tail entities of the triple
- **Distance weighting**: Closer entities get higher scores (1.0/hop_distance)

**Feature Count**: `num_query_entities × MAX_DDE_HOPS × 2 (head/tail)`

## Usage in SubgraphPlus

### Model Loading

The MLP model is loaded at startup in `src/app/retriever.py`:

```python
def load_pretrained_mlp():
    """Load pre-trained SubgraphRAG MLP model"""
    mlp_path = config.MLP_MODEL_PATH  # "models/mlp/mlp.pth"
    return torch.load(mlp_path, map_location=torch.device('cpu'))

mlp_model = load_pretrained_mlp()
```

### Scoring Function

```python
def mlp_score(query_embedding, triple_embedding, dde_features):
    """Score a triple using the MLP or fallback heuristic"""
    if mlp_model is not None:
        # Convert inputs to torch tensors
        q_emb = torch.tensor(query_embedding, dtype=torch.float32)
        t_emb = torch.tensor(triple_embedding, dtype=torch.float32)
        dde_feat = torch.tensor(dde_features, dtype=torch.float32)
        
        # Concatenate features
        combined = torch.cat([q_emb, t_emb, dde_feat])
        
        # Get score from MLP
        with torch.no_grad():
            score = mlp_model(combined).item()
        
        return score
    else:
        # Fallback to heuristic scoring
        return heuristic_score(query_embedding, triple_embedding, dde_features)
```

### Integration in Retrieval Pipeline

The MLP scorer is used in the hybrid retrieval pipeline (`hybrid_retrieve_v2`):

1. **Candidate Generation**: Graph traversal + dense retrieval via FAISS
2. **Feature Extraction**: Query embedding, triple embedding, DDE features
3. **MLP Scoring**: Each candidate triple gets a relevance score
4. **Selection**: Top-K triples selected based on scores
5. **Subgraph Assembly**: Connected subgraph construction with token budget

## Configuration

Key configuration parameters in `config/config.json`:

```json
{
  "MLP_MODEL_PATH": "models/mlp/mlp.pth",
  "MAX_DDE_HOPS": 2,
  "TOKEN_BUDGET": 4000
}
```

## Model Training

The MLP model was trained using the original SubgraphRAG training pipeline:

### Training Data
- **Datasets**: WebQSP, ComplexWebQuestions (CWQ)
- **Labels**: Binary labels for triples on gold paths to answers
- **Features**: Same input format as inference (query + triple + DDE)

### Training Process
The training notebook `notebooks/train_SubGraph_MLP.ipynb` follows these steps:

1. **Environment Setup**: Install SubgraphRAG dependencies
2. **Embedding Precomputation**: Cache embeddings for efficiency
3. **Retriever Training**: Train MLP on labeled triple relevance data
4. **Model Evaluation**: Recall@K metrics on validation set

### Training Configuration
From original SubgraphRAG:
```yaml
retriever:
  topic_pe: true
  DDE_kwargs:
    num_rounds: 2
    num_reverse_rounds: 2
optimizer:
  lr: 1e-3
train:
  num_epochs: 10000
  patience: 10
```

## Fallback Mechanism

When the MLP model is unavailable, the system falls back to heuristic scoring:

```python
def heuristic_score(query_embedding, triple_embedding, dde_value):
    """Fallback scoring when MLP unavailable"""
    cosine_sim = cosine_similarity(query_embedding, triple_embedding)
    normalized_dde = normalize_dde_value(dde_value)
    return 0.7 * cosine_sim + 0.3 * normalized_dde
```

## Performance Characteristics

### Advantages
- **Speed**: Shallow network enables fast inference
- **Quality**: Trained on QA-specific relevance data
- **Robustness**: Fallback mechanism ensures system availability

### Limitations
- **Domain Specificity**: Trained on specific QA datasets
- **Static Features**: No dynamic context adaptation
- **Binary Classification**: Simple relevance scoring

## Error Handling

The system gracefully handles MLP-related errors:

1. **Missing Model**: Falls back to heuristic scoring
2. **Dimension Mismatch**: Logs error and uses fallback
3. **Inference Errors**: Catches exceptions and continues with heuristics

## Monitoring and Logging

Key log messages for MLP operations:
- Model loading success/failure
- Scoring method used (MLP vs heuristic)
- Feature dimension validation
- Error conditions and fallbacks

## Future Improvements

Potential enhancements for the MLP retriever:

1. **Fine-tuning**: Adapt model to specific domains
2. **Feature Engineering**: Additional graph-based features
3. **Dynamic Embeddings**: Context-aware embeddings
4. **Ensemble Methods**: Combine multiple scoring approaches
5. **Online Learning**: Adapt based on user feedback 