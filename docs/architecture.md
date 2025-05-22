# SubgraphRAG+ Architecture Documentation

## Architecture Overview

SubgraphRAG+ is built as a modular system with a clear separation of concerns. This document outlines the architectural components, their interactions, and the design decisions that informed the system's structure.

## System Components

### 1. Core Components

![Architecture Diagram](../docs/images/architecture.png)

#### 1.1 API Layer (`app/api.py`)
- FastAPI-based RESTful API with SSE streaming
- Endpoint handlers for queries, graph browsing, ingestion, and monitoring
- Authentication and rate limiting middleware
- Error handling and request validation

#### 1.2 Data Models (`app/models.py`)
- Dataclass-based domain models (Triple, Entity, GraphNode, etc.)
- Request/response models for API validation
- Custom error types

#### 1.3 Configuration (`app/config.py`)
- JSON schema-based configuration validation
- Environment variable integration
- Path helpers and global constants

#### 1.4 Database Layer (`app/database.py`)
- Neo4j connection management and query execution
- SQLite connection management for staging and caching
- Connection pooling and transaction handling

### 2. Retrieval Components

#### 2.1 Main Retriever (`app/retriever.py`)
- Hybrid retrieval orchestration
- FAISS index management
- MLP scorer integration
- Candidate fusion and subgraph assembly

#### 2.2 Entity Linking (`app/utils.py`)
- Entity extraction from queries
- Entity linking to knowledge graph
- Alias resolution and fuzzy matching
- DDE (Directional-Distance Encoding) computation

#### 2.3 Machine Learning Components (`app/ml/`)
- Embedder (`app/ml/embedder.py`) - Text embedding with multiple backends
- LLM (`app/ml/llm.py`) - LLM integration with multiple backends
- MLP integration for triple scoring

### 3. Auxiliary Components

#### 3.1 Verification (`app/verify.py`)
- LLM output validation
- Citation verification
- Security checks

#### 3.2 Scripts (`scripts/`)
- Migration scripts
- Ingestion worker
- FAISS index management
- Demo and quickstart

## Data Flow

### 1. Query Processing Flow

```
User Query -> API Endpoint -> Entity Extraction -> Entity Linking -> 
Hybrid Retrieval (Graph + Vector) -> MLP Scoring -> Subgraph Assembly ->
LLM Answer Generation -> Answer Verification -> Streaming Response
```

### 2. Ingest Flow

```
API Endpoint -> Validation -> SQLite Staging -> Ingest Worker -> 
Neo4j Transaction -> Embedding Computation -> FAISS Staging -> 
Periodic FAISS Merge
```

## Core Data Structures

### 1. Triple
The fundamental unit of the knowledge graph: (head, relation, tail)

```python
@dataclass
class Triple:
    id: str               # Unique identifier
    head_id: str          # Head entity ID
    head_name: str        # Head entity name
    relation_id: str      # Relation ID
    relation_name: str    # Relation name
    tail_id: str          # Tail entity ID
    tail_name: str        # Tail entity name
    properties: dict      # Additional properties
    embedding: ndarray    # Vector embedding
    relevance_score: float # Relevance to query
```

### 2. GraphData
D3.js compatible visualization structure

```python
@dataclass
class GraphData:
    nodes: List[GraphNode]
    links: List[GraphLink]
    relevant_paths: List[Dict]
```

## Critical Path Analysis

The most performance-critical components of the system are:

1. **Entity Linking**: Initial bottleneck for query understanding
2. **FAISS Search**: Vector search performance scales with index size
3. **Subgraph Assembly**: Graph algorithms for optimal triple selection
4. **LLM Integration**: External API calls or local model inference

## Storage Architecture

### 1. Neo4j Graph Database
- Entities stored as nodes with properties
- Relationships store connection information
- Indexes on entity ID, name, and type
- Full text search index on name and aliases

### 2. FAISS Vector Index
- Triple embeddings indexed for semantic search
- ID mapping between FAISS IDs and Neo4j relation IDs
- Configurable quantization for space efficiency
- Periodic retraining for optimal clustering

### 3. SQLite Database
- Staging table for triple ingestion
- API keys and authentication data
- Feedback storage
- Local caching information

### 4. Disk Cache
- Embedding cache for frequently used entities
- DDE (Directional-Distance Encoding) cache
- LRU eviction policy with TTL

## Design Decisions

### 1. Hybrid Retrieval Strategy
**Decision**: Combine graph traversal with vector search
**Rationale**: Graph traversal provides precision but limited recall; vector search provides recall but limited precision. The combination leverages strengths of both.
**Trade-offs**: Increased complexity, but significantly improved retrieval quality

### 2. SQLite Staging for Ingestion
**Decision**: Use SQLite as an intermediate staging area
**Rationale**: Provides transactional safety, deduplication, and resilience to ingestion failures
**Trade-offs**: Slightly increased latency for ingestion, but improved reliability

### 3. Backend Abstraction
**Decision**: Abstract LLM and embedding backends
**Rationale**: Support for multiple backends (OpenAI, local models, Hugging Face) without code changes
**Trade-offs**: Some overhead from abstraction, but significant flexibility gains

### 4. SSE Streaming
**Decision**: Use Server-Sent Events for streaming
**Rationale**: Provides token-by-token streaming with event typing for rich client experiences
**Trade-offs**: More complex client handling, but improved UX

## Scalability Considerations

### 1. Vertical Scaling
- FAISS index can handle millions of triples on a single machine
- Neo4j can scale to tens of millions of nodes/relationships on a single instance
- Token budget-aware subgraph assembly manages LLM context windows

### 2. Horizontal Scaling (Future)
- Read replicas for Neo4j
- Distributed FAISS with sharding
- API load balancing
- Separate ingestion and query services

### 3. Performance Bottlenecks
- LLM inference latency (mitigated by streaming)
- FAISS index size (mitigated by quantization)
- Neo4j query complexity (mitigated by optimized queries and indexes)

## Security Architecture

### 1. Authentication
- API key-based authentication
- Key storage in environment variables
- Brute force prevention with rate limiting

### 2. Data Protection
- Input validation for all endpoints
- LLM output validation to prevent injection
- Citation verification for answer trustworthiness

### 3. Monitoring
- Comprehensive logging
- Prometheus metrics
- Health and readiness probes

## Future Architecture Extensions

### 1. Multitenancy
- Tenant isolation in Neo4j (tenant_id property)
- Separate FAISS indices per tenant
- API key to tenant mapping

### 2. Advanced Monitoring
- Request tracing
- Performance profiling
- Automated alerting

### 3. Continuous Learning
- Feedback-driven model fine-tuning
- Automated evaluation pipeline
- A/B testing infrastructure

## Conclusion

The SubgraphRAG+ architecture provides a robust foundation for knowledge graph question answering with hybrid retrieval. The modular design enables future extensions while maintaining high performance and reliability.