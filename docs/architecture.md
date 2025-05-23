# SubgraphRAG+ Architecture

This document provides a detailed technical overview of the SubgraphRAG+ system architecture, focusing on component interactions, data flow, and design decisions.

## System Overview

SubgraphRAG+ implements a hybrid retrieval architecture that combines graph traversal with vector search to provide accurate, contextual answers with explanatory visualizations.

![Architecture Diagram](../docs/images/architecture.png)

## Core Components

### 1. API Layer (`src/app/api.py`)
- **FastAPI endpoints** with OpenAPI documentation
- **SSE streaming** for real-time response delivery
- **Authentication** via API keys
- **Rate limiting** and request validation
- **Health/readiness probes** for monitoring

### 2. Data Models (`src/app/models.py`)
- **Pydantic models** for request/response validation
- **Domain objects** (Triple, GraphData, QueryRequest)
- **Type safety** and automatic serialization
- **JSON schema** generation for API docs

### 3. Configuration Management (`src/app/config.py`)
- **JSON schema validation** for config files
- **Environment variable** handling
- **Backend abstraction** for different LLM providers
- **Runtime configuration** updates

### 4. Database Layer (`src/app/database.py`)
- **Neo4j connection** management with connection pooling
- **SQLite staging** for ingestion queue and auth
- **Transaction handling** with rollback support
- **Connection health monitoring**

## Retrieval Architecture

### 1. Hybrid Retriever (`src/app/retriever.py`)
The core retrieval engine implements a two-stage process:

```mermaid
graph LR
    A[Query] --> B[Entity Extraction]
    B --> C[Entity Linking]
    C --> D[Graph Traversal]
    C --> E[Vector Search]
    D --> F[Subgraph Assembly]
    E --> F
    F --> G[MLP Scoring]
    G --> H[Context Selection]
```

**Stage 1: Candidate Retrieval**
- **Entity extraction** from natural language queries
- **Fuzzy matching** to link entities to knowledge graph
- **Graph traversal** using Neo4j Cypher queries
- **Vector search** using FAISS for semantic similarity

**Stage 2: Relevance Scoring**
- **MLP scoring model** for triple relevance
- **Subgraph assembly** with path-based ranking
- **Context window optimization** within token budget

### 2. Entity Linking (`src/app/utils.py`)
- **Named entity recognition** using spaCy
- **Fuzzy string matching** with configurable thresholds
- **Graph-based disambiguation** using node connectivity
- **Caching** for frequent entity lookups

### 3. ML Components (`src/app/ml/`)
- **Text embedding** with multiple backend support
- **LLM integration** (OpenAI, HuggingFace, MLX)
- **Triple scoring** with pre-trained MLP models
- **Batch processing** for efficiency

## Data Flow Architecture

### Query Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Retriever
    participant Neo4j
    participant FAISS
    participant LLM
    
    Client->>API: POST /query
    API->>Retriever: process_query()
    Retriever->>Neo4j: entity_linking()
    Retriever->>Neo4j: graph_traversal()
    Retriever->>FAISS: vector_search()
    Retriever->>Retriever: subgraph_assembly()
    Retriever->>LLM: generate_answer()
    LLM-->>API: SSE stream
    API-->>Client: streamed response
```

### Ingestion Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant SQLite
    participant Worker
    participant Neo4j
    participant FAISS
    
    Client->>API: POST /ingest
    API->>SQLite: stage_triples()
    Worker->>SQLite: fetch_pending()
    Worker->>Neo4j: create_nodes_relations()
    Worker->>FAISS: compute_embeddings()
    Worker->>SQLite: mark_processed()
    Note over FAISS: Periodic merge
```

## Storage Architecture

### 1. Neo4j Knowledge Graph
- **Nodes**: Entities with properties and labels
- **Relationships**: Typed connections between entities
- **Indexes**: Optimized for entity name lookups
- **Constraints**: Ensure data integrity

```cypher
// Example schema
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type);
```

### 2. FAISS Vector Index
- **Triple embeddings** for semantic search
- **Quantization** for memory efficiency
- **Staging area** for incremental updates
- **Periodic merging** to maintain performance

### 3. SQLite Staging Database
- **Ingestion queue** with status tracking
- **API key management** with rate limiting
- **Error logging** for failed operations
- **Feedback storage** for model improvement

## Performance Optimizations

### 1. Caching Strategy
- **Entity linking cache** (LRU eviction)
- **DDE feature cache** for graph patterns
- **Embedding cache** for computed vectors
- **Query result cache** for frequent patterns

### 2. Indexing Strategy
- **Neo4j indexes** on entity names and IDs
- **FAISS quantization** for large-scale vector search
- **SQLite indexes** on staging table status
- **Composite indexes** for complex queries

### 3. Concurrency Model
- **Async/await** for I/O operations
- **Connection pooling** for database connections
- **Background workers** for ingestion processing
- **Rate limiting** to prevent resource exhaustion

## Scalability Considerations

### Current Capabilities
- **FAISS**: Millions of triples with quantization
- **Neo4j**: Tens of millions of nodes/relationships
- **Concurrent users**: Limited by LLM backend latency
- **Ingestion rate**: ~1000 triples/second

### Scaling Strategies
- **Horizontal scaling**: Neo4j read replicas
- **Distributed FAISS**: Sharded vector indexes
- **API load balancing**: Multiple service instances
- **Caching layers**: Redis for shared state

## Security Architecture

### 1. Authentication & Authorization
- **API key-based** authentication
- **Environment variable** storage for secrets
- **Rate limiting** per API key
- **Request validation** with Pydantic

### 2. Input Validation
- **Schema validation** for all inputs
- **SQL injection prevention** via parameterized queries
- **XSS protection** in API responses
- **File upload restrictions** for ingestion

### 3. Monitoring & Logging
- **Structured logging** with correlation IDs
- **Health checks** for all dependencies
- **Metrics collection** for performance monitoring
- **Error tracking** with stack traces

## Extension Points

### 1. Backend Abstraction
The system supports multiple backends through a common interface:

```python
class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        pass
```

### 2. Plugin Architecture
- **Custom retrievers** via inheritance
- **Additional data sources** through adapters
- **Custom scoring models** with standardized interfaces
- **Visualization backends** for different frontends

### 3. Future Enhancements
- **Multi-tenant isolation** with namespace support
- **Federated search** across multiple knowledge graphs
- **Real-time learning** from user feedback
- **Advanced reasoning** with symbolic AI integration

This modular architecture enables incremental improvements while maintaining system stability and performance.