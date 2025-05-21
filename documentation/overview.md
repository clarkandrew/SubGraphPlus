# SubgraphRAG+ System Overview

## Introduction

SubgraphRAG+ is a comprehensive knowledge graph question answering system that enhances the original SubgraphRAG research with production-ready features. This document provides a high-level overview of the system architecture, key components, and information flow.

## System Architecture

SubgraphRAG+ follows a microservice-oriented architecture with the following main components:

1. **API Layer**: FastAPI-based REST API with streaming capabilities
2. **Data Layer**: Neo4j graph database and SQLite staging database
3. **Retrieval Layer**: Hybrid graph + vector search with FAISS
4. **Reasoning Layer**: LLM-powered answer generation with verifiable citations
5. **Visualization Layer**: D3.js compatible graph data output

### Architecture Diagram

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  API Layer    │    │ Retrieval     │    │  Storage      │
│  - FastAPI    │◄──►│ - FAISS Index │◄──►│  - Neo4j      │
│  - SSE Stream │    │ - Graph Search│    │  - SQLite     │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │                    ▲                    ▲
        ▼                    │                    │
┌───────────────┐            │                    │
│ Reasoning     │            │                    │
│ - LLM (OpenAI,│            │                    │
│   MLX, HF)    │────────────┘                    │
└───────┬───────┘                                 │
        │                                         │
        ▼                                         │
┌───────────────┐                                 │
│ Ingestion     │─────────────────────────────────┘
│ - Staging     │
│ - Processing  │
└───────────────┘
```

## Key Components

### 1. API Layer

The API layer is built using FastAPI and provides the following endpoints:

- `/query`: Main Q&A endpoint with SSE streaming
- `/graph/browse`: Paginated knowledge graph explorer
- `/ingest`: Batch triple ingestion
- `/feedback`: User feedback collection
- `/healthz`, `/readyz`, `/metrics`: Monitoring endpoints

### 2. Data Layer

#### Neo4j Graph Database
- Stores the knowledge graph as entities and relationships
- Provides graph traversal capabilities for structured retrieval
- Schema supports entity aliasing and type information

#### SQLite Staging Database
- Stages incoming triples before Neo4j ingestion
- Provides deduplication and transactional safety
- Stores API keys, feedback, and audit logs

### 3. Retrieval Layer

#### Hybrid Retrieval
SubgraphRAG+ implements a hybrid retrieval approach combining:

1. **Graph-based Retrieval**: Structured traversal using the Neo4j graph database
2. **Dense Vector Retrieval**: Semantic search using FAISS
3. **MLP Scoring**: Triple scoring using the pre-trained MLP model from SubgraphRAG

#### Retrieval Process Flow
1. Extract entities from user query
2. Link entities to the knowledge graph
3. Perform graph retrieval around linked entities
4. Perform dense vector retrieval using query embedding
5. Score all candidates using MLP or heuristic fallback
6. Merge and select final subgraph based on scores
7. Apply token budget constraints using greedy connection algorithm

### 4. Reasoning Layer

The reasoning layer uses an LLM to generate answers based on the retrieved subgraph:

- **Abstracted Backend**: Support for OpenAI, Hugging Face models, and MLX (for Apple Silicon)
- **Prompted Reasoning**: Structured prompt engineering to encourage factual answers
- **Citation Verification**: Output validation to ensure answers only cite provided facts
- **Streaming Output**: Token-by-token streaming for responsive UX

### 5. Visualization Layer

The system provides D3.js compatible graph data for visualization:

- Subgraph with entity nodes and relationship edges
- Relevance scores for nodes and edges
- Inclusion reasons to explain why each element is shown
- Path information for multi-hop reasoning

## Key Differentiators

SubgraphRAG+ enhances the original SubgraphRAG with several key improvements:

1. **Dynamic Knowledge Graph**: Support for real-time triple ingestion and updates
2. **Hybrid Retrieval**: Combined graph-structured and dense embedding search
3. **Token Budget Awareness**: Smart subgraph construction within LLM token limits
4. **Enterprise-Grade API**: Full authentication, error handling, and monitoring
5. **SSE Streaming**: Responsive UX with token-by-token streaming

## Data Flow

### Query Processing Flow
1. User submits question via `/query` endpoint
2. System extracts and links entities in the question
3. Hybrid retrieval combines graph and dense vector search
4. Retrieved triples are scored and filtered
5. LLM generates answer with citations
6. Answer, citations, and graph data are streamed to client

### Ingestion Flow
1. User submits triples via `/ingest` endpoint
2. Triples are validated and staged in SQLite
3. Ingest worker processes batches of triples
4. Neo4j graph is updated with new entities and relationships
5. Triple embeddings are computed and stored in staging
6. FAISS index is periodically updated with new embeddings

## Deployment Considerations

### Resource Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 16GB+ recommended (8GB minimum)
- **Storage**: 20GB minimum
- **GPU**: Optional, beneficial for local LLM inference

### Required Services
- **Neo4j**: Graph database with APOC plugin
- **FAISS**: Vector index for semantic search
- **Language Models**: Local models or API access

## Monitoring and Maintenance

SubgraphRAG+ includes built-in monitoring and maintenance capabilities:

- **Health Checks**: `/healthz` and `/readyz` endpoints
- **Prometheus Metrics**: `/metrics` endpoint for scraping
- **Audit Logging**: Request logging for security and debugging
- **FAISS Index Rebuilding**: Periodic optimization of vector index
- **Database Reconciliation**: Consistency checking between Neo4j and FAISS

## Conclusion

SubgraphRAG+ provides a scalable, production-ready implementation of graph-RAG technology. By combining structured graph knowledge with dense embeddings and LLM reasoning, it offers improved accuracy, controllability, and explainability compared to traditional RAG systems.

This overview document provides a high-level understanding of the system architecture. For more detailed information, please refer to the specific component documentation and the API specification.