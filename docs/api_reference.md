# API Reference

This document provides detailed technical specifications for the SubgraphRAG+ REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints require authentication via the `X-API-KEY` header:

```bash
curl -H "X-API-KEY: your_api_key_here" http://localhost:8000/endpoint
```

## Core Endpoints

### Query Endpoint

**POST** `/query`

Main question-answering endpoint with Server-Sent Events (SSE) streaming.

#### Request Body

```json
{
  "question": "string",
  "visualize_graph": true,
  "max_tokens": 500,
  "temperature": 0.7,
  "include_citations": true,
  "stream": true
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | Yes | - | Natural language question |
| `visualize_graph` | boolean | No | `false` | Include graph visualization data |
| `max_tokens` | integer | No | `500` | Maximum tokens in response |
| `temperature` | float | No | `0.7` | LLM temperature (0.0-1.0) |
| `include_citations` | boolean | No | `true` | Include source citations |
| `stream` | boolean | No | `true` | Enable SSE streaming |

#### Response (SSE Stream)

The endpoint returns Server-Sent Events with the following event types:

```
event: token
data: {"token": "Hello"}

event: citation
data: {"triple_id": "123", "relevance": 0.95}

event: graph
data: {"nodes": [...], "links": [...]}

event: complete
data: {"total_tokens": 150, "processing_time": 2.3}
```

#### Response (Non-streaming)

```json
{
  "answer": "Tesla was founded by Elon Musk in 2003...",
  "citations": [
    {
      "triple_id": "tesla_founder_123",
      "head": "Tesla",
      "relation": "founded_by",
      "tail": "Elon Musk",
      "relevance_score": 0.95
    }
  ],
  "graph_data": {
    "nodes": [
      {
        "id": "tesla",
        "name": "Tesla",
        "type": "Company",
        "relevance": 0.95
      }
    ],
    "links": [
      {
        "source": "tesla",
        "target": "elon_musk",
        "relation": "founded_by",
        "strength": 0.95
      }
    ]
  },
  "metadata": {
    "processing_time": 2.3,
    "total_tokens": 150,
    "retrieved_triples": 25
  }
}
```

### Graph Browse Endpoint

**GET** `/graph/browse`

Browse the knowledge graph with pagination and filtering.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entity` | string | No | - | Filter by entity name |
| `relation` | string | No | - | Filter by relation type |
| `limit` | integer | No | `50` | Number of results (max 1000) |
| `offset` | integer | No | `0` | Pagination offset |
| `include_properties` | boolean | No | `false` | Include node/edge properties |

#### Response

```json
{
  "triples": [
    {
      "id": "triple_123",
      "head": "Tesla",
      "head_id": "tesla_entity",
      "relation": "founded_by",
      "relation_id": "founded_by_rel",
      "tail": "Elon Musk",
      "tail_id": "elon_musk_entity",
      "properties": {
        "year": "2003",
        "location": "Palo Alto"
      }
    }
  ],
  "pagination": {
    "total": 1250,
    "limit": 50,
    "offset": 0,
    "has_next": true
  }
}
```

### Ingestion Endpoint

**POST** `/ingest`

Batch ingest knowledge graph triples.

#### Request Body

```json
{
  "triples": [
    {
      "head": "Tesla",
      "relation": "founded_by",
      "tail": "Elon Musk",
      "head_name": "Tesla Inc.",
      "relation_name": "founded by",
      "tail_name": "Elon Musk",
      "properties": {
        "year": "2003",
        "confidence": 0.95
      }
    }
  ],
  "source": "manual_input",
  "batch_id": "batch_20240101_001"
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `triples` | array | Yes | Array of triple objects |
| `source` | string | No | Data source identifier |
| `batch_id` | string | No | Batch identifier for tracking |

#### Triple Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `head` | string | Yes | Head entity identifier |
| `relation` | string | Yes | Relation type |
| `tail` | string | Yes | Tail entity identifier |
| `head_name` | string | No | Human-readable head name |
| `relation_name` | string | No | Human-readable relation name |
| `tail_name` | string | No | Human-readable tail name |
| `properties` | object | No | Additional properties |

#### Response

```json
{
  "status": "accepted",
  "batch_id": "batch_20240101_001",
  "queued_count": 150,
  "duplicate_count": 5,
  "error_count": 0,
  "estimated_processing_time": "2-5 minutes"
}
```

### Feedback Endpoint

**POST** `/feedback`

Submit feedback on query responses for model improvement.

#### Request Body

```json
{
  "query_id": "query_123",
  "rating": 4,
  "feedback_type": "accuracy",
  "comments": "Good answer but missing recent information",
  "helpful_citations": ["triple_456", "triple_789"],
  "unhelpful_citations": ["triple_123"]
}
```

#### Response

```json
{
  "status": "recorded",
  "feedback_id": "feedback_456"
}
```

## System Endpoints

### Health Check

**GET** `/healthz`

Basic health check endpoint.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Readiness Check

**GET** `/readyz`

Comprehensive readiness check including dependencies.

#### Response

```json
{
  "status": "ready",
  "checks": {
    "neo4j": "connected",
    "faiss_index": "loaded",
    "llm_backend": "available"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Metrics

**GET** `/metrics`

Prometheus-compatible metrics endpoint.

#### Response

```
# HELP subgraphrag_queries_total Total number of queries processed
# TYPE subgraphrag_queries_total counter
subgraphrag_queries_total 1234

# HELP subgraphrag_query_duration_seconds Query processing time
# TYPE subgraphrag_query_duration_seconds histogram
subgraphrag_query_duration_seconds_bucket{le="1.0"} 100
subgraphrag_query_duration_seconds_bucket{le="5.0"} 200
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Question parameter is required",
    "details": {
      "field": "question",
      "constraint": "non_empty_string"
    }
  },
  "request_id": "req_123456"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Dependency unavailable |

## Rate Limiting

- **Default limit**: 60 requests per minute per API key
- **Headers included** in responses:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## WebSocket Support

For real-time applications, WebSocket connections are available:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  type: 'query',
  data: { question: 'Who founded Tesla?' }
}));
```

## SDK Examples

### Python

```python
import requests

response = requests.post(
    'http://localhost:8000/query',
    headers={'X-API-KEY': 'your_key'},
    json={'question': 'Who founded Tesla?'}
)
```

### JavaScript

```javascript
fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your_key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    question: 'Who founded Tesla?'
  })
});
```

### cURL

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: your_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?"}'
```