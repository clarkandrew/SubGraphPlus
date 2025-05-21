# SubgraphRAG+ API Reference

## Introduction

This document provides a comprehensive reference for the SubgraphRAG+ API. All endpoints, parameters, and response formats are detailed here.

## Base URL

All endpoints are relative to the base URL of your SubgraphRAG+ instance, typically:

```
http://localhost:8000
```

## Authentication

All endpoints (except health checks) require API key authentication using the `X-API-KEY` header:

```
X-API-KEY: your_api_key_here
```

The API key is set as an environment variable `API_KEY_SECRET` when starting the server.

## Endpoints

### 1. Query Endpoint

#### `POST /query`

Submit a question to be answered using the knowledge graph.

**Request:**

```json
{
  "question": "Who founded Tesla?",
  "visualize_graph": true
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question to answer |
| `visualize_graph` | boolean | No | Whether to include graph visualization data (default: true) |

**Response:**

Server-Sent Events (SSE) stream with the following event types:

- `llm_token`: Individual tokens from the LLM response
  ```json
  {"token": "Elon "}
  ```

- `citation_data`: Citation information for facts used in the answer
  ```json
  {"id": "rel123", "text": "Elon Musk -[founded]-> Tesla Inc."}
  ```

- `metadata`: Additional information about the query
  ```json
  {
    "trust_score": 0.92,
    "latency_ms": 1530,
    "query_id": "q_abc123"
  }
  ```

- `graph_data`: Visualization data for the subgraph
  ```json
  {
    "nodes": [...],
    "links": [...],
    "relevant_paths": [...]
  }
  ```

- `error`: Error information if something goes wrong
  ```json
  {
    "code": "NO_ENTITY_MATCH",
    "message": "No entities found matching the query terms."
  }
  ```

- `end`: Indicates the end of the stream
  ```json
  {"message": "Stream ended"}
  ```

**Error Codes:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `EMPTY_QUERY` | Query is empty |
| 401 | `UNAUTHORIZED` | Invalid or missing API key |
| 404 | `NO_ENTITY_MATCH` | No entities matched in the knowledge graph |
| 409 | `AMBIGUOUS_ENTITIES` | Query contains ambiguous entity references |
| 500 | `INTERNAL_ERROR` | Server error |

### 2. Graph Browse Endpoint

#### `GET /graph/browse`

Browse the knowledge graph with pagination and filtering.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page` | integer | No | 1 | Page number |
| `limit` | integer | No | 500 | Results per page (max 2000) |
| `node_types_filter` | string[] | No | null | Filter by node types |
| `relation_types_filter` | string[] | No | null | Filter by relation types |
| `center_node_id` | string | No | null | ID of node to center graph on |
| `hops` | integer | No | 1 | Number of hops from center node (max 3) |
| `search_term` | string | No | null | Text search term (min 3 chars) |

**Response:**

```json
{
  "nodes": [
    {
      "id": "entity123",
      "name": "Eiffel Tower",
      "type": "Landmark",
      "properties": {
        "description": "A famous Paris landmark."
      }
    }
  ],
  "links": [
    {
      "source": "entity123",
      "target": "entity456",
      "relation_id": "rel456",
      "relation_name": "locatedIn",
      "properties": {}
    }
  ],
  "page": 1,
  "limit": 500,
  "total_nodes_in_filter": 12000,
  "total_links_in_filter": 34000,
  "has_more": true
}
```

### 3. Ingest Endpoint

#### `POST /ingest`

Ingest triples into the knowledge graph.

**Request:**

```json
{
  "triples": [
    {"head": "OpenAI", "relation": "foundedBy", "tail": "Sam Altman"},
    {"head": "Elon Musk", "relation": "founded", "tail": "Tesla Inc."}
  ]
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `triples` | array | Yes | Array of triples to ingest |
| `triples[].head` | string | Yes | Head entity name |
| `triples[].relation` | string | Yes | Relation name |
| `triples[].tail` | string | Yes | Tail entity name |

**Response:**

```json
{
  "status": "accepted",
  "triples_staged": 2,
  "errors": [],
  "message": "Staged 2 triples for ingestion"
}
```

**Error Codes:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `EMPTY_TRIPLES` | No triples provided |
| 400 | `VALIDATION_ERROR` | Invalid triples format |
| 401 | `UNAUTHORIZED` | Invalid or missing API key |

### 4. Feedback Endpoint

#### `POST /feedback`

Submit feedback on query results.

**Request:**

```json
{
  "query_id": "q_abc123",
  "is_correct": true,
  "comment": "Very helpful",
  "expected_answer": "Elon Musk"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_id` | string | Yes | ID of the query from metadata |
| `is_correct` | boolean | Yes | Whether the answer was correct |
| `comment` | string | No | User comment/feedback |
| `expected_answer` | string | No | Expected answer if incorrect |

**Response:**

```json
{
  "status": "accepted",
  "message": "Feedback recorded successfully"
}
```

### 5. Health Check Endpoints

#### `GET /healthz`

Basic liveness probe.

**Response:**

```json
{
  "status": "ok"
}
```

#### `GET /readyz`

Dependency readiness probe.

**Response:**

```json
{
  "status": "ready",
  "checks": {
    "sqlite": "ok",
    "neo4j": "ok",
    "faiss_index": "ok",
    "llm_backend": "ok"
  }
}
```

**Error Response (503):**

```json
{
  "status": "not_ready",
  "checks": {
    "sqlite": "ok",
    "neo4j": "failed",
    "faiss_index": "ok",
    "llm_backend": "ok"
  }
}
```

### 6. Metrics Endpoint

#### `GET /metrics`

Prometheus metrics.

**Response:**

Plain text in Prometheus format:

```
http_requests_total{method="POST",endpoint="/query",status="200"} 10.0
http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="0.1"} 0.0
http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="0.5"} 2.0
http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="1.0"} 6.0
http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="5.0"} 10.0
http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="+Inf"} 10.0
...
```

## Error Handling

All endpoints use consistent error response format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "additional": "information"
  }
}
```

Standard HTTP status codes are used:

- 200: Success
- 202: Accepted (for asynchronous operations)
- 400: Bad Request (client error)
- 401: Unauthorized (invalid API key)
- 404: Not Found (entity not found)
- 409: Conflict (ambiguous entities)
- 422: Unprocessable Entity (valid request but cannot be processed)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error
- 503: Service Unavailable (dependency unavailable)

## API Client Example

### Python Example

```python
import requests
import json
import sseclient

# API key
api_key = "your_api_key_here"

# Base URL
base_url = "http://localhost:8000"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-KEY": api_key
}

# Submit a query
def query(question):
    response = requests.post(
        f"{base_url}/query",
        headers=headers,
        json={"question": question},
        stream=True
    )
    
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event == "llm_token":
            data = json.loads(event.data)
            print(data["token"], end="")
        elif event.event == "error":
            data = json.loads(event.data)
            print(f"Error: {data['message']}")
            break
        elif event.event == "end":
            print("\nStream ended")
            break

# Browse graph
def browse_graph(page=1, limit=50):
    response = requests.get(
        f"{base_url}/graph/browse?page={page}&limit={limit}",
        headers=headers
    )
    return response.json()

# Ingest triples
def ingest_triples(triples):
    response = requests.post(
        f"{base_url}/ingest",
        headers=headers,
        json={"triples": triples}
    )
    return response.json()
```

## Rate Limiting

API requests are rate-limited based on the `API_RATE_LIMIT` configuration (default: 60 requests per minute). Exceeding this limit will result in a 429 status code.

## Versioning

The current API version is v1. Future versions will be made available under versioned endpoints (e.g., `/v2/query`).