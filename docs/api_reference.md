# SubgraphRAG+ API Reference

This document provides a concise reference for the SubgraphRAG+ API, including all endpoints, parameters, and response formats.

## Base URL & Authentication

- **Base URL**: `http://localhost:8000`
- **Authentication**: All endpoints (except health checks) require the `X-API-KEY` header
  ```
  X-API-KEY: your_api_key_here
  ```
- **API Key Configuration**: Set via the `API_KEY_SECRET` environment variable

## Endpoints

### 1. Query Endpoint (`POST /query`)

Submit a question to be answered using the knowledge graph.

**Request:**
```json
{
  "question": "Who founded Tesla?",
  "visualize_graph": true
}
```

**Parameters:**
- `question` (string, required): The question to answer
- `visualize_graph` (boolean, optional): Include graph visualization data (default: true)

**Response:**
Server-Sent Events (SSE) stream with the following event types:

| Event Type | Description | Example |
|------------|-------------|---------|
| `llm_token` | LLM response tokens | `{"token": "Elon "}` |
| `citation_data` | Citation information | `{"id": "rel123", "text": "Elon Musk -[founded]-> Tesla Inc."}` |
| `metadata` | Query metadata | `{"trust_score": 0.92, "latency_ms": 1530, "query_id": "q_abc123"}` |
| `graph_data` | Visualization data | `{"nodes": [...], "links": [...], "relevant_paths": [...]}` |
| `error` | Error information | `{"code": "NO_ENTITY_MATCH", "message": "No entities found matching the query terms."}` |
| `end` | Stream end marker | `{"message": "Stream ended"}` |

**Error Codes:**
- 400: `EMPTY_QUERY` - Query is empty
- 401: `UNAUTHORIZED` - Invalid/missing API key
- 404: `NO_ENTITY_MATCH` - No matching entities found
- 409: `AMBIGUOUS_ENTITIES` - Ambiguous entity references
- 500: `INTERNAL_ERROR` - Server error

### 2. Graph Browse Endpoint (`GET /graph/browse`)

Browse the knowledge graph with pagination and filtering.

**Parameters:**
- `page` (integer, optional): Page number (default: 1)
- `limit` (integer, optional): Results per page (default: 500, max: 2000)
- `node_types_filter` (string[], optional): Filter by node types
- `relation_types_filter` (string[], optional): Filter by relation types
- `center_node_id` (string, optional): ID of node to center graph on
- `hops` (integer, optional): Hops from center node (default: 1, max: 3)
- `search_term` (string, optional): Text search term (min 3 chars)

**Response:**
```json
{
  "nodes": [
    {
      "id": "entity123",
      "name": "Eiffel Tower",
      "type": "Landmark",
      "properties": { "description": "A famous Paris landmark." }
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

### 3. Ingest Endpoint (`POST /ingest`)

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
- `triples` (array, required): Array of triples to ingest
  - `head` (string, required): Head entity name
  - `relation` (string, required): Relation name
  - `tail` (string, required): Tail entity name

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
- 400: `EMPTY_TRIPLES` - No triples provided
- 400: `VALIDATION_ERROR` - Invalid triples format
- 401: `UNAUTHORIZED` - Invalid or missing API key

### 4. Feedback Endpoint (`POST /feedback`)

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
- `query_id` (string, required): ID of the query from metadata
- `is_correct` (boolean, required): Whether the answer was correct
- `comment` (string, optional): User comment/feedback
- `expected_answer` (string, optional): Expected answer if incorrect

**Response:**
```json
{
  "status": "accepted",
  "message": "Feedback recorded successfully"
}
```

### 5. Health & Monitoring Endpoints

**Health Check (`GET /healthz`):**
- Basic liveness probe
- Response: `{"status": "ok"}`

**Readiness Check (`GET /readyz`):**
- Dependency readiness probe
- Success response (200):
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
- Error response (503):
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

**Metrics (`GET /metrics`):**
- Prometheus metrics endpoint
- Returns plain text in Prometheus format:
  ```
  http_requests_total{method="POST",endpoint="/query",status="200"} 10.0
  http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="0.1"} 0.0
  http_request_duration_seconds_bucket{method="POST",endpoint="/query",le="0.5"} 2.0
  ...
  ```

## API Usage Details

### Error Handling

All endpoints use a consistent error format:
```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": { "additional": "information" }
}
```

**Standard HTTP Status Codes:**
- 200: Success
- 202: Accepted (async operations)
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 409: Conflict
- 422: Unprocessable Entity
- 429: Too Many Requests
- 500: Internal Server Error
- 503: Service Unavailable

### Python Client Example

```python
import requests
import json
import sseclient

# Configuration
api_key = "your_api_key_here"
base_url = "http://localhost:8000"
headers = {
    "Content-Type": "application/json",
    "X-API-KEY": api_key
}

# Query example
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
```

### Additional Information

- **Rate Limiting**: 60 requests per minute (configurable)
- **Versioning**: Current version is v1 (future versions will use `/v2/endpoint` format)
- **Authentication**: API key must be passed via `X-API-KEY` header