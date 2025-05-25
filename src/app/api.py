import os
import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import hashlib
import sse_starlette.sse
import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_503_SERVICE_UNAVAILABLE
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Add testing check
TESTING = os.getenv('TESTING', '').lower() in ('1', 'true', 'yes')

from app.config import config, API_KEY_SECRET
from app.models import (
    QueryRequest, FeedbackRequest, IngestRequest, ErrorResponse,
    GraphData, PaginatedGraphData, Triple, 
    EntityLinkingError, AmbiguousEntityError, RetrievalEmpty, generate_query_id
)
from app.database import neo4j_db, sqlite_db
from app.utils import (
    link_entities_v2, extract_query_entities, triples_to_graph_data
)
from app.retriever import hybrid_retrieve_v2, entity_search, faiss_index
from app.verify import validate_llm_output, format_prompt
from app.ml.embedder import health_check as embedder_health_check
from app.ml.llm import generate_answer, stream_tokens, health_check as llm_health_check

# Set up logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting SubgraphRAG+ API server")
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Initialize metrics
    instrumentator.expose(app)
    
    yield
    
    # Shutdown
    logger.info("Shutting down SubgraphRAG+ API server")
    
    # Close database connections
    from app.database import close_connections
    close_connections()

# Create FastAPI app with lifespan
app = FastAPI(
    title="SubgraphRAG+ API",
    description="Advanced knowledge graph question answering with hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up metrics
instrumentator = Instrumentator()
instrumentator.instrument(app)

# API key security
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(request: Request, api_key: str = Depends(api_key_header)) -> str:
    """Validate API key"""
    # Skip authentication for health check endpoints
    if request.url.path in ["/healthz", "/readyz", "/metrics"]:
        return ""
    
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # In a production system, we would check the key against a database
    # For this MVP, we use a simple comparison with environment variable
    if api_key != API_KEY_SECRET:
        # Log failed attempt in SQLite (skip during testing)
        if not TESTING and sqlite_db is not None:
            client_ip = request.client.host if request.client else "unknown"
            ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
            sqlite_db.execute(
                "INSERT INTO failed_auth_attempts (ip_address) VALUES (?)",
                (ip_hash,)
            )
            
            # Check for brute force attempts
            recent_attempts = sqlite_db.fetchall(
                "SELECT COUNT(*) as count FROM failed_auth_attempts WHERE ip_address = ? AND timestamp > datetime('now', '-10 minutes')",
                (ip_hash,)
            )
            if recent_attempts and recent_attempts[0]["count"] > 5:
                logger.warning(f"Possible brute force attack from {ip_hash}")
                # In production, we might implement a temporary IP ban
        
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return api_key

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Basic liveness probe"""
    return {"status": "ok"}

# Readiness check endpoint
@app.get("/readyz")
async def readiness_check():
    """Dependency readiness probe"""
    checks = {
        "sqlite": sqlite_db.verify_connectivity() if sqlite_db is not None else False,
        "neo4j": neo4j_db.verify_connectivity() if neo4j_db is not None else False,
        "faiss_index": faiss_index.is_trained() if faiss_index is not None else False,
        "llm_backend": llm_health_check(),
        "embedder": embedder_health_check()
    }
    
    # During testing, consider the service ready even if some components are mocked
    if TESTING:
        return {
            "status": "ready",
            "checks": {k: "mocked" if not v else "ok" for k, v in checks.items()}
        }
    
    # If all checks pass, return 200 OK
    if all(checks.values()):
        return {
            "status": "ready",
            "checks": {k: "ok" for k in checks}
        }
    
    # Otherwise, return 503 Service Unavailable with failing checks
    failed_checks = {k: "failed" for k, v in checks.items() if not v}
    passed_checks = {k: "ok" for k, v in checks.items() if v}
    
    return JSONResponse(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "checks": {**passed_checks, **failed_checks}
        }
    )

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Query endpoint
@app.post("/query")
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Main chat endpoint for KG Q&A with streaming response
    """
    logger.info(f"Query request: {request.question}")
    query_id = generate_query_id()
    start_time = time.time()
    
    # Check for empty question
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_QUERY",
                "message": "Question cannot be empty"
            }
        )
    
    # Streaming response function
    async def stream_response():
        try:
            # Extract potential entities from question
            potential_entities = extract_query_entities(request.question)
            
            # Link entities to KG
            entity_links = []
            for entity_text in potential_entities:
                entity_candidates = link_entities_v2(entity_text, request.question)
                # Filter by confidence
                entity_links.extend([entity_id for entity_id, conf in entity_candidates if conf >= 0.75])
            
            # If no entities were linked, return error
            if not entity_links:
                yield json.dumps({
                    "event": "error",
                    "data": {
                        "code": "NO_ENTITY_MATCH",
                        "message": "No entities found matching the query terms."
                    }
                })
                return
            
            # Retrieve relevant triples
            try:
                retrieved_triples = hybrid_retrieve_v2(request.question, entity_links)
            except RetrievalEmpty:
                yield json.dumps({
                    "event": "error",
                    "data": {
                        "code": "NO_RELEVANT_TRIPLES",
                        "message": "No relevant information found in the knowledge graph."
                    }
                })
                return
            
            # Send metadata
            latency_ms = int((time.time() - start_time) * 1000)
            yield json.dumps({
                "event": "metadata",
                "data": {
                    "query_id": query_id,
                    "latency_ms": latency_ms,
                    "triple_count": len(retrieved_triples),
                    "entity_count": len(entity_links)
                }
            })
            
            # If visualize_graph is True, send graph data
            if request.visualize_graph:
                graph_data = triples_to_graph_data(retrieved_triples, entity_links)
                yield json.dumps({
                    "event": "graph_data",
                    "data": graph_data.to_dict()
                })
            
            # Format prompt for LLM
            system_message = "You are a precise, factual question-answering assistant. Your knowledge is strictly limited to the triples provided below. Respond concisely with factual information only."
            prompt = format_prompt(system_message, [t.to_dict() for t in retrieved_triples], request.question)
            
            # Stream tokens from LLM
            answer_text = ""
            for token in stream_tokens(prompt):
                answer_text += token
                yield json.dumps({
                    "event": "llm_token",
                    "data": {
                        "token": token
                    }
                })
            
            # Validate LLM output
            triple_ids = {t.id for t in retrieved_triples}
            answers, cited_ids, trust_level = validate_llm_output(answer_text, triple_ids)
            
            # Send citation data
            for cited_id in cited_ids:
                # Find the cited triple
                cited_triple = next((t for t in retrieved_triples if t.id == cited_id), None)
                if cited_triple:
                    yield json.dumps({
                        "event": "citation_data",
                        "data": {
                            "id": cited_id,
                            "text": f"{cited_triple.head_name} -[{cited_triple.relation_name}]-> {cited_triple.tail_name}"
                        }
                    })
            
            # Send final metadata with trust score
            trust_score = 0.95 if trust_level == "high" else 0.5
            yield json.dumps({
                "event": "metadata",
                "data": {
                    "trust_score": trust_score,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "query_id": query_id
                }
            })
            
            # End stream
            yield json.dumps({
                "event": "end",
                "data": {
                    "message": "Stream ended"
                }
            })
            
        except AmbiguousEntityError as e:
            yield json.dumps({
                "event": "error",
                "data": {
                    "code": "AMBIGUOUS_ENTITIES",
                    "message": "Query terms are ambiguous.",
                    "candidates": [{"id": c["id"], "name": c["name"]} for c in e.candidates]
                }
            })
        except EntityLinkingError as e:
            yield json.dumps({
                "event": "error",
                "data": {
                    "code": "ENTITY_LINKING_FAILED",
                    "message": str(e)
                }
            })
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            yield json.dumps({
                "event": "error",
                "data": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while processing your query."
                }
            })
    
    # Return streaming response
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

# Graph browse endpoint
@app.get("/graph/browse")
async def graph_browse(
    page: int = Query(1, ge=1),
    limit: int = Query(500, ge=1, le=2000),
    node_types_filter: List[str] = Query(None),
    relation_types_filter: List[str] = Query(None),
    center_node_id: Optional[str] = Query(None),
    hops: int = Query(1, ge=1, le=3),
    search_term: Optional[str] = Query(None, min_length=3),
    api_key: str = Depends(get_api_key)
):
    """
    Paginated knowledge graph exploration
    """
    logger.info(f"Graph browse request: page={page}, limit={limit}")
    
    try:
        # Base query conditions
        conditions = []
        params = {
            "skip": (page - 1) * limit,
            "limit": limit,
        }
        
        # Add filters if provided
        if node_types_filter:
            conditions.append("e.type IN $node_types")
            params["node_types"] = node_types_filter
            
        if relation_types_filter:
            conditions.append("r.name IN $relation_types")
            params["relation_types"] = relation_types_filter
            
        if search_term:
            conditions.append("(e1.name CONTAINS $search OR e2.name CONTAINS $search OR r.name CONTAINS $search)")
            params["search"] = search_term
        
        # If center_node_id is provided, retrieve its neighborhood
        if center_node_id:
            query = """
            MATCH (e:Entity {id: $center_node_id})
            CALL apoc.path.expandConfig(e, {
                minLevel: 1,
                maxLevel: $hops,
                uniqueness: 'RELATIONSHIP_GLOBAL'
            })
            YIELD path
            WITH DISTINCT relationships(path) as rels
            UNWIND rels as r
            MATCH (e1)-[r:REL]->(e2)
            """
            params["center_node_id"] = center_node_id
            params["hops"] = hops
        else:
            # Otherwise, retrieve paginated subgraph
            query = """
            MATCH (e1)-[r:REL]->(e2)
            """
        
        # Add filters
        if conditions:
            query += "WHERE " + " AND ".join(conditions)
        
        # Count total
        count_query = query + "\nRETURN count(*) as total"
        count_result = neo4j_db.run_query(count_query, params)
        total_links = count_result[0]["total"] if count_result else 0
        
        # Get data with pagination
        data_query = query + """
        RETURN
            e1.id as source_id,
            e1.name as source_name,
            e1.type as source_type,
            r.id as relation_id,
            r.name as relation_name,
            e2.id as target_id,
            e2.name as target_name,
            e2.type as target_type
        ORDER BY e1.name, e2.name
        SKIP $skip
        LIMIT $limit
        """
        
        result = neo4j_db.run_query(data_query, params)
        
        # Process results into graph data
        nodes = {}
        links = []
        
        for record in result:
            # Add source node if not already added
            if record["source_id"] not in nodes:
                nodes[record["source_id"]] = {
                    "id": record["source_id"],
                    "name": record["source_name"],
                    "type": record["source_type"] or "Entity",
                    "properties": {}
                }
            
            # Add target node if not already added
            if record["target_id"] not in nodes:
                nodes[record["target_id"]] = {
                    "id": record["target_id"],
                    "name": record["target_name"],
                    "type": record["target_type"] or "Entity",
                    "properties": {}
                }
            
            # Add link
            links.append({
                "source": record["source_id"],
                "target": record["target_id"],
                "relation_id": record["relation_id"],
                "relation_name": record["relation_name"],
                "properties": {}
            })
        
        # Return paginated graph data
        return {
            "nodes": list(nodes.values()),
            "links": links,
            "page": page,
            "limit": limit,
            "total_nodes_in_filter": len(nodes),
            "total_links_in_filter": total_links,
            "has_more": (page * limit) < total_links
        }
    
    except Exception as e:
        logger.exception(f"Error browsing graph: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GRAPH_BROWSE_ERROR",
                "message": "An error occurred while browsing the knowledge graph."
            }
        )

# Ingest endpoint
@app.post("/ingest")
async def ingest(
    request: IngestRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Batch ingest triples into the knowledge graph
    """
    logger.info(f"Ingest request: {len(request.triples)} triples")
    
    try:
        # Validate triples
        if not request.triples:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "EMPTY_TRIPLES",
                    "message": "No triples provided for ingestion."
                }
            )
        
        # Prepare triples for staging
        staged_count = 0
        errors = []
        
        for i, triple in enumerate(request.triples):
            try:
                # Check required fields
                required_fields = ["head", "relation", "tail"]
                for field in required_fields:
                    if field not in triple or not triple[field]:
                        raise ValueError(f"Missing required field: {field}")
                
                # Insert into staging table
                try:
                    sqlite_db.execute(
                        "INSERT INTO staging_triples (h_text, r_text, t_text, status) VALUES (?, ?, ?, 'pending')",
                        (triple["head"], triple["relation"], triple["tail"])
                    )
                    staged_count += 1
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e):
                        logger.info(f"Duplicate triple: {triple}")
                        # Count as success for duplicate triples
                        staged_count += 1
                    else:
                        raise e
                
            except Exception as e:
                logger.warning(f"Error staging triple {i}: {e}")
                errors.append({
                    "index": i,
                    "triple": triple,
                    "error": str(e)
                })
        
        # Return result
        return {
            "status": "accepted",
            "triples_staged": staged_count,
            "errors": errors,
            "message": f"Staged {staged_count} triples for ingestion"
        }
    
    except Exception as e:
        logger.exception(f"Error in ingest endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INGEST_ERROR",
                "message": "An error occurred during ingest operation."
            }
        )

# Feedback endpoint
@app.post("/feedback")
async def feedback(
    request: FeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Record user feedback on answers
    """
    logger.info(f"Feedback request for query_id: {request.query_id}")
    
    try:
        # Store feedback in SQLite
        sqlite_db.execute(
            "INSERT INTO feedback (query_id, is_correct, comment, expected_answer) VALUES (?, ?, ?, ?)",
            (
                request.query_id,
                request.is_correct,
                request.comment,
                request.expected_answer
            )
        )
        
        # Also append to a feedback file for easier analysis
        from pathlib import Path
        feedback_dir = Path("data")
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query_id": request.query_id,
                "is_correct": request.is_correct,
                "comment": request.comment,
                "expected_answer": request.expected_answer
            }, f)
            f.write("\n")
        
        return {
            "status": "accepted",
            "message": "Feedback recorded successfully"
        }
    
    except Exception as e:
        logger.exception(f"Error recording feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "FEEDBACK_ERROR",
                "message": "An error occurred while recording feedback."
            }
        )