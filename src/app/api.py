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
from app.retriever import hybrid_retrieve_v2, entity_search, get_faiss_index
from app.verify import validate_llm_output, format_prompt
from app.ml.embedder import health_check as embedder_health_check
from app.ml.llm import generate_answer, stream_tokens, health_check as llm_health_check

# Import services
from app.services.information_extraction import get_information_extraction_service
from app.services.ingestion import get_ingestion_service

# RULE:import-rich-logger-correctly - Use centralized rich logger
from .log import logger, log_and_print, CONSOLEx
from rich.console import Console

# Initialize rich console for pretty CLI output


# API Models for Information Extraction
class ExtractRequest(BaseModel):
    text: str
    max_length: int = 256
    num_beams: int = 3

class IETriple(BaseModel):
    head: str
    relation: str
    tail: str
    head_type: Optional[str] = "ENTITY"
    tail_type: Optional[str] = "ENTITY"
    confidence: float = 1.0

class ExtractResponse(BaseModel):
    triples: List[IETriple]
    raw_output: str
    processing_time: float

# Text ingestion models
class TextIngestRequest(BaseModel):
    text: str
    source: str = "api_input"
    chunk_size: int = 1000

class TextIngestResponse(BaseModel):
    total_triples: int
    successful_triples: int
    failed_triples: int
    processing_time: float
    errors: List[str]
    warnings: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting SubgraphRAG+ Unified API server")
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Initialize metrics
    instrumentator.expose(app)
    
    # Initialize services (but don't preload models)
    logger.info("Initializing services...")
    ie_service = get_information_extraction_service()
    logger.info("IE service initialized")
    
    # Load models after FastAPI fork to avoid Apple Silicon hanging issues
    logger.info("🚀 Loading models in FastAPI lifespan to avoid forking issues...")
    from app.services.information_extraction import init_models_for_fastapi
    models_loaded = init_models_for_fastapi()
    if models_loaded:
        logger.info("✅ Models loaded successfully in FastAPI lifespan")
    else:
        logger.warning("⚠️ Models failed to load in FastAPI lifespan - will use mock responses")
    
    # Temporarily comment out ingestion service to debug hang
    # ingestion_service = get_ingestion_service()
    logger.info("Services initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SubgraphRAG+ Unified API server")
    
    # Close database connections
    from app.database import close_connections
    close_connections()

# Create FastAPI app with lifespan
app = FastAPI(
    title="SubgraphRAG+ Unified API",
    description="Advanced knowledge graph question answering with hybrid retrieval and information extraction",
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
    if request.url.path in ["/healthz", "/readyz", "/metrics", "/ie/health", "/ie/info"]:
        logger.info(f"🔐 AUTH SKIP - Health check endpoint: {request.url.path}")
        return ""
    
    logger.info(f"🔐 AUTH CHECK - Validating API key...")
    
    if not api_key:
        logger.warning(f"🔐 AUTH FAIL - Missing API key")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # In a production system, we would check the key against a database
    # For this MVP, we use a simple comparison with environment variable
    if api_key != API_KEY_SECRET:
        logger.warning(f"🔐 AUTH FAIL - Invalid API key provided")
        
        # Log failed attempt in SQLite (skip during testing)
        if not TESTING and sqlite_db is not None:
            # ASYNC SAFETY: Skip SQLite operations in async context to prevent blocking
            # This prevents the event loop from freezing on database calls
            logger.warning(f"🔐 AUTH DB - Skipping SQLite operations to prevent async blocking")
            
            # TODO: Implement proper async database operations using asyncio
            # For now, we'll just log the failed attempt without storing it
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(f"🔐 AUTH FAIL - Failed auth attempt from IP: {client_ip} (not stored)")
        else:
            logger.info(f"🔐 AUTH DB - SQLite operations skipped (TESTING={TESTING}, sqlite_db={sqlite_db is not None})")
        
        logger.warning(f"🔐 AUTH FAIL - Raising HTTPException for invalid key")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    logger.info(f"🔐 AUTH SUCCESS - Valid API key provided")
    return api_key

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Basic liveness probe"""
    return {"status": "ok"}

# Readiness check endpoint
@app.get("/readyz")
async def readiness_check(skip_model_loading: bool = Query(True, description="Skip expensive model loading checks")):
    """Dependency readiness probe"""
    ie_service = get_information_extraction_service()
    
    # Basic checks (fast)
    checks = {
        "sqlite": sqlite_db.verify_connectivity() if sqlite_db is not None else False,
        "neo4j": neo4j_db.verify_connectivity() if neo4j_db is not None else False,
        "faiss_index": get_faiss_index() is not None and get_faiss_index().is_trained(),
        "llm_backend": llm_health_check(),  # Now lightweight
        "embedder": embedder_health_check(),  # Now lightweight
        "rebel_service": ie_service.is_service_available(),  # Check service, not model loading
        "ie_models_available": ie_service.is_service_available()  # IE models available
    }
    
    # Expensive checks (only if requested)
    if not skip_model_loading:
        from app.ml.llm import model_readiness_check as llm_readiness
        from app.ml.embedder import model_readiness_check as embedder_readiness
        
        checks.update({
            "llm_model_loaded": llm_readiness(),
            "embedder_model_loaded": embedder_readiness(),
            "rebel_model_loaded": ie_service.is_rebel_loaded(),
            "ner_model_loaded": ie_service.is_ner_loaded()
        })
    
    # During testing, consider the service ready even if some components are mocked
    if TESTING:
        return {
            "status": "ready",
            "checks": {k: "mocked" if not v else "ok" for k, v in checks.items()},
            "note": "Testing mode - some checks mocked"
        }
    
    # Calculate status
    basic_checks = ["sqlite", "neo4j", "faiss_index", "llm_backend", "embedder", "rebel_service"]
    basic_ready = all(checks.get(k, False) for k in basic_checks)
    
    if basic_ready:
        status_code = 200
        status = "ready"
    else:
        status_code = HTTP_503_SERVICE_UNAVAILABLE
        status = "not_ready"
    
    # Format response
    response_data = {
        "status": status,
        "checks": {k: ("ok" if v else "failed") for k, v in checks.items()},
        "model_loading_skipped": skip_model_loading
    }
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Information Extraction Endpoints
@app.post("/ie/extract", response_model=ExtractResponse)
async def extract_triples(
    request: ExtractRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Extract triples from input text using REBEL
    
    Args:
        request: ExtractRequest with text and optional parameters
        
    Returns:
        ExtractResponse with extracted triples and metadata
    """
    logger.info(f"🎯 Starting IE extraction for text: '{request.text[:50]}...' (length: {len(request.text)})")
    CONSOLEx.print(f"[bold green]🎯 IE Extraction Request[/bold green] - Text length: {len(request.text)} chars")
    
    try:
        # Delegate to information extraction service
        ie_service = get_information_extraction_service()
        
        start_service_call = time.time()
        result = await ie_service.extract_triples(
            text=request.text,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        service_call_time = time.time() - start_service_call
        
        logger.info(f"✅ IE extraction completed: {len(result.triples) if result.success else 0} triples in {service_call_time:.2f}s")
        CONSOLEx.print(f"[bold green]✅ Extraction completed[/bold green] - Found {len(result.triples) if result.success else 0} triples")
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {result.error_message}"
            )
        
        # Convert to API response format
        triples = [
            IETriple(
                head=t['head'],
                relation=t['relation'],
                tail=t['tail'],
                head_type=t.get('head_type', 'ENTITY'),
                tail_type=t.get('tail_type', 'ENTITY'),
                confidence=t['confidence']
            )
            for t in result.triples
        ]
        
        response = ExtractResponse(
            triples=triples,
            raw_output=result.raw_output,
            processing_time=result.processing_time
        )
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Error in extract triples API: {e}")
        CONSOLEx.print_exception()
        raise HTTPException(500, f"Extraction failed: {e}")

@app.get("/ie/health")
async def ie_health_check():
    """Health check endpoint for IE functionality"""
    try:
        ie_service = get_information_extraction_service()
        model_info = ie_service.get_model_info()
        
        # Check individual model status
        rebel_status = "loaded" if any(m["loaded"] and m["purpose"] == "relation_extraction" for m in model_info["models"]) else "not_loaded"
        ner_status = "loaded" if any(m["loaded"] and m["purpose"] == "entity_typing" for m in model_info["models"]) else "not_loaded"
        
        overall_status = "healthy" if model_info["overall_status"] == "ready" else "partial" if model_info["overall_status"] == "partial" else "unhealthy"
        
        return {
            "status": overall_status,
            "models": {
                "rebel": {
                    "name": "Babelscape/rebel-large",
                    "purpose": "relation_extraction",
                    "status": rebel_status
                },
                "ner": {
                    "name": "tner/roberta-large-ontonotes5",
                    "purpose": "entity_typing",
                    "status": ner_status
                }
            },
            "overall_status": model_info["overall_status"]
        }
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"IE health check error: {e}")
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/ie/info")
async def ie_model_info():
    """Get Information Extraction model information"""
    try:
        ie_service = get_information_extraction_service()
        model_info = ie_service.get_model_info()
        
        return {
            **model_info,
            "service": "SubgraphRAG+ Unified API - IE Module"
        }
    except Exception as e:
        logger.exception("Failed to get IE model info")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/load-test")
async def debug_load_test():
    """Debug endpoint to test model loading in isolation"""
    try:
        import time
        from transformers import AutoTokenizer
        
        start_time = time.time()
        logger.info("🔍 Debug: Starting tokenizer load test...")
        
        # Try to load just the tokenizer first
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        tokenizer_time = time.time() - start_time
        
        logger.info(f"🔍 Debug: Tokenizer loaded in {tokenizer_time:.2f}s")
        
        return {
            "status": "success",
            "tokenizer_load_time": tokenizer_time,
            "tokenizer_vocab_size": tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "unknown"
        }
    except Exception as e:
        logger.exception("Debug load test failed")
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }
        # RULE:rich-error-handling-required
        logger.error(f"Error getting IE model info: {e}")
        raise HTTPException(500, f"Failed to get model info: {e}")

# Text Ingestion Endpoint
@app.post("/ingest/text", response_model=TextIngestResponse)
async def ingest_text(
    request: TextIngestRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Ingest raw text content through the full IE and staging pipeline
    
    Args:
        request: TextIngestRequest with text content and parameters
        
    Returns:
        TextIngestResponse with processing statistics
    """
    logger.info(f"Text ingestion request: {len(request.text)} characters")
    
    try:
        # Delegate to ingestion service
        ingestion_service = get_ingestion_service()
        result = await ingestion_service.process_text_content(
            text=request.text,
            source=request.source
        )
        
        return TextIngestResponse(
            total_triples=result.total_triples,
            successful_triples=result.successful_triples,
            failed_triples=result.failed_triples,
            processing_time=result.processing_time,
            errors=result.errors,
            warnings=result.warnings
        )
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Error in text ingestion API: {e}")
        CONSOLEx.print_exception()
        raise HTTPException(500, f"Text ingestion failed: {e}")

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

# Triple batch ingest endpoint
@app.post("/ingest")
async def ingest(
    request: IngestRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Batch ingest pre-extracted triples into the knowledge graph
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
        
        # Delegate to ingestion service
        ingestion_service = get_ingestion_service()
        result = ingestion_service.stage_triples_batch(
            triples=request.triples,
            source="api_batch"
        )
        
        # Return result
        return {
            "status": "accepted",
            "triples_staged": result.successful_triples,
            "errors": [{"error": err} for err in result.errors],
            "message": f"Staged {result.successful_triples} triples for ingestion"
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