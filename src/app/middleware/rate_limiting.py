"""
Rate limiting middleware for SubgraphRAG+ API
"""

import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import config

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        now = time.time()
        
        # Refill tokens based on time elapsed
        time_elapsed = now - self.last_refill
        tokens_to_add = time_elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until enough tokens are available
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available
        """
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm"""
    
    def __init__(
        self, 
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        cleanup_interval: int = 300  # 5 minutes
    ):
        """
        Initialize rate limiting middleware
        
        Args:
            app: ASGI application
            requests_per_minute: Base rate limit per minute
            burst_size: Maximum burst size (defaults to 2x rate limit)
            cleanup_interval: Interval to clean up old buckets (seconds)
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.burst_size = burst_size or (requests_per_minute * 2)
        self.cleanup_interval = cleanup_interval
        
        # Storage for token buckets per client
        self.buckets: Dict[str, TokenBucket] = {}
        self.last_cleanup = time.time()
        
        logger.info(f"Rate limiting enabled: {requests_per_minute} req/min, burst: {self.burst_size}")
    
    def get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get API key first
        api_key = request.headers.get("X-API-KEY")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def get_bucket(self, client_id: str) -> TokenBucket:
        """
        Get or create token bucket for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            TokenBucket instance
        """
        if client_id not in self.buckets:
            self.buckets[client_id] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.requests_per_second
            )
        return self.buckets[client_id]
    
    def cleanup_old_buckets(self):
        """Remove inactive token buckets to prevent memory leaks"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets that haven't been used recently
        cutoff_time = now - self.cleanup_interval
        to_remove = []
        
        for client_id, bucket in self.buckets.items():
            if bucket.last_refill < cutoff_time:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self.buckets[client_id]
        
        self.last_cleanup = now
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit buckets")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/healthz", "/readyz", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self.get_client_id(request)
        
        # Get token bucket for client
        bucket = self.get_bucket(client_id)
        
        # Try to consume a token
        if not bucket.consume():
            # Rate limit exceeded
            retry_after = bucket.time_until_available()
            
            logger.warning(f"Rate limit exceeded for client {client_id}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Rate limit exceeded. Please try again later.",
                        "retry_after": retry_after
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(int(bucket.tokens)),
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                    "Retry-After": str(int(retry_after))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        # Cleanup old buckets periodically
        self.cleanup_old_buckets()
        
        return response


class AdaptiveRateLimitMiddleware(RateLimitMiddleware):
    """Rate limiting middleware with adaptive limits based on endpoint"""
    
    def __init__(
        self, 
        app: ASGIApp,
        default_requests_per_minute: int = 60,
        endpoint_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize adaptive rate limiting middleware
        
        Args:
            app: ASGI application
            default_requests_per_minute: Default rate limit
            endpoint_limits: Per-endpoint rate limits
        """
        super().__init__(app, default_requests_per_minute)
        
        self.endpoint_limits = endpoint_limits or {
            "/query": 30,  # More expensive endpoint
            "/ingest": 10,  # Heavy write operations
            "/graph/browse": 120,  # Lighter read operations
            "/feedback": 60  # Standard rate
        }
        
        logger.info(f"Adaptive rate limiting enabled with endpoint-specific limits")
    
    def get_rate_limit_for_endpoint(self, path: str) -> Tuple[int, int]:
        """
        Get rate limit for specific endpoint
        
        Args:
            path: Request path
            
        Returns:
            Tuple of (requests_per_minute, burst_size)
        """
        requests_per_minute = self.endpoint_limits.get(path, self.requests_per_minute)
        burst_size = requests_per_minute * 2
        return requests_per_minute, burst_size
    
    def get_bucket(self, client_id: str, endpoint: str = None) -> TokenBucket:
        """
        Get or create token bucket for client and endpoint
        
        Args:
            client_id: Client identifier
            endpoint: Endpoint path
            
        Returns:
            TokenBucket instance
        """
        bucket_key = f"{client_id}:{endpoint}" if endpoint else client_id
        
        if bucket_key not in self.buckets:
            requests_per_minute, burst_size = self.get_rate_limit_for_endpoint(endpoint)
            requests_per_second = requests_per_minute / 60.0
            
            self.buckets[bucket_key] = TokenBucket(
                capacity=burst_size,
                refill_rate=requests_per_second
            )
        
        return self.buckets[bucket_key]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with adaptive rate limiting
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/healthz", "/readyz", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier and endpoint
        client_id = self.get_client_id(request)
        endpoint = request.url.path
        
        # Get token bucket for client and endpoint
        bucket = self.get_bucket(client_id, endpoint)
        
        # Get rate limit for this endpoint
        requests_per_minute, _ = self.get_rate_limit_for_endpoint(endpoint)
        
        # Try to consume a token
        if not bucket.consume():
            # Rate limit exceeded
            retry_after = bucket.time_until_available()
            
            logger.warning(f"Rate limit exceeded for client {client_id} on endpoint {endpoint}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit exceeded for {endpoint}. Please try again later.",
                        "retry_after": retry_after,
                        "endpoint_limit": requests_per_minute
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": str(int(bucket.tokens)),
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                    "Retry-After": str(int(retry_after))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        # Cleanup old buckets periodically
        self.cleanup_old_buckets()
        
        return response 