# Deployment Guide

This guide covers production deployment of SubgraphRAG+ with best practices for security, scalability, and monitoring.

## Quick Production Setup

### Docker Deployment (Recommended)

```bash
# Clone and setup
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Production configuration
cp .env.example .env.production
# Edit .env.production with production values

# Deploy with Docker
cd deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Production Configuration

### Environment Variables

Create a production `.env` file:

```bash
# Database Configuration
NEO4J_URI=neo4j://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password_here

# Security
API_KEY_SECRET=your_very_secure_random_key_here
ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Model Configuration
MODEL_BACKEND=openai
OPENAI_API_KEY=your_openai_api_key

# Performance
WORKERS=4
MAX_CONNECTIONS=100
CACHE_SIZE=1000

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
SENTRY_DSN=your_sentry_dsn_here
```

### Application Configuration

Update `config/config.json` for production:

```json
{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss_index.bin",
  "TOKEN_BUDGET": 4000,
  "MLP_MODEL_PATH": "models/mlp_pretrained.pt",
  "CACHE_DIR": "cache/",
  "MAX_DDE_HOPS": 2,
  "LOG_LEVEL": "INFO",
  "API_RATE_LIMIT": 100,
  "ENABLE_CORS": true,
  "CORS_ORIGINS": ["https://yourdomain.com"],
  "MAX_QUERY_LENGTH": 1000,
  "MAX_BATCH_SIZE": 1000,
  "EMBEDDING_CACHE_SIZE": 10000,
  "CONNECTION_POOL_SIZE": 20
}
```

## Docker Production Setup

### Production Docker Compose

Create `deployment/docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  subgraphrag:
    environment:
      - WORKERS=4
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  neo4j:
    environment:
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    deploy:
      resources:
        limits:
          memory: 6G
          cpus: '2.0'
        reservations:
          memory: 4G
          cpus: '1.0'
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - subgraphrag
    restart: always
```

### Nginx Configuration

Create `deployment/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream subgraphrag {
        server subgraphrag:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://subgraphrag;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # SSE support
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding off;
        }

        location /healthz {
            proxy_pass http://subgraphrag/healthz;
            access_log off;
        }
    }
}
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: subgraphrag
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: subgraphrag-config
  namespace: subgraphrag
data:
  config.json: |
    {
      "MODEL_BACKEND": "openai",
      "FAISS_INDEX_PATH": "data/faiss_index.bin",
      "TOKEN_BUDGET": 4000,
      "API_RATE_LIMIT": 100
    }
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: subgraphrag-api
  namespace: subgraphrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: subgraphrag-api
  template:
    metadata:
      labels:
        app: subgraphrag-api
    spec:
      containers:
      - name: subgraphrag
        image: subgraphrag:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          value: "neo4j://neo4j-service:7687"
        - name: API_KEY_SECRET
          valueFrom:
            secretKeyRef:
              name: subgraphrag-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: subgraphrag-config
      - name: data
        persistentVolumeClaim:
          claimName: subgraphrag-data
```

### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: subgraphrag-service
  namespace: subgraphrag
spec:
  selector:
    app: subgraphrag-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: subgraphrag-ingress
  namespace: subgraphrag
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: subgraphrag-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: subgraphrag-service
            port:
              number: 8000
```

## Security Best Practices

### 1. API Key Management

```bash
# Generate secure API keys
openssl rand -hex 32

# Use environment-specific keys
# Development: simple keys for testing
# Production: cryptographically secure keys
```

### 2. Network Security

- Use HTTPS/TLS for all communications
- Implement rate limiting at multiple layers
- Use Web Application Firewall (WAF)
- Restrict database access to application only

### 3. Container Security

```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Scan for vulnerabilities
RUN apt-get update && apt-get upgrade -y
```

### 4. Secrets Management

```yaml
# Kubernetes secrets
apiVersion: v1
kind: Secret
metadata:
  name: subgraphrag-secrets
type: Opaque
data:
  api-key: <base64-encoded-key>
  neo4j-password: <base64-encoded-password>
  openai-api-key: <base64-encoded-key>
```

## Monitoring and Observability

### 1. Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'subgraphrag'
    static_configs:
      - targets: ['subgraphrag:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### 2. Grafana Dashboard

Key metrics to monitor:
- Request rate and latency
- Error rates by endpoint
- Database connection health
- Memory and CPU usage
- FAISS index size and query performance

### 3. Logging

```python
# Structured logging configuration
LOGGING = {
    'version': 1,
    'formatters': {
        'json': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['stdout']
    }
}
```

### 4. Health Checks

```bash
# Application health
curl -f http://localhost:8000/healthz

# Dependency readiness
curl -f http://localhost:8000/readyz

# Detailed status
curl -f http://localhost:8000/status
```

## Backup and Recovery

### 1. Database Backup

```bash
# Neo4j backup
docker exec neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j-$(date +%Y%m%d).dump

# SQLite backup
sqlite3 data/staging.db ".backup data/staging-$(date +%Y%m%d).db"
```

### 2. FAISS Index Backup

```bash
# Backup FAISS index
tar -czf faiss-backup-$(date +%Y%m%d).tar.gz data/faiss/
```

### 3. Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Neo4j backup
docker exec neo4j neo4j-admin dump --database=neo4j --to=/data/neo4j-backup.dump
docker cp neo4j:/data/neo4j-backup.dump $BACKUP_DIR/

# Application data backup
tar -czf $BACKUP_DIR/app-data.tar.gz data/ config/

# Upload to cloud storage
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/subgraphrag/ --recursive
```

## Performance Optimization

### 1. Database Tuning

```bash
# Neo4j memory configuration
NEO4J_dbms_memory_heap_initial__size=4G
NEO4J_dbms_memory_heap_max__size=8G
NEO4J_dbms_memory_pagecache_size=4G
```

### 2. Application Tuning

```python
# Connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30

# FAISS optimization
FAISS_NPROBE=32
FAISS_USE_GPU=true
```

### 3. Caching Strategy

```python
# Redis for distributed caching
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600
ENABLE_QUERY_CACHE=true
```

## Scaling Strategies

### 1. Horizontal Scaling

- Multiple API instances behind load balancer
- Neo4j read replicas for query distribution
- Distributed FAISS indices

### 2. Vertical Scaling

- Increase memory for larger FAISS indices
- More CPU cores for parallel processing
- SSD storage for faster I/O

### 3. Auto-scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: subgraphrag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: subgraphrag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce FAISS index size or enable quantization
2. **Slow Queries**: Check Neo4j query plans and add indexes
3. **Connection Timeouts**: Increase connection pool sizes
4. **Rate Limiting**: Adjust rate limits based on usage patterns

### Debug Commands

```bash
# Check container logs
docker logs subgraphrag_api

# Monitor resource usage
docker stats

# Database connectivity
docker exec subgraphrag_api python -c "from src.app.database import test_connection; test_connection()"
```

This deployment guide provides a comprehensive foundation for running SubgraphRAG+ in production environments with proper security, monitoring, and scalability considerations.
