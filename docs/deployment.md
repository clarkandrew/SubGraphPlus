# 🚀 Deployment Guide

This guide covers deploying SubgraphRAG+ in various environments, from local Docker setups to cloud platforms.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free space
- Network: Stable internet connection

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: High-bandwidth connection

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Git
- (Optional) Kubernetes 1.20+

## 🐳 Docker Deployment

### Quick Start with Docker Compose

1. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/SubgraphRAG-Plus.git
   cd SubgraphRAG-Plus
   cp .env.example .env
   ```

2. **Configure Environment**
   ```bash
   # Edit .env file with your settings
   nano .env
   ```

3. **Deploy**
   ```bash
   # Start all services
   docker-compose up -d
   
   # Check status
   docker-compose ps
   
   # View logs
   docker-compose logs -f
   ```

4. **Initialize Data**
   ```bash
   # Load sample data
   docker-compose exec api python scripts/stage_ingest.py --sample
   docker-compose exec api python scripts/ingest_worker.py --process-all
   ```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - WORKERS=4
      - NEO4J_URI=neo4j://neo4j:7687
    depends_on:
      - neo4j
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  redis_data:
```

### Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p data logs cache models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

## ☁️ Cloud Platform Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name subgraphrag-plus
   ```

2. **Build and Push Image**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
   
   # Build image
   docker build -f Dockerfile.prod -t subgraphrag-plus .
   
   # Tag image
   docker tag subgraphrag-plus:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/subgraphrag-plus:latest
   
   # Push image
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/subgraphrag-plus:latest
   ```

3. **ECS Task Definition**
   ```json
   {
     "family": "subgraphrag-plus",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "api",
         "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/subgraphrag-plus:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "NEO4J_URI",
             "value": "neo4j://your-neo4j-endpoint:7687"
           }
         ],
         "secrets": [
           {
             "name": "NEO4J_PASSWORD",
             "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:neo4j-password"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/subgraphrag-plus",
             "awslogs-region": "us-west-2",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Using AWS Lambda (Serverless)

```python
# lambda_handler.py
import json
from mangum import Mangum
from app.main import app

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and Deploy**
   ```bash
   # Build image
   gcloud builds submit --tag gcr.io/PROJECT_ID/subgraphrag-plus
   
   # Deploy to Cloud Run
   gcloud run deploy subgraphrag-plus \
     --image gcr.io/PROJECT_ID/subgraphrag-plus \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --set-env-vars NEO4J_URI=neo4j://your-endpoint:7687
   ```

2. **Cloud Run YAML**
   ```yaml
   apiVersion: serving.knative.dev/v1
   kind: Service
   metadata:
     name: subgraphrag-plus
     annotations:
       run.googleapis.com/ingress: all
   spec:
     template:
       metadata:
         annotations:
           run.googleapis.com/cpu-throttling: "false"
       spec:
         containerConcurrency: 100
         containers:
         - image: gcr.io/PROJECT_ID/subgraphrag-plus
           ports:
           - containerPort: 8000
           resources:
             limits:
               cpu: "2"
               memory: "2Gi"
           env:
           - name: NEO4J_URI
             value: "neo4j://your-endpoint:7687"
           - name: NEO4J_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: neo4j-secret
                 key: password
   ```

### Microsoft Azure

#### Using Container Instances

```bash
# Create resource group
az group create --name subgraphrag-rg --location eastus

# Create container instance
az container create \
  --resource-group subgraphrag-rg \
  --name subgraphrag-plus \
  --image your-registry/subgraphrag-plus:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    NEO4J_URI=neo4j://your-endpoint:7687 \
  --secure-environment-variables \
    NEO4J_PASSWORD=your-password
```

## ⚙️ Kubernetes Deployment

### Kubernetes Manifests

#### Namespace
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: subgraphrag
```

#### ConfigMap
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: subgraphrag-config
  namespace: subgraphrag
data:
  NEO4J_URI: "neo4j://neo4j-service:7687"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
```

#### Secret
```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: subgraphrag-secrets
  namespace: subgraphrag
type: Opaque
data:
  NEO4J_PASSWORD: <base64-encoded-password>
  API_KEY_SECRET: <base64-encoded-secret>
  OPENAI_API_KEY: <base64-encoded-key>
```

#### Deployment
```yaml
# deployment.yaml
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
      - name: api
        image: your-registry/subgraphrag-plus:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: subgraphrag-config
        - secretRef:
            name: subgraphrag-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: subgraphrag-service
  namespace: subgraphrag
spec:
  selector:
    app: subgraphrag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: subgraphrag-ingress
  namespace: subgraphrag
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
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
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n subgraphrag

# View logs
kubectl logs -f deployment/subgraphrag-api -n subgraphrag

# Scale deployment
kubectl scale deployment subgraphrag-api --replicas=5 -n subgraphrag
```

## 🔒 Production Security

### SSL/TLS Configuration

#### Nginx Configuration
```nginx
# nginx.conf
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Security

```bash
# Production environment variables
export DEBUG=false
export API_KEY_SECRET=$(openssl rand -hex 32)
export NEO4J_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET_KEY=$(openssl rand -hex 32)

# Restrict file permissions
chmod 600 .env
chmod 600 config/*.json
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: subgraphrag-network-policy
  namespace: subgraphrag
spec:
  podSelector:
    matchLabels:
      app: subgraphrag-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
```

## 📊 Monitoring and Logging

### Prometheus Monitoring

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'subgraphrag'
      static_configs:
      - targets: ['subgraphrag-service:80']
      metrics_path: /metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SubgraphRAG+ Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Centralized Logging

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*subgraphrag*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
    </source>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch-service
      port 9200
      index_name subgraphrag
    </match>
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: subgraphrag-plus
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f Dockerfile.prod -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster production --service subgraphrag-plus --force-new-deployment
```

## 🚨 Backup and Recovery

### Database Backup

```bash
# Neo4j backup script
#!/bin/bash
BACKUP_DIR="/backups/neo4j/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Create backup
docker exec neo4j neo4j-admin database dump --to-path=/backups neo4j

# Compress backup
tar -czf $BACKUP_DIR/neo4j_backup.tar.gz -C /var/lib/neo4j/backups .

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/neo4j_backup.tar.gz s3://your-backup-bucket/
```

### Application Data Backup

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"

mkdir -p $BACKUP_DIR

# Backup configuration
cp -r config/ $BACKUP_DIR/
cp .env $BACKUP_DIR/

# Backup data directory
tar -czf $BACKUP_DIR/data_backup.tar.gz data/

# Backup models
tar -czf $BACKUP_DIR/models_backup.tar.gz models/

echo "Backup completed: $BACKUP_DIR"
```

### Recovery Procedures

```bash
# Restore from backup
#!/bin/bash
BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

BACKUP_DIR="/backups/$BACKUP_DATE"

# Stop services
docker-compose down

# Restore data
tar -xzf $BACKUP_DIR/data_backup.tar.gz
tar -xzf $BACKUP_DIR/models_backup.tar.gz

# Restore configuration
cp -r $BACKUP_DIR/config/ .
cp $BACKUP_DIR/.env .

# Restart services
docker-compose up -d

echo "Recovery completed from backup: $BACKUP_DATE"
```

## 🔍 Health Checks and Monitoring

### Application Health Checks

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    # Check database connectivity
    try:
        await check_neo4j_connection()
        await check_redis_connection()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service not ready")
```

### Monitoring Alerts

```yaml
# alertmanager.yml
groups:
- name: subgraphrag
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    annotations:
      summary: High response time detected
```

---

**🎯 Next Steps**: After deployment, monitor your application using the health checks and metrics endpoints. Set up alerts for critical issues and establish a regular backup schedule.**
