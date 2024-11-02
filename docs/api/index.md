# API Reference

## 🌐 Overview

The NepaliGov-RAG-Bench API provides comprehensive endpoints for document processing, semantic search, and bilingual translation. The API follows RESTful principles and returns JSON responses.

## 🔗 Base URL

### Local Development
```
http://localhost:8093/api/v1
```

### Docker Deployment
```bash
# If running with Docker Compose
docker compose up -d

# Access via:
http://localhost:8093/api/v1

# Health check endpoint
http://localhost:8093/health
```

## 🔐 Authentication

Currently, the API does not require authentication for basic operations. Future versions will include JWT-based authentication.

```http
Authorization: Bearer <jwt_token>
```

## 📊 Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

## 🏥 Health & Monitoring Endpoints

### GET /health

Check the health status of the application (Docker health check endpoint).

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-19T16:53:31.173813",
  "service": "nepali-gov-rag-bench",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

**Use Cases:**
- Docker health checks
- Load balancer health monitoring
- System monitoring and alerting

## 🔍 Search Endpoints

### POST /search

Perform semantic search on government documents.

**Request:**
```json
{
  "query": "What are the requirements for starting a business?",
  "language": "en",
  "max_results": 10,
  "include_sources": true,
  "filters": {
    "document_type": "policy",
    "authority": "Ministry of Industry"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "To start a business in Nepal, you need to register with the Department of Industry...",
        "source": {
          "doc_id": "doc_123",
          "title": "Business Registration Guide",
          "authority": "Ministry of Industry",
          "page": 15,
          "similarity_score": 0.95
        },
        "translated_text": "नेपालमा व्यापार सुरु गर्नका लागि तपाईंले उद्योग विभागमा दर्ता गर्नुपर्छ...",
        "confidence": 0.95
      }
    ],
    "total_results": 5,
    "query_processing_time": 1.2,
    "translated_query": "व्यापार सुरु गर्नका लागि के के आवश्यकता छ?"
  }
}
```

### POST /similar

Find documents similar to a given document.

**Request:**
```json
{
  "doc_id": "doc_123",
  "max_results": 5,
  "similarity_threshold": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "similar_documents": [
      {
        "doc_id": "doc_124",
        "title": "Business License Requirements",
        "similarity_score": 0.92,
        "authority": "Ministry of Industry"
      }
    ]
  }
}
```

### GET /suggestions

Get query suggestions based on popular queries and document content.

**Request:**
```http
GET /suggestions?query=business&limit=5
```

**Response:**
```json
{
  "success": true,
  "data": {
    "suggestions": [
      "business registration requirements",
      "business license application",
      "business tax obligations",
      "business permit process",
      "business compliance checklist"
    ]
  }
}
```

## 📄 Document Endpoints

### POST /documents

Upload a new document for processing.

**Request:**
```http
POST /documents
Content-Type: multipart/form-data

file: <pdf_file>
title: "Business Registration Guide"
authority: "Ministry of Industry"
document_type: "policy"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "doc_id": "doc_456",
    "status": "processing",
    "estimated_completion": "2024-01-01T00:05:00Z"
  }
}
```

### GET /documents

List all documents with optional filtering.

**Request:**
```http
GET /documents?limit=10&offset=0&type=policy&authority=Ministry%20of%20Industry
```

**Response:**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "doc_id": "doc_123",
        "title": "Business Registration Guide",
        "authority": "Ministry of Industry",
        "document_type": "policy",
        "upload_date": "2024-01-01T00:00:00Z",
        "status": "processed",
        "page_count": 25,
        "language": "en"
      }
    ],
    "total_count": 150,
    "has_more": true
  }
}
```

### GET /documents/{doc_id}

Get detailed information about a specific document.

**Request:**
```http
GET /documents/doc_123
```

**Response:**
```json
{
  "success": true,
  "data": {
    "doc_id": "doc_123",
    "title": "Business Registration Guide",
    "authority": "Ministry of Industry",
    "document_type": "policy",
    "upload_date": "2024-01-01T00:00:00Z",
    "processed_date": "2024-01-01T00:02:00Z",
    "status": "processed",
    "page_count": 25,
    "language": "en",
    "file_size": 2048576,
    "checksum": "sha256:abc123...",
    "ocr_quality": 0.95,
    "embedding_status": "completed"
  }
}
```

### DELETE /documents/{doc_id}

Delete a document and all associated data.

**Request:**
```http
DELETE /documents/doc_123
```

**Response:**
```json
{
  "success": true,
  "data": {
    "doc_id": "doc_123",
    "status": "deleted"
  }
}
```

## 🌐 Translation Endpoints

### POST /translate

Translate text between English and Nepali.

**Request:**
```json
{
  "text": "What are the business registration requirements?",
  "source_language": "en",
  "target_language": "ne"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "original_text": "What are the business registration requirements?",
    "translated_text": "व्यापार दर्ताका आवश्यकताहरू के के हुन्?",
    "source_language": "en",
    "target_language": "ne",
    "confidence": 0.95,
    "translation_service": "marianmt"
  }
}
```

### GET /languages

Get supported languages and their capabilities.

**Request:**
```http
GET /languages
```

**Response:**
```json
{
  "success": true,
  "data": {
    "supported_languages": [
      {
        "code": "en",
        "name": "English",
        "native_name": "English",
        "supports_translation": true,
        "supports_ocr": true
      },
      {
        "code": "ne",
        "name": "Nepali",
        "native_name": "नेपाली",
        "supports_translation": true,
        "supports_ocr": true
      }
    ]
  }
}
```

## 📊 Analytics Endpoints

### GET /metrics

Get system performance metrics.

**Request:**
```http
GET /metrics?period=24h
```

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "24h",
    "queries_processed": 1250,
    "average_response_time": 1.2,
    "documents_processed": 25,
    "ocr_success_rate": 0.95,
    "translation_accuracy": 0.92,
    "system_uptime": 0.99,
    "error_rate": 0.01
  }
}
```

### GET /health

Check system health status.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-01-01T00:00:00Z",
    "components": {
      "database": "healthy",
      "vector_index": "healthy",
      "ocr_engine": "healthy",
      "translation_service": "healthy"
    },
    "performance": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.1
    }
  }
}
```

## 🔧 Configuration Endpoints

### GET /config

Get current system configuration.

**Request:**
```http
GET /config
```

**Response:**
```json
{
  "success": true,
  "data": {
    "ocr": {
      "engine": "paddle",
      "confidence_threshold": 0.8,
      "fallback_threshold": 0.6
    },
    "search": {
      "max_results": 10,
      "similarity_threshold": 0.7
    },
    "translation": {
      "service": "marianmt",
      "quality_threshold": 0.85
    }
  }
}
```

### PUT /config

Update system configuration (admin only).

**Request:**
```json
{
  "search": {
    "max_results": 15,
    "similarity_threshold": 0.75
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Configuration updated successfully",
    "updated_fields": ["search.max_results", "search.similarity_threshold"]
  }
}
```

## 🚨 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `PROCESSING_ERROR` | 422 | Document processing failed |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## 📝 Request Examples

### cURL Examples

**Search Query:**
```bash
curl -X POST http://localhost:8093/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are business requirements?",
    "language": "en",
    "max_results": 5
  }'
```

**Upload Document:**
```bash
curl -X POST http://localhost:8093/api/v1/documents \
  -F "file=@business_guide.pdf" \
  -F "title=Business Registration Guide" \
  -F "authority=Ministry of Industry"
```

**Translate Text:**
```bash
curl -X POST http://localhost:8093/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are business requirements?",
    "source_language": "en",
    "target_language": "ne"
  }'
```

### Python Examples

**Search Query:**
```python
import requests

response = requests.post(
    'http://localhost:8093/api/v1/search',
    json={
        'query': 'What are business requirements?',
        'language': 'en',
        'max_results': 5
    }
)

results = response.json()['data']['results']
for result in results:
    print(f"Text: {result['text']}")
    print(f"Source: {result['source']['title']}")
    print(f"Score: {result['source']['similarity_score']}")
```

**Upload Document:**
```python
import requests

with open('business_guide.pdf', 'rb') as f:
    files = {'file': f}
    data = {
        'title': 'Business Registration Guide',
        'authority': 'Ministry of Industry'
    }
    
    response = requests.post(
        'http://localhost:8093/api/v1/documents',
        files=files,
        data=data
    )

doc_id = response.json()['data']['doc_id']
print(f"Document ID: {doc_id}")
```

**Translate Text:**
```python
import requests

response = requests.post(
    'http://localhost:8093/api/v1/translate',
    json={
        'text': 'What are business requirements?',
        'source_language': 'en',
        'target_language': 'ne'
    }
)

translation = response.json()['data']['translated_text']
print(f"Translation: {translation}")
```

### JavaScript Examples

**Search Query:**
```javascript
const searchQuery = async (query) => {
  const response = await fetch('http://localhost:8093/api/v1/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query: query,
      language: 'en',
      max_results: 5
    })
  });
  
  const data = await response.json();
  return data.data.results;
};

// Usage
searchQuery('What are business requirements?')
  .then(results => {
    results.forEach(result => {
      console.log(`Text: ${result.text}`);
      console.log(`Source: ${result.source.title}`);
    });
  });
```

## 🔄 Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Search requests**: 100 requests per minute per IP
- **Upload requests**: 10 requests per minute per IP
- **Translation requests**: 50 requests per minute per IP

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 📈 WebSocket Support

For real-time updates (future feature):

```javascript
const ws = new WebSocket('ws://localhost:8093/ws');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Subscribe to document processing updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'document_processing',
  doc_id: 'doc_123'
}));
```

---

**Next**: Explore [Configuration Guide](../configuration.md) or [Troubleshooting](../troubleshooting.md).
