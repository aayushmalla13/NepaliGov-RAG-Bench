# System Architecture

## 🏗️ Architecture Overview

The NepaliGov-RAG-Bench system follows a modular, phase-based architecture designed for scalability, maintainability, and incremental development. The architecture is built around 12 distinct phases (Phase 1-12), each representing a specific functional component.

## 🐳 Docker Architecture

The project is containerized using Docker for consistent deployment across different environments:

### Container Structure

```mermaid
graph TB
    subgraph "Docker Environment"
        A[nepali-gov-rag-bench Container]
        B[nepali-gov-docs Container]
        C[Volume Mounts]
    end
    
    A --> D[Python 3.11 Runtime]
    A --> E[Tesseract OCR + Nepali Support]
    A --> F[FAISS Vector Search]
    A --> G[Health Monitoring]
    
    B --> H[MkDocs Server]
    B --> I[Material Theme]
    B --> J[Live Documentation]
    
    C --> K[./data → Document Corpus]
    C --> L[./cache → Translation Cache]
    C --> M[./config → Configuration]
    C --> N[./mlruns → MLflow Data]
```

### Container Features

- **Multi-stage builds** for optimized image size
- **Non-root user** for security
- **Health checks** for monitoring
- **Volume persistence** for data retention
- **Network isolation** between services
- **Automatic restart** policies

## 🔄 Phase Architecture

### Phase 1: Authority Detection
**Purpose**: Identify and validate government document authorities
**Components**:
- Document metadata extraction
- Authority classification
- Validation rules engine

```mermaid
graph LR
    A[PDF Input] --> B[Metadata Extraction]
    B --> C[Authority Classifier]
    C --> D[Validation Engine]
    D --> E[Authority Tagged Doc]
```

### Phase 2: Document Types & Filtering
**Purpose**: Categorize documents by type and apply filtering rules
**Components**:
- Document type classifier
- Content filtering engine
- Quality assessment module

### Phase 3: OCR Processing
**Purpose**: Extract text from PDF documents using multi-stage OCR
**Components**:
- Primary OCR engine (PaddleOCR)
- Fallback OCR engine (Tesseract)
- Quality assessment and validation

```mermaid
graph TB
    A[PDF Document] --> B[Image Extraction]
    B --> C[Primary OCR]
    C --> D{Quality Check}
    D -->|Good| E[Text Output]
    D -->|Poor| F[Fallback OCR]
    F --> G[Quality Check 2]
    G -->|Good| E
    G -->|Poor| H[Manual Review Queue]
```

### Phase 4: Vector Storage
**Purpose**: Create and manage vector embeddings for semantic search
**Components**:
- Text preprocessing pipeline
- Embedding generation (sentence-transformers)
- FAISS index management

### Phase 5: Semantic Search
**Purpose**: Perform semantic similarity search on document vectors
**Components**:
- Query embedding generation
- FAISS similarity search
- Result ranking and filtering

### Phase 6: Language Preference
**Purpose**: Handle user language preferences and document language detection
**Components**:
- Language detection engine
- User preference management
- Language-specific processing

### Phase 7: QA Citation
**Purpose**: Generate answers with proper source citations
**Components**:
- Context assembly
- LLM integration
- Citation generation

### Phase 8: Metrics & Evaluation
**Purpose**: Collect and analyze system performance metrics
**Components**:
- Metrics collection engine
- Performance monitoring
- Quality assessment

### Phase 9: Language Selection
**Purpose**: Manage language selection for responses
**Components**:
- Language preference engine
- Response language selection
- Translation management

### Phase 10: Refusal Handling
**Purpose**: Handle queries that cannot be answered
**Components**:
- Refusal detection
- Alternative suggestions
- User guidance

### Phase 11: API & UI
**Purpose**: Provide web interface and API endpoints
**Components**:
- RESTful API
- Web interface
- Real-time processing

### Phase 12: Batch Processing
**Purpose**: Handle incremental document updates and processing
**Components**:
- Change detection
- Delta processing
- Rollback capabilities

## 🗄️ Data Architecture

### Document Storage Structure

```
data/
├── raw_documents/           # Original PDF files
│   ├── government/         # Government documents
│   ├── policies/           # Policy documents
│   └── reports/            # Official reports
├── processed/              # Processed documents
│   ├── ocr_output/         # OCR extracted text
│   ├── embeddings/         # Vector embeddings
│   └── metadata/           # Document metadata
├── corpus/                 # Searchable corpus
│   ├── parquet/            # Parquet format corpus
│   ├── json/               # JSON format corpus
│   └── faiss/              # FAISS indexes
└── cache/                  # Temporary cache files
    ├── embeddings/         # Cached embeddings
    ├── translations/       # Cached translations
    └── responses/          # Cached responses
```

### Database Schema

#### Document Metadata Table
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authority TEXT,
    document_type TEXT,
    language TEXT,
    file_path TEXT,
    file_size INTEGER,
    upload_date TIMESTAMP,
    processed_date TIMESTAMP,
    ocr_quality REAL,
    embedding_status TEXT,
    checksum TEXT
);
```

#### Query Log Table
```sql
CREATE TABLE query_logs (
    query_id TEXT PRIMARY KEY,
    user_id TEXT,
    query_text TEXT,
    query_language TEXT,
    response_time REAL,
    result_count INTEGER,
    timestamp TIMESTAMP,
    success BOOLEAN
);
```

## 🔧 Component Architecture

### Core Components

#### 1. Document Processing Engine
```python
class DocumentProcessor:
    def __init__(self):
        self.ocr_engine = PaddleOCR()
        self.text_processor = TextProcessor()
        self.embedding_generator = EmbeddingGenerator()
    
    def process_document(self, pdf_path):
        # Extract text using OCR
        text = self.ocr_engine.extract_text(pdf_path)
        
        # Process and clean text
        processed_text = self.text_processor.clean_text(text)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate(processed_text)
        
        return {
            'text': processed_text,
            'embeddings': embeddings,
            'metadata': self.extract_metadata(pdf_path)
        }
```

#### 2. Vector Search Engine
```python
class VectorSearchEngine:
    def __init__(self):
        self.faiss_index = None
        self.document_store = DocumentStore()
    
    def build_index(self, documents):
        # Create FAISS index from document embeddings
        vectors = [doc['embeddings'] for doc in documents]
        self.faiss_index = faiss.IndexFlatIP(vectors[0].shape[1])
        self.faiss_index.add(np.array(vectors))
    
    def search(self, query_embedding, top_k=5):
        # Perform similarity search
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), top_k
        )
        return self.document_store.get_by_indices(indices[0])
```

#### 3. Language Processing Engine
```python
class LanguageProcessor:
    def __init__(self):
        self.translator = MarianMTTranslator()
        self.language_detector = LanguageDetector()
    
    def detect_language(self, text):
        return self.language_detector.detect(text)
    
    def translate(self, text, target_language):
        if self.detect_language(text) == target_language:
            return text
        return self.translator.translate(text, target_language)
```

## 🌐 API Architecture

### RESTful API Design

#### Endpoints Structure
```
/api/v1/
├── documents/
│   ├── GET    /                    # List documents
│   ├── POST   /                    # Upload document
│   ├── GET    /{doc_id}            # Get document details
│   └── DELETE /{doc_id}            # Delete document
├── search/
│   ├── POST   /query               # Semantic search
│   ├── POST   /similar             # Find similar documents
│   └── GET    /suggestions         # Query suggestions
├── translate/
│   ├── POST   /text                # Translate text
│   └── GET    /languages           # Supported languages
└── health/
    ├── GET    /status              # System status
    └── GET    /metrics             # Performance metrics
```

#### Request/Response Format
```json
// Search Request
{
  "query": "What are the requirements for starting a business?",
  "language": "en",
  "max_results": 5,
  "include_sources": true
}

// Search Response
{
  "results": [
    {
      "text": "To start a business in Nepal...",
      "source": {
        "doc_id": "doc_123",
        "title": "Business Registration Guide",
        "page": 15,
        "similarity_score": 0.95
      }
    }
  ],
  "translated_query": "व्यापार सुरु गर्नका लागि के के आवश्यकता छ?",
  "response_time": 1.2,
  "total_results": 5
}
```

## 🔄 Processing Pipelines

### Document Ingestion Pipeline

```mermaid
flowchart TD
    A[PDF Upload] --> B[File Validation]
    B --> C[Metadata Extraction]
    C --> D[Authority Detection]
    D --> E[Document Classification]
    E --> F[OCR Processing]
    F --> G[Text Quality Check]
    G --> H{Quality OK?}
    H -->|Yes| I[Text Processing]
    H -->|No| J[Manual Review]
    I --> K[Embedding Generation]
    K --> L[Index Update]
    L --> M[Corpus Update]
    M --> N[Processing Complete]
```

### Query Processing Pipeline

```mermaid
flowchart TD
    A[User Query] --> B[Language Detection]
    B --> C[Query Preprocessing]
    C --> D[Embedding Generation]
    D --> E[Vector Search]
    E --> F[Result Ranking]
    F --> G[Context Assembly]
    G --> H[Answer Generation]
    H --> I[Translation if Needed]
    I --> J[Citation Addition]
    J --> K[Response Formatting]
    K --> L[Return to User]
```

## 🚀 Performance Architecture

### Caching Strategy

#### Multi-Level Caching
1. **Browser Cache**: Static assets and UI components
2. **CDN Cache**: Global content delivery
3. **Application Cache**: Query results and embeddings
4. **Database Cache**: Frequently accessed data

#### Cache Invalidation
```python
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_result(self, query_hash):
        return self.redis_client.get(f"query:{query_hash}")
    
    def cache_result(self, query_hash, result):
        self.redis_client.setex(
            f"query:{query_hash}", 
            self.cache_ttl, 
            json.dumps(result)
        )
```

### Load Balancing

#### Horizontal Scaling
```mermaid
graph TB
    A[Load Balancer] --> B[App Instance 1]
    A --> C[App Instance 2]
    A --> D[App Instance 3]
    
    B --> E[Shared Database]
    C --> E
    D --> E
    
    B --> F[Shared Cache]
    C --> F
    D --> F
```

## 🔒 Security Architecture

### Authentication & Authorization
- **JWT Tokens**: Stateless authentication
- **Role-based Access**: Different permission levels
- **API Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize user inputs

### Data Security
- **Encryption at Rest**: Database encryption
- **Encryption in Transit**: HTTPS/TLS
- **Access Logging**: Audit trail
- **Data Anonymization**: Privacy protection

## 📊 Monitoring Architecture

### Metrics Collection
```python
class MetricsCollector:
    def __init__(self):
        self.prometheus_client = PrometheusClient()
    
    def record_query_time(self, duration):
        self.prometheus_client.histogram(
            'query_duration_seconds', 
            duration
        )
    
    def record_document_processed(self):
        self.prometheus_client.counter(
            'documents_processed_total'
        ).inc()
```

### Health Checks
- **System Health**: CPU, memory, disk usage
- **Service Health**: API endpoint availability
- **Data Health**: Database connectivity and integrity
- **Performance Health**: Response times and throughput

## 🔧 Configuration Management

### Environment Configuration
```yaml
# config/production.yml
database:
  url: postgresql://user:pass@localhost/db
  pool_size: 20

cache:
  redis_url: redis://localhost:6379
  ttl: 3600

ocr:
  engine: paddle
  confidence_threshold: 0.8

api:
  rate_limit: 100
  timeout: 30
```

### Feature Flags
```python
class FeatureFlags:
    def __init__(self):
        self.flags = {
            'advanced_ocr': True,
            'real_time_translation': False,
            'beta_features': False
        }
    
    def is_enabled(self, feature):
        return self.flags.get(feature, False)
```

---

**Next**: Explore [Installation Guide](installation.md) or [API Documentation](api/index.md).
