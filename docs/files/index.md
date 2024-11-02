# File Reference

This section provides detailed documentation for each file in the NepaliGov-RAG-Bench project, including their purpose, implementation details, and impact on the overall system.

## 📁 Directory Structure

```
NepaliGov-RAG-Bench/
├── 📁 api/                    # API implementations
├── 📁 cache/                  # Caching layer
├── 📁 config/                 # Configuration files
├── 📁 data/                   # Document storage and corpus
├── 📁 docs/                   # Documentation
├── 📁 ops/                    # Operational scripts
├── 📁 scripts/                # Utility scripts
├── 📁 src/                    # Source code
├── 📁 ui/                     # UI components
├── 📄 enhanced_web_app.py     # Main web application
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md              # Project overview
```

## 🔧 Core Application Files

### enhanced_web_app.py
**Purpose**: Main web application implementing the complete RAG system
**Size**: ~100KB (2,500+ lines)
**Key Components**:
- HTTP server with Flask
- Document processing pipeline
- Vector search engine
- Bilingual translation system
- Modern web UI

**Implementation Details**:
```python
class EnhancedWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.search_engine = VectorSearchEngine()
        self.translator = BilingualTranslator()
        self.document_processor = DocumentProcessor()
    
    def setup_routes(self):
        # API endpoints
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            # Handle user queries
            
        @self.app.route('/upload', methods=['POST'])
        def upload_document():
            # Handle document uploads
```

**Impact**: 
- Central component of the entire system
- Handles all user interactions
- Integrates all 12 checkpoint stages
- Provides both API and web interface

### requirements.txt
**Purpose**: Python package dependencies
**Content**: Lists all required Python packages with versions
**Key Dependencies**:
- `flask`: Web framework
- `transformers`: NLP models
- `faiss-cpu`: Vector search
- `paddleocr`: OCR processing
- `sentence-transformers`: Text embeddings

**Impact**: Ensures consistent environment across deployments

## 📁 API Directory

### api/nepal_gov_api.py
**Purpose**: Core API implementation for government document processing
**Key Features**:
- Document ingestion endpoints
- Search and retrieval APIs
- Language processing
- Authentication and authorization

### api/fixed_nepal_gov_api.py
**Purpose**: Fixed version of the Nepal government API with bug fixes
**Improvements**:
- Better error handling
- Performance optimizations
- Enhanced security
- Improved response formatting

## 📁 Configuration Directory

### config/cp6_synonyms.json
**Purpose**: Synonym mappings for improved search accuracy
**Content**: JSON mapping of synonyms and related terms
**Example**:
```json
{
  "business": ["enterprise", "company", "corporation"],
  "license": ["permit", "authorization", "approval"]
}
```

### config/cp9_mappings.json
**Purpose**: Language mapping configurations
**Content**: Language code mappings and preferences

### config/cp10_thresholds.yaml
**Purpose**: Quality thresholds for various system components
**Content**: YAML configuration for quality thresholds
```yaml
ocr:
  minimum_confidence: 0.8
  fallback_threshold: 0.6

search:
  minimum_similarity: 0.7
  maximum_results: 10

translation:
  minimum_quality: 0.85
```

### config/cp11_5_ui_translation.yaml
**Purpose**: UI translation configurations
**Content**: UI element translations for bilingual support

## 📁 Data Directory

### data/real_corpus.parquet
**Purpose**: Main document corpus in Parquet format
**Structure**: 
- `doc_id`: Unique document identifier
- `text`: Extracted document text
- `metadata`: Document metadata
- `embeddings`: Vector embeddings

### data/seed_manifest.yaml
**Purpose**: Manifest of seed documents for initial corpus
**Content**: List of initial documents with metadata
```yaml
documents:
  - id: "doc_001"
    title: "Business Registration Guide"
    url: "https://example.com/business-guide.pdf"
    authority: "Ministry of Industry"
    type: "policy"
```

### data/index.manifest.json
**Purpose**: Index manifest tracking processed documents
**Content**: Processing status and metadata for each document

## 📁 Operations Directory

### ops/lineage_index.py
**Purpose**: Manages file-to-document lineage tracking
**Key Features**:
- Document version tracking
- Change detection
- Rollback capabilities
- Audit trail maintenance

### ops/cron_ingest.py
**Purpose**: Automated document ingestion and processing
**Features**:
- Incremental processing
- Change detection
- Quality monitoring
- Automatic rollback on failures

### ops/safety_monitor.py
**Purpose**: Monitors system safety and quality metrics
**Features**:
- Performance monitoring
- Quality assessment
- Anomaly detection
- Alert generation

## 📁 Scripts Directory

### scripts/add_url.py
**Purpose**: Utility script to add documents from URLs
**Features**:
- PDF download from URLs
- SHA-256 checksum calculation
- Manifest updates
- Batch processing

### scripts/extract_pdf_titles.py
**Purpose**: Extract and standardize PDF titles
**Features**:
- Title extraction from metadata
- Title normalization
- Duplicate detection
- Quality assessment

## 📁 Source Directory

### src/cp1_authority.py
**Purpose**: Authority detection implementation
**Features**:
- Government authority classification
- Authority validation
- Metadata extraction

### src/cp2_types.py
**Purpose**: Document type classification
**Features**:
- Document type detection
- Content filtering
- Priority calculation

### src/cp3_ocr.py
**Purpose**: OCR processing implementation
**Features**:
- Multi-stage OCR processing
- Quality assessment
- Fallback mechanisms

### src/cp4_store.py
**Purpose**: Vector storage management
**Features**:
- Embedding generation
- FAISS index management
- Document storage

### src/cp5_search.py
**Purpose**: Semantic search implementation
**Features**:
- Vector similarity search
- Result ranking
- Query processing

## 📁 UI Directory

### ui/templates/
**Purpose**: HTML templates for web interface
**Files**:
- `index.html`: Main application interface
- `search.html`: Search results page
- `upload.html`: Document upload interface

### ui/static/
**Purpose**: Static assets (CSS, JavaScript, images)
**Files**:
- `css/main.css`: Main stylesheet
- `js/app.js`: Client-side JavaScript
- `images/`: UI images and icons

## 🔧 Utility Files

### run_with_reload.py
**Purpose**: Development server with auto-reload
**Features**:
- File watching
- Automatic restart on changes
- Development-friendly configuration

### force_reload_corpus.py
**Purpose**: Force reload document corpus
**Features**:
- Corpus validation
- Index rebuilding
- Cache clearing

### process_uploaded_cv.py
**Purpose**: Process uploaded CV documents
**Features**:
- CV-specific processing
- Format standardization
- Metadata extraction

## 📊 Testing and Analysis Files

### Analysis Files
**Purpose**: System analysis and optimization scripts
**Features**:
- Performance analysis
- Quality assessment
- System optimization recommendations

### dynamic_search.py
**Purpose**: Search performance analysis and optimization
**Features**:
- Analyzes how well the search system performs
- Identifies common search patterns
- Provides recommendations for improvement

### create_real_corpus.py
**Purpose**: Build a collection of real government documents
**Features**:
- Downloads documents from official sources
- Processes them through the system
- Creates a searchable document collection

## 📋 Documentation Files

### README.md
**Purpose**: Project overview and quick start guide
**Content**:
- Project description
- Installation instructions
- Usage examples
- Contributing guidelines

### ENHANCEMENT_SUMMARY.md
**Purpose**: Summary of system enhancements
**Content**:
- Enhancement descriptions
- Performance improvements
- Feature additions

### CP3_ENHANCEMENTS_SUMMARY.md
**Purpose**: CP3 OCR enhancements summary
**Content**:
- OCR improvements
- Quality metrics
- Performance gains

### CP11_ENHANCEMENT_REPORT.md
**Purpose**: CP11 API and UI enhancement report
**Content**:
- API improvements
- UI enhancements
- User experience improvements

## 🔧 Build and Deployment Files

### Makefile
**Purpose**: Build automation and common tasks
**Targets**:
- `install`: Install dependencies
- `test`: Run tests
- `build`: Build application
- `deploy`: Deploy to production

### launch_web.sh
**Purpose**: Web application launcher script
**Features**:
- Environment setup
- Dependency checking
- Application startup

### pyproject.toml
**Purpose**: Python project configuration
**Content**:
- Project metadata
- Build configuration
- Development dependencies

## 📊 Log Files

### server.log
**Purpose**: Server operation logs
**Content**:
- Request logs
- Error logs
- Performance metrics

### web_app.log
**Purpose**: Web application logs
**Content**:
- Application events
- User interactions
- System status

### reload.log
**Purpose**: Auto-reload operation logs
**Content**:
- File change events
- Restart events
- Error tracking

## 🎯 File Impact Analysis

### Critical Files (System Core)
- `enhanced_web_app.py`: **Critical** - Main application
- `requirements.txt`: **Critical** - Dependencies
- `data/real_corpus.parquet`: **Critical** - Document corpus

### Important Files (Key Features)
- `api/*.py`: **Important** - API functionality
- `src/cp*.py`: **Important** - Core processing
- `config/*.yaml`: **Important** - Configuration

### Supporting Files (Utilities)
- `scripts/*.py`: **Supporting** - Utilities
- `demo_*.py`: **Supporting** - Demonstrations
- `*.md`: **Supporting** - Documentation

## 🔄 File Dependencies

### Core Dependencies
```
enhanced_web_app.py
├── api/nepal_gov_api.py
├── src/cp1_authority.py
├── src/cp2_types.py
├── src/cp3_ocr.py
├── src/cp4_store.py
├── src/cp5_search.py
├── config/cp*.yaml
└── data/real_corpus.parquet
```

### Processing Dependencies
```
ops/cron_ingest.py
├── scripts/add_url.py
├── src/cp3_ocr.py
├── src/cp4_store.py
├── data/seed_manifest.yaml
└── data/index.manifest.json
```

---

**Next**: Explore [API Documentation](api/index.md) or [Configuration Guide](configuration.md).
