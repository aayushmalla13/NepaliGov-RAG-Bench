# Configuration Guide

## ⚙️ Overview

The NepaliGov-RAG-Bench system is highly configurable through various configuration files. This guide explains how to customize the system behavior, performance, and features.

## 📁 Configuration Structure

```
config/
├── cp6_synonyms.json          # Synonym mappings
├── cp9_mappings.json          # Language mappings
├── cp10_thresholds.yaml       # Quality thresholds
├── cp11_5_ui_translation.yaml # UI translations
└── production.yml             # Production settings
```

## 🔧 Core Configuration Files

### cp10_thresholds.yaml

Controls quality thresholds and processing parameters.

```yaml
# OCR Processing Thresholds
ocr:
  minimum_confidence: 0.8      # Minimum OCR confidence
  fallback_threshold: 0.6      # Threshold for fallback OCR
  quality_check: true          # Enable quality assessment
  max_retries: 3               # Maximum OCR retry attempts

# Search Configuration
search:
  minimum_similarity: 0.7      # Minimum similarity score
  maximum_results: 10          # Default max results
  chunk_size: 512              # Text chunk size for embeddings
  overlap_size: 50             # Overlap between chunks

# Translation Settings
translation:
  minimum_quality: 0.85        # Minimum translation quality
  use_fallback: true           # Use fallback translation service
  cache_translations: true     # Cache translation results
  max_length: 5000             # Maximum text length for translation

# Processing Limits
processing:
  batch_size: 5                # Documents per batch
  max_workers: 4               # Maximum worker threads
  timeout: 300                 # Processing timeout (seconds)
  memory_limit: "8GB"          # Memory usage limit

# Performance Tuning
performance:
  faiss_index_type: "IndexFlatIP"  # FAISS index type
  embedding_model: "all-MiniLM-L6-v2"  # Embedding model
  cache_size: 1000             # Cache size for embeddings
  enable_gpu: false            # Enable GPU acceleration
```

### cp6_synonyms.json

Defines synonym mappings for improved search accuracy.

```json
{
  "business": [
    "enterprise",
    "company",
    "corporation",
    "firm",
    "organization"
  ],
  "license": [
    "permit",
    "authorization",
    "approval",
    "certificate",
    "clearance"
  ],
  "government": [
    "state",
    "public",
    "official",
    "administrative",
    "bureaucratic"
  ],
  "registration": [
    "enrollment",
    "enlistment",
    "enrollment",
    "signup",
    "application"
  ],
  "tax": [
    "duty",
    "levy",
    "tariff",
    "assessment",
    "contribution"
  ],
  "ministry": [
    "department",
    "agency",
    "bureau",
    "office",
    "administration"
  ]
}
```

### cp9_mappings.json

Language mapping configurations and preferences.

```json
{
  "language_codes": {
    "en": {
      "name": "English",
      "native_name": "English",
      "iso_code": "en",
      "supports_ocr": true,
      "supports_translation": true
    },
    "ne": {
      "name": "Nepali",
      "native_name": "नेपाली",
      "iso_code": "ne",
      "supports_ocr": true,
      "supports_translation": true
    }
  },
  "translation_pairs": [
    ["en", "ne"],
    ["ne", "en"]
  ],
  "default_language": "en",
  "fallback_language": "en"
}
```

### cp11_5_ui_translation.yaml

UI element translations for bilingual support.

```yaml
# English UI Elements
en:
  search_placeholder: "Ask a question about government documents..."
  search_button: "Search"
  language_selector: "Language"
  upload_document: "Upload Document"
  quick_questions: "Quick Questions"
  no_results: "No results found. Try rephrasing your question."
  loading: "Processing your request..."
  error: "An error occurred. Please try again."

# Nepali UI Elements
ne:
  search_placeholder: "सरकारी कागजातका बारेमा प्रश्न सोध्नुहोस्..."
  search_button: "खोज्नुहोस्"
  language_selector: "भाषा"
  upload_document: "कागजात अपलोड गर्नुहोस्"
  quick_questions: "छिटो प्रश्नहरू"
  no_results: "कुनै परिणाम फेला परेन। कृपया आफ्नो प्रश्न फेरि लेख्नुहोस्।"
  loading: "तपाईंको अनुरोध प्रशोधन गरिँदैछ..."
  error: "एउटा त्रुटि भयो। कृपया फेरि प्रयास गर्नुहोस्।"
```

## 🌐 Environment Configuration

### .env File

Create a `.env` file in the project root for environment-specific settings.

```bash
# Application Settings
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///data/app.db
# For PostgreSQL: postgresql://user:password@localhost/dbname

# Cache Configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# OCR Configuration
OCR_ENGINE=paddle
TESSERACT_PATH=/usr/bin/tesseract

# Translation Configuration
TRANSLATION_SERVICE=marianmt
GOOGLE_TRANSLATE_API_KEY=your-api-key

# Performance Settings
FAISS_CPU_ONLY=1
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security Settings
CORS_ORIGINS=*
RATE_LIMIT=100

# File Upload Settings
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=pdf
UPLOAD_FOLDER=data/uploads
```

## 🔧 Advanced Configuration

### Production Configuration

Create `config/production.yml` for production settings.

```yaml
# Production Settings
app:
  host: "0.0.0.0"
  port: 8093
  debug: false
  threaded: true

# Database Settings
database:
  url: "postgresql://user:password@localhost/nepaligov_rag"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30

# Cache Settings
cache:
  redis_url: "redis://localhost:6379"
  ttl: 3600
  max_connections: 100

# Security Settings
security:
  secret_key: "production-secret-key"
  cors_origins: ["https://yourdomain.com"]
  rate_limit: 1000
  session_timeout: 3600

# Performance Settings
performance:
  workers: 4
  timeout: 30
  keepalive: 2
  max_requests: 1000

# Monitoring Settings
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

### Development Configuration

Create `config/development.yml` for development settings.

```yaml
# Development Settings
app:
  host: "127.0.0.1"
  port: 8093
  debug: true
  reload: true

# Database Settings
database:
  url: "sqlite:///data/dev.db"
  echo: true

# Cache Settings
cache:
  redis_url: "redis://localhost:6379"
  ttl: 300

# Logging Settings
logging:
  level: "DEBUG"
  file: "logs/dev.log"
  console: true

# Development Features
features:
  hot_reload: true
  debug_toolbar: true
  profiling: true
```

## 📊 Performance Tuning

### Memory Optimization

```yaml
# config/performance.yml
memory:
  max_memory_usage: "8GB"
  gc_threshold: 0.8
  cache_cleanup_interval: 300

# Reduce memory usage
processing:
  batch_size: 1
  chunk_size: 256
  max_workers: 2

# FAISS Configuration
faiss:
  index_type: "IndexFlatIP"
  use_gpu: false
  memory_map: true
```

### CPU Optimization

```yaml
# config/cpu.yml
cpu:
  max_threads: 4
  affinity: [0, 1, 2, 3]
  
# OCR Optimization
ocr:
  use_gpu: false
  batch_size: 1
  timeout: 60

# Translation Optimization
translation:
  batch_size: 1
  cache_size: 100
  timeout: 30
```

## 🔍 Search Configuration

### Embedding Models

```yaml
# config/embeddings.yml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  max_length: 512
  batch_size: 32
  
# Alternative models
alternative_models:
  - "sentence-transformers/all-mpnet-base-v2"
  - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  - "sentence-transformers/distiluse-base-multilingual-cased"
```

### Search Parameters

```yaml
# config/search.yml
search:
  default_top_k: 10
  max_top_k: 50
  similarity_threshold: 0.7
  
# Query Processing
query_processing:
  max_length: 1000
  remove_stopwords: true
  normalize_text: true
  
# Result Ranking
ranking:
  similarity_weight: 0.7
  recency_weight: 0.2
  authority_weight: 0.1
```

## 🌐 Language Configuration

### Translation Services

```yaml
# config/translation.yml
translation:
  primary_service: "marianmt"
  fallback_service: "google"
  
# MarianMT Configuration
marianmt:
  model: "Helsinki-NLP/opus-mt-en-ne"
  max_length: 5000
  batch_size: 1
  
# Google Translate Configuration
google_translate:
  api_key: "your-api-key"
  project_id: "your-project-id"
  region: "global"
```

### Language Detection

```yaml
# config/language_detection.yml
language_detection:
  model: "langdetect"
  confidence_threshold: 0.8
  fallback_language: "en"
  
# Supported Languages
supported_languages:
  - code: "en"
    name: "English"
    weight: 1.0
  - code: "ne"
    name: "Nepali"
    weight: 1.0
```

## 📁 File Processing Configuration

### OCR Settings

```yaml
# config/ocr.yml
ocr:
  primary_engine: "paddle"
  fallback_engine: "tesseract"
  
# PaddleOCR Configuration
paddle:
  use_angle_cls: true
  lang: "en"
  use_gpu: false
  show_log: false
  
# Tesseract Configuration
tesseract:
  lang: "eng+nep"
  psm: 6
  oem: 3
```

### Document Processing

```yaml
# config/document_processing.yml
document_processing:
  max_file_size: "50MB"
  allowed_formats: ["pdf"]
  temp_directory: "data/temp"
  
# Text Processing
text_processing:
  min_length: 10
  max_length: 10000
  remove_headers: true
  remove_footers: true
  
# Quality Assessment
quality_assessment:
  min_confidence: 0.8
  check_coherence: true
  check_completeness: true
```

## 🔒 Security Configuration

### Authentication

```yaml
# config/security.yml
security:
  enable_auth: false
  jwt_secret: "your-jwt-secret"
  jwt_expiry: 3600
  
# Rate Limiting
rate_limiting:
  enabled: true
  requests_per_minute: 100
  burst_limit: 200
  
# CORS Configuration
cors:
  enabled: true
  origins: ["*"]
  methods: ["GET", "POST", "PUT", "DELETE"]
  headers: ["Content-Type", "Authorization"]
```

### Data Protection

```yaml
# config/data_protection.yml
data_protection:
  encrypt_at_rest: true
  encrypt_in_transit: true
  
# File Security
file_security:
  scan_uploads: true
  max_file_size: "50MB"
  allowed_extensions: ["pdf"]
  
# Logging Security
logging_security:
  mask_sensitive_data: true
  log_level: "INFO"
  audit_log: true
```

## 📊 Monitoring Configuration

### Metrics Collection

```yaml
# config/monitoring.yml
monitoring:
  enable_metrics: true
  metrics_port: 9090
  
# Performance Metrics
performance_metrics:
  response_time: true
  memory_usage: true
  cpu_usage: true
  disk_usage: true
  
# Business Metrics
business_metrics:
  query_count: true
  document_count: true
  user_count: true
  error_rate: true
```

### Logging Configuration

```yaml
# config/logging.yml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
# Log Files
log_files:
  app: "logs/app.log"
  error: "logs/error.log"
  access: "logs/access.log"
  audit: "logs/audit.log"
  
# Log Rotation
log_rotation:
  max_bytes: 10485760  # 10MB
  backup_count: 5
  when: "midnight"
```

## 🔄 Configuration Management

### Configuration Loading

```python
# config/loader.py
import yaml
import json
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir="config"):
        self.config_dir = Path(config_dir)
        self.config = {}
    
    def load_config(self):
        """Load all configuration files"""
        # Load YAML files
        for yaml_file in self.config_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                self.config[yaml_file.stem] = yaml.safe_load(f)
        
        # Load JSON files
        for json_file in self.config_dir.glob("*.json"):
            with open(json_file) as f:
                self.config[json_file.stem] = json.load(f)
        
        # Load environment variables
        self.load_env_config()
        
        return self.config
    
    def load_env_config(self):
        """Load configuration from environment variables"""
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith("NEPALIGOV_"):
                config_key = key[10:].lower()
                env_config[config_key] = value
        
        self.config["environment"] = env_config
```

### Configuration Validation

```python
# config/validator.py
from jsonschema import validate, ValidationError

class ConfigValidator:
    def __init__(self):
        self.schemas = self.load_schemas()
    
    def validate_config(self, config):
        """Validate configuration against schemas"""
        errors = []
        
        for config_name, config_data in config.items():
            if config_name in self.schemas:
                try:
                    validate(config_data, self.schemas[config_name])
                except ValidationError as e:
                    errors.append(f"{config_name}: {e.message}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        return True
    
    def load_schemas(self):
        """Load JSON schemas for validation"""
        # Define schemas for each configuration file
        return {
            "cp10_thresholds": {
                "type": "object",
                "properties": {
                    "ocr": {"type": "object"},
                    "search": {"type": "object"},
                    "translation": {"type": "object"}
                },
                "required": ["ocr", "search", "translation"]
            }
        }
```

## 🔄 Configuration Updates

### Hot Reloading

```python
# config/hot_reload.py
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloader(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.yaml', '.json', '.yml')):
            print(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload_config()

class ConfigManager:
    def __init__(self):
        self.config = {}
        self.loader = ConfigLoader()
        self.validator = ConfigValidator()
        self.setup_hot_reload()
    
    def load_config(self):
        """Load and validate configuration"""
        self.config = self.loader.load_config()
        self.validator.validate_config(self.config)
        return self.config
    
    def reload_config(self):
        """Reload configuration on file changes"""
        try:
            new_config = self.loader.load_config()
            self.validator.validate_config(new_config)
            self.config = new_config
            print("Configuration reloaded successfully")
        except Exception as e:
            print(f"Configuration reload failed: {e}")
    
    def setup_hot_reload(self):
        """Setup hot reloading for configuration files"""
        event_handler = ConfigReloader(self)
        observer = Observer()
        observer.schedule(event_handler, "config", recursive=False)
        observer.start()
```

---

**Next**: Check out [Troubleshooting Guide](troubleshooting.md) or [API Documentation](api/index.md).
