# Troubleshooting Guide

## 🚨 Common Issues and Solutions

This guide helps you diagnose and resolve common issues with the NepaliGov-RAG-Bench system.

## 🔧 Installation Issues

### Python Version Issues

**Problem**: `python: command not found` or wrong Python version

**Solutions**:
```bash
# Check Python version
python --version

# Install Python 3.12 (Ubuntu/Debian)
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install Python 3.12 (macOS)
brew install python@3.12

# Install Python 3.12 (Windows)
# Download from https://www.python.org/downloads/
```

### Virtual Environment Issues

**Problem**: `ModuleNotFoundError` or dependency conflicts

**Solutions**:
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Permission Issues

**Problem**: Permission denied errors during installation

**Solutions**:
```bash
# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER /path/to/project
chmod -R 755 /path/to/project

# Use user install (avoid sudo)
pip install --user -r requirements.txt
```

## 🖥️ Application Startup Issues

### Port Already in Use

**Problem**: `Address already in use` error

**Solutions**:
```bash
# Find process using port 8093
lsof -ti:8093
# or
netstat -tulpn | grep 8093

# Kill the process
kill -9 $(lsof -ti:8093)

# Or use a different port
export FLASK_PORT=8094
python enhanced_web_app.py
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'flask'`

**Solutions**:
```bash
# Check virtual environment is activated
which python
# Should show path to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt

# Check if packages are installed
pip list | grep flask
```

### Memory Issues

**Problem**: `MemoryError` or application crashes

**Solutions**:
```bash
# Check available memory
free -h

# Reduce memory usage
export FAISS_CPU_ONLY=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Close other applications
# Restart with more memory if possible
```

## 🔍 OCR Processing Issues

### Tesseract Not Found

**Problem**: `tesseract: command not found`

**Solutions**:
```bash
# Install Tesseract (Ubuntu/Debian)
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep

# Install Tesseract (macOS)
brew install tesseract tesseract-lang

# Install Tesseract (Windows)
# Download from https://github.com/UB-Mannheim/tesseract/wiki

# Verify installation
tesseract --version
```

### PaddleOCR Issues

**Problem**: PaddleOCR fails to load or process documents

**Solutions**:
```bash
# Update PaddleOCR
pip install --upgrade paddlepaddle paddleocr

# Clear PaddleOCR cache
rm -rf ~/.paddleocr/

# Use CPU-only version if GPU issues
pip uninstall paddlepaddle-gpu
pip install paddlepaddle
```

### Poor OCR Quality

**Problem**: Low-quality text extraction

**Solutions**:
```bash
# Check document quality
# - Ensure PDF is text-based, not image-based
# - Use high-resolution scans
# - Avoid handwritten documents

# Adjust OCR settings in config/cp10_thresholds.yaml
ocr:
  minimum_confidence: 0.7  # Lower threshold
  fallback_threshold: 0.5
```

## 🔍 Search Issues

### No Results Found

**Problem**: Search returns no results

**Solutions**:
```bash
# Check if corpus is loaded
ls -la data/real_corpus.parquet

# Rebuild corpus if needed
python create_real_corpus.py

# Check search thresholds in config/cp10_thresholds.yaml
search:
  minimum_similarity: 0.5  # Lower threshold
  maximum_results: 20

# Try broader queries
# Instead of: "specific business license requirement"
# Try: "business license" or "license requirements"
```

### Slow Search Performance

**Problem**: Search takes too long

**Solutions**:
```bash
# Check system resources
htop  # or top

# Optimize FAISS index
# Rebuild index with smaller chunks
python scripts/rebuild_index.py --chunk_size 256

# Use CPU-only FAISS
export FAISS_CPU_ONLY=1
```

### Poor Search Relevance

**Problem**: Search results are not relevant

**Solutions**:
```bash
# Update synonyms in config/cp6_synonyms.json
{
  "business": ["enterprise", "company", "corporation", "firm"],
  "license": ["permit", "authorization", "approval", "certificate"]
}

# Adjust embedding model
# Consider using a different sentence-transformer model
# Update in enhanced_web_app.py
```

## 🌐 Translation Issues

### Translation Service Not Working

**Problem**: Translation fails or returns empty results

**Solutions**:
```bash
# Check internet connection
ping google.com

# Clear translation cache
rm -rf cache/translations/

# Update translation models
pip install --upgrade transformers

# Check translation service in config
translation:
  service: "marianmt"  # or "google" if available
```

### Poor Translation Quality

**Problem**: Translations are inaccurate or awkward

**Solutions**:
```bash
# Adjust translation thresholds in config/cp10_thresholds.yaml
translation:
  minimum_quality: 0.8  # Lower threshold
  use_fallback: true

# Update translation models
pip install --upgrade transformers torch

# Consider using different translation service
# Update in enhanced_web_app.py
```

## 💾 Data Issues

### Corpus Not Loading

**Problem**: `FileNotFoundError: data/real_corpus.parquet`

**Solutions**:
```bash
# Check if data directory exists
ls -la data/

# Create corpus if missing
python create_real_corpus.py

# Check file permissions
chmod 644 data/real_corpus.parquet
```

### Database Connection Issues

**Problem**: Database connection errors

**Solutions**:
```bash
# Check database file
ls -la data/app.db

# Recreate database
rm data/app.db
python -c "from enhanced_web_app import create_app; app = create_app(); app.app_context().push(); db.create_all()"

# Check disk space
df -h
```

### Cache Issues

**Problem**: Stale cache causing incorrect results

**Solutions**:
```bash
# Clear all caches
rm -rf cache/
mkdir cache

# Clear specific caches
rm -rf cache/embeddings/
rm -rf cache/translations/
rm -rf cache/responses/
```

## 🔄 Performance Issues

### High CPU Usage

**Problem**: System becomes slow or unresponsive

**Solutions**:
```bash
# Limit CPU usage
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Use CPU-only FAISS
export FAISS_CPU_ONLY=1

# Reduce batch sizes in config
processing:
  batch_size: 1
  max_workers: 2
```

### High Memory Usage

**Problem**: Out of memory errors

**Solutions**:
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Reduce memory usage
export FAISS_CPU_ONLY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Process documents in smaller batches
# Update batch_size in configuration
```

### Slow Document Processing

**Problem**: Documents take too long to process

**Solutions**:
```bash
# Check system resources
htop

# Optimize OCR settings
# Use faster OCR engine for simple documents
ocr:
  engine: "tesseract"  # Faster than PaddleOCR for simple docs

# Process in parallel
# Update max_workers in configuration
```

## 🌐 Network Issues

### Model Download Issues

**Problem**: Models fail to download

**Solutions**:
```bash
# Check internet connection
ping huggingface.co

# Clear model cache
rm -rf ~/.cache/huggingface/

# Use mirror or proxy if needed
export HF_ENDPOINT=https://hf-mirror.com

# Download models manually
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

### API Connection Issues

**Problem**: API requests fail

**Solutions**:
```bash
# Check if server is running
curl http://localhost:8093/health

# Check firewall settings
sudo ufw status

# Check port availability
netstat -tulpn | grep 8093

# Restart server
pkill -f enhanced_web_app.py
python enhanced_web_app.py
```

## 🔧 Configuration Issues

### Invalid Configuration

**Problem**: Configuration errors causing startup failures

**Solutions**:
```bash
# Validate YAML configuration
python -c "import yaml; yaml.safe_load(open('config/cp10_thresholds.yaml'))"

# Check JSON configuration
python -c "import json; json.load(open('config/cp6_synonyms.json'))"

# Reset to defaults
git checkout config/
```

### Environment Variable Issues

**Problem**: Environment variables not being read

**Solutions**:
```bash
# Check environment variables
env | grep FLASK

# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=1

# Use .env file
echo "FLASK_ENV=development" > .env
echo "FLASK_DEBUG=1" >> .env
```

## 📊 Monitoring and Debugging

### Enable Debug Mode

**Solutions**:
```bash
# Set debug mode
export FLASK_DEBUG=1
export FLASK_ENV=development

# Enable verbose logging
export LOG_LEVEL=DEBUG

# Start with debug output
python enhanced_web_app.py --debug
```

### Check Logs

**Solutions**:
```bash
# Check application logs
tail -f server.log

# Check system logs
journalctl -u your-service-name

# Check error logs
grep -i error logs/*.log
```

### Performance Monitoring

**Solutions**:
```bash
# Monitor system resources
htop

# Monitor disk usage
df -h
du -sh data/

# Monitor network
netstat -i
```

## 🆘 Emergency Recovery

### Complete Reset

**Solutions**:
```bash
# Stop all processes
pkill -f enhanced_web_app.py

# Clear all data
rm -rf data/
rm -rf cache/
rm -rf logs/

# Recreate directories
mkdir -p data/{raw_documents,processed,corpus,cache}
mkdir -p cache/{embeddings,translations,responses}

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Restart application
python enhanced_web_app.py
```

### Backup and Restore

**Solutions**:
```bash
# Create backup
tar -czf backup-$(date +%Y%m%d).tar.gz data/ config/

# Restore backup
tar -xzf backup-20240101.tar.gz

# Restore specific files
tar -xzf backup-20240101.tar.gz data/real_corpus.parquet
```

## 📞 Getting Help

### Before Asking for Help

1. **Check this guide** for your specific issue
2. **Check the logs** for error messages
3. **Try the solutions** listed above
4. **Document the problem** with:
   - Error messages
   - System information
   - Steps to reproduce

### System Information

When reporting issues, include:

```bash
# System information
uname -a
python --version
pip list | grep -E "(flask|transformers|faiss|paddle)"

# Application status
curl http://localhost:8093/health

# Resource usage
free -h
df -h
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| `Address already in use` | Port conflict | Kill existing process or change port |
| `MemoryError` | Insufficient RAM | Close other apps or increase RAM |
| `tesseract: command not found` | Missing OCR engine | Install Tesseract |
| `No module named 'paddleocr'` | Missing OCR library | `pip install paddleocr` |
| `FileNotFoundError: data/real_corpus.parquet` | Missing corpus | `python create_real_corpus.py` |

---

**Still having issues?** Check the [FAQ](faq.md) or [API Documentation](api/index.md) for more help.
