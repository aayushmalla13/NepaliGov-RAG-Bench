# Installation Guide

Detailed installation instructions for NepaliGov-RAG-Bench.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for downloading models

## Installation Methods

### Method 1: Docker (Recommended)

The easiest way to get started is using Docker:

```bash
# Clone the repository
git clone https://github.com/aayushmalla13/NepaliGov-RAG-Bench.git
cd NepaliGov-RAG-Bench

# Build and start services
docker-compose up -d

# Verify installation
curl http://localhost:8000/health
```

### Method 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aayushmalla13/NepaliGov-RAG-Bench.git
   cd NepaliGov-RAG-Bench
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the application:**
   ```bash
   python -m src.initialize
   ```

## Configuration

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Database Configuration
DATABASE_URL=sqlite:///./data/rag_bench.db

# Model Configuration
MODEL_PATH=./models
CACHE_DIR=./cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Verification

Test your installation:

```bash
# Check API health
curl http://localhost:8000/health

# Run tests
pytest tests/

# Check documentation
mkdocs serve
```

## Troubleshooting

Common issues and solutions:

1. **Port already in use**: Change the port in `.env` or `docker-compose.yml`
2. **Permission errors**: Ensure proper file permissions
3. **Memory issues**: Increase available RAM or reduce model size
4. **Network issues**: Check firewall settings

For more help, see our [troubleshooting guide](troubleshooting.md).
