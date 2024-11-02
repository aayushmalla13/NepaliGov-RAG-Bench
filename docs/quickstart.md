# Quick Start Guide

Get up and running with NepaliGov-RAG-Bench in minutes.

## Prerequisites

- Python 3.8+
- Docker (optional)
- Git

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/aayushmalla13/NepaliGov-RAG-Bench.git
cd NepaliGov-RAG-Bench

# Start the services
docker-compose up -d

# Access the application
# API: http://localhost:8000
# Documentation: http://localhost:8080
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/aayushmalla13/NepaliGov-RAG-Bench.git
cd NepaliGov-RAG-Bench

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn src.api.app:app --reload
```

## Usage

Once the application is running, you can:

1. Access the API at `http://localhost:8000`
2. View the documentation at `http://localhost:8080`
3. Use the web interface for document processing

## Next Steps

- Read the [full documentation](index.md)
- Explore the [API endpoints](api/)
- Learn about [development workflows](development/)
