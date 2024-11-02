.PHONY: setup cp1 cp2 cp3 cp4 cp5 cp6 cp7 cp8 cp9 cp10 cp11 cp12 clean test lint format

# CP0 - Bootstrap & Setup
setup:
	@echo "Setting up NepaliGov-RAG-Bench environment..."
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .
	@echo "✅ Setup complete. Run './venv/bin/python -c \"import sys; sys.path.insert(0, 'venv/lib/python3.12/site-packages'); import fitz,faiss,transformers; print('All imports successful')\"' to verify."

# Checkpoint placeholders
cp1:
	@echo "CP1 - PDF Ingestion & Manifest"
	@echo "TODO: Implement PDF ingestion pipeline"

cp2:
	@echo "CP2 - OCR & Layout Analysis"
	@echo "TODO: Implement OCR processing"

cp3:
	@echo "CP3 - Chunking & Corpus Export"
	@echo "TODO: Implement text chunking and corpus export"

cp4:
	@echo "CP4 - Wikipedia Distractor Ingestion"
	@echo "TODO: Implement Wikipedia distractor corpus"

cp5:
	@echo "CP5 - Embedding & FAISS Index"
	@echo "TODO: Implement vector indexing"

cp6:
	@echo "CP6 - Retrieval & Reranking"
	@echo "TODO: Implement retrieval and reranking"

cp7:
	@echo "CP7 - Citation-Constrained Answering"
	@echo "TODO: Implement citation-aware answering"

cp8:
	@echo "CP8 - Q&A Dataset & Evaluation Metrics"
	@echo "TODO: Implement evaluation framework"

cp9:
	@echo "CP9 - FastAPI Service"
	@echo "TODO: Implement FastAPI endpoints"

cp10:
	@echo "CP10 - Streamlit UI"
	@echo "TODO: Implement Streamlit interface"

cp11:
	@echo "CP11 - End-to-End Integration"
	@echo "TODO: Implement full integration testing"

cp12:
	@echo "CP12 - Production & Deployment"
	@echo "TODO: Implement deployment configuration"

# Development utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/

test:
	pytest tests/ -v

lint:
	flake8 src/ api/ ui/ ops/
	mypy src/ api/ ui/ ops/

format:
	black src/ api/ ui/ ops/ tests/
	isort src/ api/ ui/ ops/ tests/

# Verify PDF corpus
check-pdfs:
	@echo "Checking PDF corpus in data/raw_pdfs/..."
	@ls -la data/raw_pdfs/*.pdf | wc -l | xargs echo "PDF count:"
	@ls data/raw_pdfs/
