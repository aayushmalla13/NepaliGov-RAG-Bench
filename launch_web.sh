#!/bin/bash
# 🇳🇵 Nepal Government Q&A System Web Launcher

echo "🇳🇵 Nepal Government Q&A System"
echo "================================="

cd /home/fm-pc-lt-241/Documents/project/NepaliGov-RAG-Bench

# Activate virtual environment
source venv/bin/activate

# Check if corpus exists
if [ ! -f "data/real_corpus.parquet" ]; then
    echo "❌ Real corpus not found!"
    echo "🔧 Creating corpus from PDFs..."
    python create_real_corpus.py
fi

echo "✅ Starting system components..."

# Start API in background
echo "🚀 Starting API server on port 8000..."
python nepal_gov_api.py &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to initialize..."
sleep 8

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running!"
    
    # Start Streamlit UI
    echo "🌐 Starting web interface on port 8501..."
    echo "📱 Open your browser to: http://localhost:8501"
    echo ""
    echo "🎯 You can now ask questions like:"
    echo "• What are the fundamental rights in Nepal?"
    echo "• What health services are available?"
    echo "• How did Nepal respond to COVID-19?"
    echo ""
    
    streamlit run nepal_gov_ui.py --server.port 8501 --server.address 0.0.0.0
    
else
    echo "❌ API failed to start"
fi

# Cleanup
echo "🛑 Stopping API server..."
kill $API_PID 2>/dev/null
echo "✅ System stopped"


