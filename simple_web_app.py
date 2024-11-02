#!/usr/bin/env python3
"""
🇳🇵 Simple Web App for Nepal Government Q&A

A simple web interface that works without complex dependencies.
Uses the dynamic search system.
"""

import pandas as pd
import re
import time
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Import our dynamic search functions
def load_corpus():
    """Load corpus dynamically."""
    corpus_path = Path("data/real_corpus.parquet")
    if not corpus_path.exists():
        return None
    return pd.read_parquet(corpus_path)

def load_title_mapping():
    """Load proper PDF titles from JSON file."""
    import json
    from pathlib import Path
    
    title_file = Path("data/pdf_titles.json")
    if title_file.exists():
        with open(title_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def generate_dynamic_title(doc_id: str, text_sample: str = "", title_mapping=None) -> str:
    """Generate document title using proper titles from mapping."""
    if title_mapping and doc_id in title_mapping:
        return title_mapping[doc_id]
    
    # Fallback to dynamic generation
    title = doc_id.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[0-9]+\s*', '', title)
    return ' '.join(word.capitalize() for word in title.split())

def calculate_semantic_similarity(query: str, text: str) -> float:
    """Calculate semantic similarity."""
    query_lower = query.lower()
    text_lower = text.lower()
    
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    
    score = 0.0
    
    # Exact phrase matching
    if query_lower in text_lower:
        score += 25.0
    
    # Multi-word phrases
    query_tokens = query_lower.split()
    for i in range(len(query_tokens)):
        for j in range(i + 2, min(i + 6, len(query_tokens) + 1)):
            phrase = ' '.join(query_tokens[i:j])
            if phrase in text_lower:
                score += 5.0 + ((j - i) * 2.0)
    
    # Word overlap
    if query_words and text_words:
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        jaccard = len(intersection) / len(union) if union else 0
        coverage = len(intersection) / len(query_words)
        score += jaccard * 15.0 + coverage * 10.0
    
    # Structural content
    if re.search(r'\(\d+\)', text):
        score += 4.0
    if re.search(r'Article \d+|Section \d+', text, re.IGNORECASE):
        score += 3.0
    
    return score

def dynamic_search(df, query: str, max_results: int = 8, title_mapping=None):
    """Dynamic search without hardcoding."""
    results = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        doc_id = str(row['doc_id'])
        
        similarity_score = calculate_semantic_similarity(query, text)
        
        if similarity_score > 3.0:
            doc_title = generate_dynamic_title(doc_id, text[:200], title_mapping)
            
            result = {
                'doc_id': doc_id,
                'doc_title': doc_title,
                'text': text,
                'similarity_score': round(similarity_score, 2),
                'page_num': int(row.get('page_num', 1)),
                'language': str(row.get('language', 'en'))
            }
            results.append(result)
    
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:max_results]

def clean_text(text: str) -> str:
    """Clean and format text for better display."""
    if not text:
        return text
    
    # Remove control characters but preserve Nepali text
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove table markers
    text = re.sub(r'\[TABLE[^\]]*\]', '', text)
    
    return text.strip()

def extract_relevant_portion(text: str, query: str) -> str:
    """Extract relevant text portion with better formatting."""
    # Clean the text first
    text = clean_text(text)
    
    if len(text) <= 300:
        return text
    
    query_words = set(query.lower().split())
    sentences = re.split(r'[.!?।]+', text)  # Added Nepali sentence delimiter ।
    
    sentence_scores = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        score = overlap / len(query_words) if query_words else 0
        sentence_scores.append((sentence, score))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_sentences = [s[0] for s in sentence_scores[:3] if s[1] > 0]
    
    if relevant_sentences:
        result = '. '.join(relevant_sentences) + '.'
        return result[:400] + "..." if len(result) > 400 else result
    else:
        return text[:297] + "..."

def generate_answer(results, query: str):
    """Generate answer from results with proper titles."""
    if not results:
        return {
            'query': query,
            'answer': f"I couldn't find specific information about '{query}' in the documents.",
            'confidence': 0.0,
            'sources': []
        }
    
    top_results = results[:3]
    answer_parts = []
    sources = []
    
    for i, result in enumerate(top_results):
        text = result['text']
        doc_title = result['doc_title']  # This now has proper titles
        
        relevant_text = extract_relevant_portion(text, query)
        
        if i == 0:
            answer_parts.append(f"**According to {doc_title}:**\n{relevant_text}")
        else:
            answer_parts.append(f"**{doc_title} also states:**\n{relevant_text}")
        
        sources.append({
            'doc_title': doc_title,
            'page_num': result['page_num'],
            'similarity_score': result['similarity_score']
        })
    
    answer = "\n\n".join(answer_parts)
    
    if sources:
        answer += "\n\n**📚 Sources:**"
        for i, source in enumerate(sources, 1):
            answer += f"\n{i}. {source['doc_title']} (Page {source['page_num']}, Relevance: {source['similarity_score']:.1f})"
    
    avg_similarity = sum(r['similarity_score'] for r in top_results) / len(top_results)
    confidence = min(avg_similarity / 25.0, 1.0)
    
    return {
        'query': query,
        'answer': answer,
        'confidence': round(confidence, 2),
        'sources': sources
    }

# Load corpus and title mapping globally
print("🔍 Loading corpus...")
CORPUS_DF = load_corpus()
TITLE_MAPPING = load_title_mapping()

if CORPUS_DF is not None:
    print(f"✅ Loaded {len(CORPUS_DF)} chunks from {len(CORPUS_DF['doc_id'].unique())} documents")
    print(f"✅ Loaded {len(TITLE_MAPPING)} proper document titles")
else:
    print("❌ Failed to load corpus")

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html lang="ne">
<head>
    <title>🇳🇵 Nepal Government Q&A</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Inter', 'Noto Sans Devanagari', Arial, sans-serif; 
            max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; 
            line-height: 1.6;
        }
        .header { background: linear-gradient(45deg, #DC143C, #1E3A8A); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .question-box { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .answer-box { background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        input[type="text"] { 
            width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 5px; 
            font-size: 16px; font-family: 'Inter', 'Noto Sans Devanagari', Arial, sans-serif;
            direction: ltr; /* Support both LTR and RTL */
        }
        button { background: #DC143C; color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin: 10px 5px; }
        button:hover { background: #B91C3C; }
        .loading { text-align: center; padding: 20px; }
        .confidence { display: inline-block; padding: 5px 10px; border-radius: 15px; color: white; font-size: 12px; }
        .conf-high { background: #28a745; }
        .conf-medium { background: #ffc107; }
        .conf-low { background: #dc3545; }
        pre { 
            white-space: pre-wrap; line-height: 1.6; 
            font-family: 'Inter', 'Noto Sans Devanagari', Arial, sans-serif;
            background: #f8f9fa; padding: 15px; border-radius: 5px;
        }
        .nepali-text { font-family: 'Noto Sans Devanagari', Arial, sans-serif; }
        .sources { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .source-item { margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🇳🇵 Nepal Government Q&A System</h1>
        <p>Ask questions about Nepal's Constitution, Health Policies, Government Services, and more!</p>
        <p><em>Dynamic system - works with any government document</em></p>
    </div>
    
    <div class="question-box">
        <h3>❓ Ask Your Question</h3>
        <input type="text" id="question" placeholder="Ask in English or Nepali: What are the policies relating to political and governance system?" />
        <button onclick="askQuestion()">🔍 Get Answer</button>
        
        <div style="margin-top: 15px;">
            <strong>Quick Questions (English):</strong><br>
            <button onclick="setQuestion('What are the policies relating to political and governance system of State?')">🏛️ Governance Policies</button>
            <button onclick="setQuestion('What are the fundamental rights in Nepal?')">📜 Fundamental Rights</button>
            <button onclick="setQuestion('What health services are available in Nepal?')">🏥 Health Services</button>
            <br><br>
            <strong>नेपालीमा प्रश्न (Nepali Questions):</strong><br>
            <button onclick="setQuestion('नेपालमा मौलिक अधिकारहरू के के छन्?')" class="nepali-text">📜 मौलिक अधिकार</button>
            <button onclick="setQuestion('स्वास्थ्य सेवाहरू के छन्?')" class="nepali-text">🏥 स्वास्थ्य सेवा</button>
            <button onclick="setQuestion('शासन व्यवस्था कस्तो छ?')" class="nepali-text">🏛️ शासन व्यवस्था</button>
        </div>
    </div>
    
    <div id="result"></div>
    
    <script>
        function setQuestion(q) {
            document.getElementById('question').value = q;
        }
        
        function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            document.getElementById('result').innerHTML = '<div class="loading">🔍 Searching government documents...</div>';
            
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                document.getElementById('result').innerHTML = '<div class="answer-box">❌ Error: ' + error + '</div>';
            });
        }
        
        function displayResult(data) {
            let confClass = data.confidence >= 0.7 ? 'conf-high' : data.confidence >= 0.4 ? 'conf-medium' : 'conf-low';
            let confText = data.confidence >= 0.7 ? 'High' : data.confidence >= 0.4 ? 'Medium' : 'Low';
            
            let sourcesHtml = '';
            if (data.sources && data.sources.length > 0) {
                sourcesHtml = `
                    <div class="sources">
                        <h4>📚 Sources:</h4>
                        ${data.sources.map((source, index) => `
                            <div class="source-item">
                                <strong>${index + 1}. ${source.doc_title || 'Unknown Document'}</strong><br>
                                Page ${source.page_num || 'N/A'}, Relevance: ${(source.similarity_score || 0).toFixed(1)}
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            let html = `
                <div class="answer-box">
                    <h3>✅ Answer <span class="confidence ${confClass}">${confText} Confidence (${(data.confidence*100).toFixed(0)}%)</span></h3>
                    <pre class="nepali-text">${data.answer}</pre>
                    ${sourcesHtml}
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }
        
        // Allow Enter key to submit
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                question = data.get('question', '').strip()
                
                if not question:
                    raise ValueError("Question cannot be empty")
                
                if CORPUS_DF is None:
                    raise ValueError("Corpus not loaded")
                
                # Perform search
                start_time = time.time()
                results = dynamic_search(CORPUS_DF, question, title_mapping=TITLE_MAPPING)
                answer_data = generate_answer(results, question)
                processing_time = (time.time() - start_time) * 1000
                
                answer_data['processing_time_ms'] = round(processing_time, 1)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(answer_data, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                error_response = {
                    'query': data.get('question', '') if 'data' in locals() else '',
                    'answer': f'Error: {str(e)}',
                    'confidence': 0.0,
                    'sources': []
                }
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def start_server(port=8080):
    """Start the simple web server."""
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"🚀 Nepal Government Q&A Server starting...")
    print(f"📍 Open your browser to: http://localhost:{port}")
    print(f"🎯 Ready to answer questions about any government document!")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        server.shutdown()

if __name__ == "__main__":
    if CORPUS_DF is not None:
        print(f"🔍 Sample title mapping check:")
        test_id = '9789240061262-eng'
        print(f"  {test_id} -> {TITLE_MAPPING.get(test_id, 'NOT FOUND')}")
        start_server(8086)  # Use different port to force fresh start
    else:
        print("❌ Cannot start server - corpus not loaded")
        print("🔧 Please ensure data/real_corpus.parquet exists")
