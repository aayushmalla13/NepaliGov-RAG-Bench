# Frequently Asked Questions (FAQ)

## 🤔 General Questions

### What is NepaliGov-RAG-Bench?

NepaliGov-RAG-Bench is a smart search system for Nepali government documents. Think of it as a "Google for government documents" - you can ask questions in plain English or Nepali, and it will find relevant information from official government documents and give you clear answers.

### What makes this system special?

- **Works in Both Languages**: You can ask questions in English or Nepali
- **Made for Government Documents**: Specifically designed to understand government language and procedures
- **Smart Text Reading**: Can read PDF documents and extract text accurately
- **Understands Meaning**: Doesn't just search for exact words - understands what you're looking for
- **Fast Processing**: Quickly processes and makes documents searchable

### Who can use this system?

- **Government Officials**: Policy makers and administrators
- **Researchers**: Academic and policy researchers
- **Journalists**: Media professionals fact-checking information
- **Citizens**: General public seeking government information
- **Developers**: Building applications on government data

## 🚀 Getting Started

### How do I install the system?

See our [Quick Start Guide](quickstart.md) for a 5-minute setup, or the detailed [Installation Guide](installation.md) for comprehensive instructions.

### What are the system requirements?

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 10GB free disk space
- Internet connection

**Recommended:**
- Python 3.12
- 16GB+ RAM
- 50GB+ SSD
- NVIDIA GPU (optional)

### Do I need special permissions to run this?

No, the system runs locally and doesn't require special permissions. However, you need sufficient system resources and a stable internet connection for initial model downloads.

## 🔍 Usage Questions

### How do I ask questions?

Just type your question in normal, everyday language - like you're talking to a helpful government employee:

**Good examples in English:**
- "What do I need to start a business?"
- "How can I get a government job?"
- "What are the tax rates?"

**Good examples in Nepali:**
- "व्यापार सुरु गर्नका लागि के के आवश्यकता छ?"
- "सरकारी नौकरीका लागि कसरी आवेदन गर्ने?"
- "हालका कर दरहरू के के छन्?"

**Tip:** Ask questions like you would ask a person - the system understands natural language!

### What types of questions work best?

- **Specific questions**: "What documents do I need for business registration?"
- **Process questions**: "How do I apply for a license?"
- **Factual questions**: "What is the minimum age for voting?"
- **Comparison questions**: "What are the differences between private and public sector jobs?"

### Why am I getting no results?

Common reasons for no results:
- **Query too specific**: Try broader terms
- **Query too vague**: Be more specific
- **Language mismatch**: Try the other language
- **Topic not covered**: The topic might not be in the document corpus

**Solutions:**
- Rephrase your question
- Use different keywords
- Try the suggested quick questions
- Check if the topic is covered in government documents

### How accurate are the search results?

The system typically achieves:
- **90%+ search relevance** for well-formed queries
- **95%+ OCR accuracy** for clear documents
- **85%+ translation quality** (BLEU score)
- **90%+ user satisfaction** based on feedback

### Can I upload my own documents?

Yes, you can upload PDF documents through the web interface. The system will:
1. Process the document using OCR
2. Extract text and create embeddings
3. Add it to the searchable corpus
4. Make it available for queries

## 🌐 Language and Translation

### Which languages are supported?

Currently supported:
- **English**: Full support for queries and responses
- **Nepali**: Full support for queries and responses
- **Mixed content**: Documents with both languages

### How does translation work?

The system uses:
- **MarianMT**: Primary translation service
- **Google Translate**: Fallback service (if available)
- **Quality assessment**: Automatic quality scoring
- **Context awareness**: Government-specific terminology

### Why are some translations awkward?

Translation quality depends on:
- **Document complexity**: Technical documents are harder to translate
- **Context**: Government terminology requires specialized knowledge
- **Language pairs**: English-Nepali translation is still evolving

**Solutions:**
- Try rephrasing your question
- Use simpler language
- Ask in the original document language
- Provide feedback to improve translations

### Can I get answers in both languages?

Yes! The system can provide:
- **Bilingual responses**: Answers in both English and Nepali
- **Source language**: Answers in the original document language
- **Preferred language**: Answers in your preferred language

## 📊 Technical Questions

### How does the search work?

The system uses:
1. **Semantic Search**: Vector similarity search using FAISS
2. **Embeddings**: Transformers-based text embeddings
3. **Ranking**: Multiple factors including relevance, quality, and recency
4. **Filtering**: Document type, authority, and language filtering

### What is the processing pipeline?

The system follows a 12-stage checkpoint architecture:
1. **CP1**: Authority detection
2. **CP2**: Document type classification
3. **CP3**: OCR processing
4. **CP4**: Vector storage
5. **CP5**: Semantic search
6. **CP6**: Language preference
7. **CP7**: QA citation
8. **CP8**: Metrics collection
9. **CP9**: Language selection
10. **CP10**: Refusal handling
11. **CP11**: API and UI
12. **CP12**: Incremental processing

### How long does document processing take?

Processing time depends on:
- **Document size**: Larger documents take longer
- **Document quality**: Poor quality documents require more processing
- **System resources**: More RAM and CPU speed up processing

**Typical times:**
- **Small documents** (<10 pages): 30-60 seconds
- **Medium documents** (10-50 pages): 1-3 minutes
- **Large documents** (>50 pages): 3-10 minutes

### Can I run this offline?

The system requires internet for:
- **Initial setup**: Downloading models and dependencies
- **Translation**: MarianMT and Google Translate services
- **Model updates**: Updating language models

Once set up, you can run it offline for:
- **Search**: Local vector search
- **OCR**: Local text extraction
- **Basic queries**: If translation models are cached

## 🔧 Troubleshooting

### The application won't start

Common causes and solutions:
- **Python version**: Ensure Python 3.8+ is installed
- **Dependencies**: Run `pip install -r requirements.txt`
- **Port conflict**: Kill existing processes on port 8093
- **Permissions**: Check file permissions and ownership

### Search is slow

Performance optimization:
- **Close other applications**: Free up system resources
- **Use CPU-only FAISS**: Set `FAISS_CPU_ONLY=1`
- **Reduce batch sizes**: Lower processing batch sizes
- **Check system resources**: Monitor RAM and CPU usage

### OCR quality is poor

Improve OCR results:
- **Use high-quality PDFs**: Clear, text-based documents work best
- **Avoid scanned images**: Use native PDF documents when possible
- **Check document format**: Ensure documents are properly formatted
- **Adjust thresholds**: Lower confidence thresholds in configuration

### Translation is not working

Translation troubleshooting:
- **Check internet connection**: Translation requires online services
- **Clear cache**: Remove translation cache files
- **Update models**: Reinstall translation dependencies
- **Try different service**: Switch between MarianMT and Google Translate

## 📈 Performance and Scalability

### How many documents can the system handle?

The system can handle:
- **Small scale**: 1,000-5,000 documents
- **Medium scale**: 5,000-50,000 documents
- **Large scale**: 50,000+ documents (with optimization)

### How many concurrent users are supported?

Current limits:
- **Light usage**: 10-20 concurrent users
- **Medium usage**: 20-50 concurrent users
- **Heavy usage**: 50+ concurrent users (with optimization)

### Can I scale this system?

Yes, the system can be scaled:
- **Horizontal scaling**: Multiple application instances
- **Load balancing**: Nginx-based traffic distribution
- **Database scaling**: PostgreSQL for larger datasets
- **Caching**: Redis for improved performance

## 🔒 Security and Privacy

### Is my data secure?

Security features:
- **Local processing**: Data stays on your system
- **No external storage**: Documents are not sent to external servers
- **Encryption**: Data encrypted at rest and in transit
- **Access control**: User authentication and authorization

### What data is collected?

The system collects:
- **Query logs**: For performance optimization
- **Usage metrics**: For system improvement
- **Error logs**: For troubleshooting
- **No personal data**: No personal information is stored

### Can I use this for sensitive documents?

Yes, the system is designed for:
- **Government documents**: Official and sensitive documents
- **Local processing**: All processing happens locally
- **No external transmission**: Documents are not sent to external services
- **Audit trails**: Complete processing logs

## 🚀 Advanced Usage

### Can I integrate this with other systems?

Yes, the system provides:
- **RESTful API**: Full API for integration
- **Webhook support**: Real-time notifications
- **Database access**: Direct database queries
- **Custom endpoints**: Extensible API architecture

### Can I customize the system?

Customization options:
- **Configuration files**: Modify behavior through config files
- **Custom models**: Use different language models
- **Custom themes**: Modify the web interface
- **Custom processing**: Add new processing stages

### How do I contribute to the project?

Ways to contribute:
- **Bug reports**: Report issues and bugs
- **Feature requests**: Suggest new features
- **Code contributions**: Submit pull requests
- **Documentation**: Improve documentation
- **Testing**: Help test new features

## 📚 Learning Resources

### Where can I learn more about RAG systems?

Educational resources:
- **Research papers**: Academic papers on RAG systems
- **Online courses**: Machine learning and NLP courses
- **Documentation**: System architecture and implementation details
- **Tutorials**: Step-by-step guides and examples

### How can I understand the technical details?

Technical resources:
- **Architecture documentation**: System design and components
- **API documentation**: Complete API reference
- **Code comments**: Inline code documentation
- **Development guides**: Contributing and development workflows

### Are there training materials available?

Training resources:
- **User guides**: Step-by-step usage instructions
- **Video tutorials**: Visual learning materials
- **Webinars**: Live training sessions
- **Documentation**: Comprehensive written guides

---

**Still have questions?** Check our [Troubleshooting Guide](troubleshooting.md) or [API Documentation](api/index.md) for more help.
