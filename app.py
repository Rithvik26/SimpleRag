from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import tempfile
import logging
import json
from werkzeug.utils import secure_filename

# Import SimpleRAG core components
from simplerag import EmbeddingService, VectorDBService, DocumentProcessor, LLMService

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'simplerag_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
@app.template_filter('nl2br')
def nl2br_filter(text):
    if not text:
        return ""
    return text.replace('\n', '<br>')
# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleRAG-Web')

# Configuration path
CONFIG_PATH = os.path.join(tempfile.gettempdir(), 'simplerag_config.json')

# Default configuration
DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "claude_api_key": "",
    "qdrant_url": "https://3cbcacc0-1fe5-42a1-8be0-81515a21771b.us-west-2-0.aws.cloud.qdrant.io",
    "qdrant_api_key": "",
    "collection_name": "simple_rag_docs",
    "embedding_dimension": 768,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "preferred_llm": "claude"
}

# SimpleRAG services
embedding_service = None
vector_db_service = None
document_processor = None
llm_service = None

def load_config():
    """Load configuration from file or use default."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    else:
        # Write default config if not exists
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def initialize_services():
    """Initialize SimpleRAG services based on configuration."""
    global embedding_service, vector_db_service, document_processor, llm_service
    
    config = load_config()
    
    # Only initialize if API keys are available
    if config.get("gemini_api_key"):
        embedding_service = EmbeddingService(config)
        document_processor = DocumentProcessor(config)
        vector_db_service = VectorDBService(config)
        
        if (config.get("preferred_llm") == "claude" and config.get("claude_api_key")) or \
           (config.get("preferred_llm") == "xai" and config.get("xai_api_key")):
            llm_service = LLMService(config)

def index_document(file_path):
    """Process and index a document."""
    if not embedding_service or not vector_db_service or not document_processor:
        return False, "Services not initialized. Please configure API keys."
    
    try:
        # Extract text from document
        logger.info(f"Extracting text from {file_path}")
        text = document_processor.extract_text_from_file(file_path)
        
        # Create metadata
        filename = os.path.basename(file_path)
        metadata = {
            "filename": filename,
            "path": file_path,
            "created_at": os.path.getmtime(file_path),
            "file_type": os.path.splitext(filename)[1][1:].lower(),
            "session_id": session.get('session_id', 'default')  # Track which session uploaded this
        }
        
        # Chunk the document
        chunks = document_processor.chunk_text(text, metadata)
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return False, "No content could be extracted from document."
        
        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            logger.info(f"Generating embedding for chunk from {filename}")
            embedding = embedding_service.get_embedding(chunk["text"])
            embeddings.append(embedding)
        
        # Store in vector database
        logger.info(f"Storing {len(chunks)} chunks in vector database")
        vector_db_service.insert_documents(chunks, embeddings)
        
        logger.info(f"Successfully indexed document: {filename}")
        return True, f"Successfully indexed {filename} into {len(chunks)} chunks."
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to Qdrant: {str(e)}")
        return False, "Connection to vector database failed. Please check your internet connection and Qdrant API key."
    except Exception as e:
        logger.error(f"Error indexing document {file_path}: {str(e)}")
        return False, f"Error: {str(e)}"

def process_query(question):
    """Query the indexed documents and return an answer."""
    if not embedding_service or not vector_db_service:
        return "Services not initialized. Please configure API keys."
    
    try:
        # Generate embedding for the query
        logger.info(f"Generating embedding for query: {question}")
        query_embedding = embedding_service.get_embedding(question)
        
        # Retrieve similar contexts
        logger.info(f"Searching for relevant contexts")
        contexts = vector_db_service.search_similar(
            query_embedding,
            top_k=load_config()["top_k"]
        )
        
        if not contexts:
            logger.warning("No relevant contexts found")
            return "I couldn't find any relevant information to answer your question."
        
        # If LLM service is available, use it to generate an answer
        if llm_service:
            logger.info(f"Generating answer using LLM")
            answer = llm_service.generate_answer(question, contexts)
            return answer
        else:
            # If no LLM, just return the relevant chunks
            logger.info(f"No LLM configured, returning raw chunks")
            results = []
            for i, ctx in enumerate(contexts):
                results.append(f"--- Result {i+1} (Score: {ctx['score']:.2f}) ---\n")
                results.append(f"Source: {ctx['metadata'].get('filename', 'Unknown')}\n")
                results.append(f"{ctx['text']}\n\n")
            return "\n".join(results)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing your query: {str(e)}"

@app.route('/')
def home():
    """Render home page."""
    # Generate a session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    config = load_config()
    is_configured = bool(config.get("gemini_api_key"))
    
    return render_template('index.html', 
                          is_configured=is_configured, 
                          config=config)

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Configure API keys."""
    if request.method == 'POST':
        config = load_config()
        
        # Update config with form values
        config["gemini_api_key"] = request.form.get("gemini_api_key", "")
        config["claude_api_key"] = request.form.get("claude_api_key", "")
        config["qdrant_api_key"] = request.form.get("qdrant_api_key", "")
        config["qdrant_url"] = request.form.get("qdrant_url", DEFAULT_CONFIG["qdrant_url"])
        config["preferred_llm"] = request.form.get("preferred_llm", "claude")
        
        # Advanced settings
        try:
            config["chunk_size"] = int(request.form.get("chunk_size", 1000))
            config["chunk_overlap"] = int(request.form.get("chunk_overlap", 200))
            config["top_k"] = int(request.form.get("top_k", 5))
        except ValueError:
            flash("Invalid values for numeric fields. Using defaults.")
        
        save_config(config)
        initialize_services()
        
        flash("Configuration saved successfully!")
        return redirect(url_for('home'))
    
    return render_template('setup.html', config=load_config())

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document uploads."""
    if request.method == 'POST':
        # Check if any file was uploaded
        if 'document' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['document']
        
        # Check if file was selected
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if file is allowed
        allowed_extensions = {'pdf', 'txt', 'docx', 'html', 'htm'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flash('File type not supported. Please upload PDF, TXT, DOCX, or HTML files.')
            return redirect(request.url)
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Index the document
        success, message = index_document(file_path)
        
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
        
        return redirect(url_for('query'))
    
    return render_template('upload.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Handle document querying."""
    answer = None
    question = None
    
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            answer = process_query(question)
    
    return render_template('query.html', question=question, answer=answer)

@app.route('/advanced')
def advanced():
    """Advanced settings and information."""
    return render_template('advanced.html', config=load_config())

def create_templates():
    """Create template files for the Flask app."""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Base template
    with open('templates/base.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SimpleRAG{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">SimpleRAG</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/upload">Upload</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/query">Query</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/setup">Settings</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/advanced">Advanced</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category if category != 'message' else 'info' }}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <footer class="container mt-5 text-center text-muted">
        <hr>
        <p>SimpleRAG - A Retrieval-Augmented Generation System</p>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>''')
    
    # Index template
    with open('templates/index.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Home{% endblock %}

{% block content %}
<div class="jumbotron">
    <h1 class="display-4">SimpleRAG</h1>
    <p class="lead">A Retrieval-Augmented Generation System for Document Q&A</p>
    <hr class="my-4">
    
    {% if is_configured %}
        <p>Your system is configured and ready to use. Start by uploading documents or asking questions.</p>
        <div class="d-grid gap-2 d-sm-flex justify-content-sm-center mt-4">
            <a href="{{ url_for('upload') }}" class="btn btn-primary btn-lg px-4 me-sm-3">Upload Documents</a>
            <a href="{{ url_for('query') }}" class="btn btn-outline-secondary btn-lg px-4">Ask Questions</a>
        </div>
    {% else %}
        <p>Welcome to SimpleRAG! To get started, you need to configure your API keys.</p>
        <div class="d-grid gap-2 d-sm-flex justify-content-sm-center mt-4">
            <a href="{{ url_for('setup') }}" class="btn btn-primary btn-lg px-4">Configure System</a>
        </div>
    {% endif %}
</div>

<div class="row mt-5">
    <div class="col-md-4">
        <h3>📚 Document Indexing</h3>
        <p>Upload PDF, TXT, DOCX, and HTML files to create a searchable knowledge base.</p>
    </div>
    <div class="col-md-4">
        <h3>🔍 Semantic Search</h3>
        <p>Ask questions in natural language and get relevant information from your documents.</p>
    </div>
    <div class="col-md-4">
        <h3>🤖 LLM Integration</h3>
        <p>Get high-quality answers using Claude or other LLMs based on your documents.</p>
    </div>
</div>
{% endblock %}''')
    
    # Setup template
    with open('templates/setup.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Setup{% endblock %}

{% block content %}
<h2>Configure SimpleRAG</h2>
<p>Enter your API keys and settings to configure the system.</p>

<form method="post" class="mt-4">
    <div class="card mb-4">
        <div class="card-header">
            <h5>API Keys</h5>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <label for="gemini_api_key" class="form-label">Gemini API Key <span class="text-danger">*</span></label>
                <input type="password" class="form-control" id="gemini_api_key" name="gemini_api_key" 
                       value="{{ config.gemini_api_key }}" required>
                <div class="form-text">Required for embeddings. Get a key from <a href="https://makersuite.google.com/" target="_blank">Google AI Studio</a>.</div>
            </div>
            
            <div class="mb-3">
                <label for="claude_api_key" class="form-label">Claude API Key</label>
                <input type="password" class="form-control" id="claude_api_key" name="claude_api_key" 
                       value="{{ config.claude_api_key }}">
                <div class="form-text">Required if using Claude LLM. Get a key from <a href="https://anthropic.com/" target="_blank">Anthropic</a>.</div>
            </div>
            
            <div class="mb-3">
                <label for="qdrant_api_key" class="form-label">Qdrant API Key</label>
                <input type="password" class="form-control" id="qdrant_api_key" name="qdrant_api_key" 
                       value="{{ config.qdrant_api_key }}">
                <div class="form-text">Required for vector database. Get a key from <a href="https://cloud.qdrant.io/" target="_blank">Qdrant Cloud</a>.</div>
            </div>
                
            <div class="mb-3">
                <label for="qdrant_url" class="form-label">Qdrant URL</label>
                <input type="text" class="form-control" id="qdrant_url" name="qdrant_url" 
                    value="{{ config.qdrant_url }}">
                <div class="form-text">URL of your Qdrant instance (e.g., https://your-instance.qdrant.io)</div>
            </div>
        </div>
    </div>
    
    <div class="card mb-4">
        <div class="card-header">
            <h5>LLM Settings</h5>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <label class="form-label">Preferred LLM</label>
                <div class="form-check">
                    <input class="form-check-input" type="radio" name="preferred_llm" id="claude" value="claude" 
                           {% if config.preferred_llm == 'claude' %}checked{% endif %}>
                    <label class="form-check-label" for="claude">Claude</label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="radio" name="preferred_llm" id="raw" value="raw" 
                           {% if config.preferred_llm == 'raw' %}checked{% endif %}>
                    <label class="form-check-label" for="raw">Raw Results (No LLM)</label>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card mb-4">
        <div class="card-header">
            <h5>Advanced Settings</h5>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <label for="chunk_size" class="form-label">Chunk Size</label>
                <input type="number" class="form-control" id="chunk_size" name="chunk_size" 
                       value="{{ config.chunk_size }}" min="100" max="5000">
                <div class="form-text">Size of text chunks in characters (500-5000 recommended).</div>
            </div>
            
            <div class="mb-3">
                <label for="chunk_overlap" class="form-label">Chunk Overlap</label>
                <input type="number" class="form-control" id="chunk_overlap" name="chunk_overlap" 
                       value="{{ config.chunk_overlap }}" min="0" max="1000">
                <div class="form-text">Overlap between adjacent chunks (50-500 recommended).</div>
            </div>
            
            <div class="mb-3">
                <label for="top_k" class="form-label">Results Count (Top K)</label>
                <input type="number" class="form-control" id="top_k" name="top_k" 
                       value="{{ config.top_k }}" min="1" max="20">
                <div class="form-text">Number of results to retrieve for each query (1-20).</div>
            </div>
        </div>
    </div>
    
    <div class="d-grid gap-2">
        <button type="submit" class="btn btn-primary">Save Configuration</button>
    </div>
</form>
{% endblock %}''')
    
    # Upload template
    with open('templates/upload.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Upload Documents{% endblock %}

{% block content %}
<h2>Upload Documents</h2>
<p>Upload documents to index them into the vector database.</p>

<div class="card mt-4">
    <div class="card-body">
        <form method="post" enctype="multipart/form-data" class="mb-3">
            <div class="mb-3">
                <label for="document" class="form-label">Select Document</label>
                <input type="file" class="form-control" id="document" name="document">
                <div class="form-text">Supported formats: PDF, TXT, DOCX, HTML</div>
            </div>
            <button type="submit" class="btn btn-primary">Upload & Index</button>
        </form>
    </div>
</div>

<div class="mt-4">
    <h4>Indexing Process</h4>
    <ol>
        <li>Your document is uploaded and text is extracted</li>
        <li>Text is split into chunks with some overlap</li>
        <li>Each chunk is converted to a vector embedding</li>
        <li>Embeddings are stored in the vector database</li>
    </ol>
    <p class="alert alert-info">
        <strong>Note:</strong> Document processing can take some time depending on the file size and complexity.
    </p>
</div>
{% endblock %}''')
    
    # Query template
    with open('templates/query.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Query Documents{% endblock %}

{% block content %}
<h2>Ask Questions</h2>
<p>Ask questions about your indexed documents.</p>

<div class="card mt-4">
    <div class="card-body">
        <form method="post">
            <div class="mb-3">
                <label for="question" class="form-label">Your Question</label>
                <input type="text" class="form-control" id="question" name="question" 
                       value="{{ question|default('') }}" required placeholder="What would you like to know about your documents?">
            </div>
            <button type="submit" class="btn btn-primary">Ask</button>
        </form>
    </div>
</div>

{% if answer %}
<div class="card mt-4">
    <div class="card-header">
        <h5>Answer</h5>
    </div>
    <div class="card-body">
        <div class="answer-text">
            {{ answer | nl2br }}
        </div>
    </div>
</div>
{% endif %}
{% endblock %}''')
    
    # Advanced template
    with open('templates/advanced.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Advanced{% endblock %}

{% block content %}
<h2>Advanced Information</h2>

<div class="card mb-4">
    <div class="card-header">
        <h5>Current Configuration</h5>
    </div>
    <div class="card-body">
        <table class="table">
            <tr>
                <th>Setting</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Embedding Model</td>
                <td>Gemini Embedding API</td>
            </tr>
            <tr>
                <td>Vector Database</td>
                <td>Qdrant Cloud</td>
            </tr>
            <tr>
                <td>Preferred LLM</td>
                <td>{{ config.preferred_llm }}</td>
            </tr>
            <tr>
                <td>Chunk Size</td>
                <td>{{ config.chunk_size }}</td>
            </tr>
            <tr>
                <td>Chunk Overlap</td>
                <td>{{ config.chunk_overlap }}</td>
            </tr>
            <tr>
                <td>Results Count (Top K)</td>
                <td>{{ config.top_k }}</td>
            </tr>
        </table>
    </div>
</div>

<div class="card mb-4">
    <div class="card-header">
        <h5>How It Works</h5>
    </div>
    <div class="card-body">
        <h6>1. Document Processing</h6>
        <p>Documents are parsed, extracted, and split into chunks of text with some overlap.</p>

        <h6>2. Vector Embeddings</h6>
        <p>Each chunk is converted into a vector embedding using the Gemini API.</p>

        <h6>3. Vector Storage</h6>
        <p>Embeddings are stored in Qdrant vector database for efficient similarity search.</p>

        <h6>4. Query Processing</h6>
        <p>When you ask a question, it's converted to an embedding and used to find the most similar chunks.</p>

        <h6>5. LLM Integration</h6>
        <p>The most relevant chunks are sent to Claude LLM along with your question to generate a comprehensive answer.</p>
    </div>
</div>

<div class="card mb-4">
    <div class="card-header">
        <h5>About SimpleRAG</h5>
    </div>
    <div class="card-body">
        <p>SimpleRAG is a Retrieval-Augmented Generation system for document Q&A. It allows you to create a searchable knowledge base from your documents and ask questions in natural language.</p>
        
        <p>This web application is a simple interface for the SimpleRAG system. For more advanced usage, you can use the command-line interface or integrate it into your own applications.</p>
    </div>
</div>
{% endblock %}''')
    
    # CSS styles
    with open('static/style.css', 'w') as f:
        f.write('''/* Custom styles for SimpleRAG */

.answer-text {
    white-space: pre-line;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 5px;
}

.jumbotron {
    padding: 2rem 1rem;
    background-color: #e9ecef;
    border-radius: 0.3rem;
}

/* Add a nl2br filter to the Jinja environment */
''')

# Initialize services on startup
initialize_services()

if __name__ == '__main__':
    # Create template files if they don't exist
    create_templates()
    
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))