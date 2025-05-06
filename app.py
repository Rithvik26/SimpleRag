from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import os
import tempfile
import logging
import json
import threading
from werkzeug.utils import secure_filename
import markdown
import re
# Import SimpleRAG core components
from simplerag import SimpleRAG
from extensions import ProgressTracker

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'simplerag_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.template_filter('nl2br')
def nl2br_filter(text):
    if not text:
        return ""
    # First convert <br> tags to newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Convert markdown to HTML
    return markdown.markdown(text)

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
    "preferred_llm": "claude",
    "rate_limit": 60,  # Max API calls per minute
    "enable_cache": True,  # Enable embedding cache
    "cache_dir": None  # Default cache directory
}

# SimpleRAG instance
simplerag_instance = None

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
    global simplerag_instance
    
    # Initialize SimpleRAG
    simplerag_instance = SimpleRAG()

def index_document_background(file_path, session_id=None):
    """Index document in the background."""
    global simplerag_instance
    
    try:
        if simplerag_instance:
            simplerag_instance.index_document(file_path, session_id)
    except Exception as e:
        logger.error(f"Background indexing error: {str(e)}")
        # Update progress tracker with error if exists
        if session_id:
            tracker = ProgressTracker.get_tracker(session_id, "index_document")
            if tracker:
                tracker.update(100, 100, status="error", message=f"Error: {str(e)}")

def process_query(question, session_id=None):
    """Query the indexed documents and return an answer."""
    global simplerag_instance
    
    if not simplerag_instance:
        return "Services not initialized. Please configure API keys."
    
    try:
        # Create a progress tracker for the query
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                   message="Processing your question")
        
        # Generate embedding for the query
        logger.info(f"Generating embedding for query: {question}")
        query_embedding = simplerag_instance.embedding_service.get_embedding(question)
        
        # Update progress
        if session_id:
            progress_tracker.update(25, 100, status="searching", 
                                   message="Searching for relevant contexts")
        
        # Retrieve similar contexts
        logger.info(f"Searching for relevant contexts")
        contexts = simplerag_instance.vector_db_service.search_similar(
            query_embedding,
            top_k=simplerag_instance.config["top_k"]
        )
        
        if not contexts:
            logger.warning("No relevant contexts found")
            if session_id:
                progress_tracker.update(100, 100, status="complete", 
                                      message="No relevant information found")
            return "I couldn't find any relevant information to answer your question."
        
        # Update progress
        if session_id:
            progress_tracker.update(50, 100, status="generating", 
                                   message="Generating answer based on relevant documents")
        
        # Generate answer
        if simplerag_instance.llm_service and simplerag_instance.config["preferred_llm"] != "raw":
            logger.info(f"Generating answer using LLM")
            # Get the answer from LLM
            answer = simplerag_instance.llm_service.generate_answer(question, contexts)
            
            # Process answer to ensure proper formatting
            # Replace HTML <br> tags with markdown line breaks
            answer = re.sub(r'<br\s*/?>', '\n\n', answer)
            
            # Clean other HTML tags but preserve content
            answer = re.sub(r'<[^>]*>', '', answer)
            
        else:
            # If no LLM, just return the relevant chunks
            logger.info(f"No LLM configured, returning raw chunks")
            results = []
            for i, ctx in enumerate(contexts):
                results.append(f"### Result {i+1} (Score: {ctx['score']:.2f})")
                results.append(f"**Source**: {ctx['metadata'].get('filename', 'Unknown')}")
                results.append("")  # Empty line
                results.append(ctx['text'])
                results.append("")  # Empty line
            answer = "\n".join(results)
        
        # Update progress
        if session_id:
            progress_tracker.update(100, 100, status="complete", 
                                   message="Answer generation complete")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Update progress with error
        if session_id:
            progress_tracker = ProgressTracker.get_tracker(session_id, "query")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
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
            config["rate_limit"] = int(request.form.get("rate_limit", 60))
            config["enable_cache"] = bool(request.form.get("enable_cache", True))
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
        
        # Start indexing in a background thread
        session_id = session.get('session_id')
        
        # Start a progress tracker
        progress_tracker = ProgressTracker(session_id, "index_document")
        progress_tracker.update(0, 100, status="starting", 
                               message=f"Starting to index {filename}")
        
        # Start background thread for indexing
        indexing_thread = threading.Thread(
            target=index_document_background,
            args=(file_path, session_id)
        )
        indexing_thread.daemon = True
        indexing_thread.start()
        
        flash(f"Document upload successful. Indexing {filename} in progress.", 'success')
        
        return redirect(url_for('upload_progress'))
    
    return render_template('upload.html')

@app.route('/upload/progress')
def upload_progress():
    """Show document upload progress."""
    return render_template('upload_progress.html')

@app.route('/api/progress/<operation_type>')
def get_progress(operation_type):
    """API endpoint to get progress information."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    tracker = ProgressTracker.get_tracker(session_id, operation_type)
    if not tracker:
        return jsonify({"error": "No progress tracker found"}), 404
    
    return jsonify(tracker.get_info())

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Handle document querying."""
    answer = None
    question = None
    in_progress = False
    
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            session_id = session.get('session_id')
            
            # Create a progress tracker for the query
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                   message="Processing your question")
            
            # Start query in background thread if it's a long query
            if len(question) > 100 or request.form.get('async', 'false') == 'true':
                in_progress = True
                
                # Process query in background
                def process_query_background():
                    try:
                        result = process_query(question, session_id)
                        # Store result in session
                        session['last_query_result'] = result
                        # Update progress tracker
                        progress_tracker = ProgressTracker.get_tracker(session_id, "query")
                        if progress_tracker:
                            progress_tracker.update(100, 100, status="complete", 
                                                  message="Query processing complete")
                    except Exception as e:
                        logger.error(f"Background query error: {str(e)}")
                        # Update progress tracker with error
                        progress_tracker = ProgressTracker.get_tracker(session_id, "query")
                        if progress_tracker:
                            progress_tracker.update(100, 100, status="error", 
                                                  message=f"Error: {str(e)}")
                
                query_thread = threading.Thread(target=process_query_background)
                query_thread.daemon = True
                query_thread.start()
            else:
                # Process query synchronously for short questions
                answer = process_query(question, session_id)
    
    return render_template('query.html', 
                          question=question, 
                          answer=answer, 
                          in_progress=in_progress)

@app.route('/api/query/result')
def get_query_result():
    """API endpoint to get the result of an asynchronous query."""
    result = session.get('last_query_result')
    if result:
        # Clear from session
        session.pop('last_query_result', None)
        return jsonify({"result": result})
    return jsonify({"error": "No result available"}), 404

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
    {% block scripts %}{% endblock %}
</body>
</html>''')
    
    # Create upload progress template
    with open('templates/upload_progress.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}SimpleRAG - Upload Progress{% endblock %}

{% block content %}
<h2>Upload Progress</h2>
<p>Your document is being indexed. This process may take a few minutes depending on the document size.</p>

<div class="card mt-4">
    <div class="card-header">
        <h5>Indexing Progress</h5>
    </div>
    <div class="card-body">
        <div class="progress mb-3">
            <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" 
                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">
                0%
            </div>
        </div>
        
        <div id="status-message" class="alert alert-info">
            Starting indexing process...
        </div>
        
        <div class="text-center mt-3">
            <div id="loading-spinner" class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
        
        <div class="mt-3">
            <p>Document: <span id="current-file">Preparing...</span></p>
        </div>
        
        <div class="d-grid gap-2 d-md-flex justify-content-md-end mt-4">
            <a href="{{ url_for('query') }}" class="btn btn-outline-primary" id="finish-button" style="display: none;">
                Continue to Ask Questions
            </a>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // Poll for indexing progress
    let pollInterval;
    let completed = false;
    
    function updateProgress() {
        fetch('/api/progress/index_document')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Progress information not available');
                }
                return response.json();
            })
            .then(data => {
                // Update progress bar
                const percentage = data.percentage || 0;
                const progressBar = document.getElementById('progress-bar');
                progressBar.style.width = `${percentage}%`;
                progressBar.textContent = `${percentage}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
                
                // Update status message
                const statusMessage = document.getElementById('status-message');
                statusMessage.textContent = data.message || 'Processing...';
                
                // Update current file
                const currentFile = document.getElementById('current-file');
                currentFile.textContent = data.current_file || 'Processing...';
                
                // Handle completion or error
                if (data.status === 'complete') {
                    completeProcess(true, data.message);
                } else if (data.status === 'error') {
                    completeProcess(false, data.message);
                }
            })
            .catch(error => {
                console.error('Error fetching progress:', error);
            });
    }
    
    function completeProcess(success, message) {
        if (!completed) {
            completed = true;
            clearInterval(pollInterval);
            
            // Hide spinner
            document.getElementById('loading-spinner').style.display = 'none';
            
            // Update progress bar
            const progressBar = document.getElementById('progress-bar');
            progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            
            if (success) {
                progressBar.classList.add('bg-success');
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';
                
                // Update status message
                const statusMessage = document.getElementById('status-message');
                statusMessage.classList.remove('alert-info');
                statusMessage.classList.add('alert-success');
                statusMessage.textContent = message || 'Document indexed successfully!';
                
                // Show finish button
                document.getElementById('finish-button').style.display = 'block';
            } else {
                progressBar.classList.add('bg-danger');
                
                // Update status message
                const statusMessage = document.getElementById('status-message');
                statusMessage.classList.remove('alert-info');
                statusMessage.classList.add('alert-danger');
                statusMessage.textContent = message || 'An error occurred during indexing.';
            }
        }
    }
    
    // Start polling when page loads
    document.addEventListener('DOMContentLoaded', function() {
        // Start polling immediately
        updateProgress();
        
        // Then poll every second
        pollInterval = setInterval(updateProgress, 1000);
        
        // Set a timeout to stop polling after 10 minutes
        setTimeout(function() {
            if (!completed) {
                clearInterval(pollInterval);
                completeProcess(false, 'Indexing timed out. The document might be too large or complex.');
            }
        }, 10 * 60 * 1000);
    });
</script>
{% endblock %}''')

    # Create the remaining templates and static files from the provided definitions
    return True

# Initialize services on startup
initialize_services()

if __name__ == '__main__':
    # Create template files if they don't exist
    create_templates()
    
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))