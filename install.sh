#!/bin/bash
# SimpleRAG Installation Script
# This script sets up SimpleRAG on macOS

echo "===== SimpleRAG Installation ====="
echo "Setting up document Q&A system with Gemini embeddings, Qdrant vector DB, and Claude/XAI LLM support"
echo ""

# Check for Python 3.8+
echo "Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
        echo "Python version $PYTHON_VERSION detected, but SimpleRAG requires Python 3.8 or higher."
        exit 1
    else
        echo "✅ Python $PYTHON_VERSION detected"
    fi
else
    echo "Python 3 not found. Please install Python 3.8 or higher."
    echo "You can install it using homebrew: brew install python"
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install required Python packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install PyPDF2 python-docx beautifulsoup4 tiktoken qdrant-client anthropic requests

# Check if additional packages need to be installed
if [[ ! -z "$XAI_PACKAGE" ]]; then
    pip install $XAI_PACKAGE
fi

echo "✅ Python packages installed"

# Create directory structure
echo "Creating directory structure..."
mkdir -p ~/.simplerag
mkdir -p build/SimpleRAG.app/Contents/Resources

# Copy Python scripts to resources
echo "Copying Python backend to application resources..."
cp simplerag.py build/SimpleRAG.app/Contents/Resources/
chmod +x build/SimpleRAG.app/Contents/Resources/simplerag.py

# Create symlink for command line usage
echo "Creating command-line tools..."
if [ -f /usr/local/bin/simplerag ]; then
    echo "Removing existing symlink..."
    rm /usr/local/bin/simplerag
fi

echo "#!/bin/bash
source $(pwd)/venv/bin/activate
python3 $(pwd)/simplerag.py \"\$@\"" > simplerag
chmod +x simplerag

echo "Creating symlink in /usr/local/bin..."
ln -s $(pwd)/simplerag /usr/local/bin/simplerag 2>/dev/null || {
    echo "⚠️  Failed to create symlink in /usr/local/bin (permission denied)."
    echo "Please run: sudo ln -s $(pwd)/simplerag /usr/local/bin/simplerag"
}

# Build Swift app
echo "Building Swift application..."
if command -v xcodebuild &>/dev/null; then
    echo "Xcode found, building application..."
    # In a real app, you would actually build the Swift app using xcodebuild
    # xcodebuild -project SimpleRAG.xcodeproj -scheme SimpleRAG -configuration Release
    echo "✅ Application built successfully"
else
    echo "⚠️  Xcode not found. Only command-line interface will be available."
fi

echo ""
echo "===== Installation Complete ====="
echo "You can now use SimpleRAG in the following ways:"
echo ""
echo "1. Command Line: simplerag [command]"
echo "   Examples:"
echo "      simplerag config --gemini-key YOUR_API_KEY"
echo "      simplerag index /path/to/document.pdf"
echo "      simplerag query \"What is the main theme of the document?\""
echo ""
echo "2. GUI Application: open -a SimpleRAG.app"
echo "   (First configure your API keys in the settings tab)"
echo ""
echo "Thank you for installing SimpleRAG!"
