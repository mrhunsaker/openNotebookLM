#!/bin/bash

# AMD ROCm OpenNotebookLM Setup Script
# This script helps set up the OpenNotebookLM project with AMD ROCm GPU support

set -e

echo "🚀 Setting up OpenNotebookLM with AMD ROCm GPU Support"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check for required commands
check_dependencies() {
    print_status "Checking dependencies..."

    local deps=("podman" "podman-compose" "git")
    local missing_deps=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install missing dependencies and run this script again"
        exit 1
    fi

    print_success "All dependencies found"
}

# Check ROCm installation
check_rocm() {
    print_status "Checking ROCm installation..."

    if command -v rocm-smi &> /dev/null; then
        print_success "ROCm found - checking GPU status..."
        rocm-smi --showproductname

        # Check if user is in required groups
        if groups | grep -q render && groups | grep -q video; then
            print_success "User is in render and video groups"
        else
            print_warning "User should be in render and video groups for GPU access"
            print_status "Run: sudo usermod -a -G render,video \$USER"
            print_status "Then log out and back in"
        fi
    else
        print_warning "ROCm not detected. GPU acceleration may not work"
        print_status "Install ROCm: https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.6/page/Introduction_to_AMD_ROCm_Installation_Guide_for_Linux.html"
    fi
}

# Setup project directories
setup_directories() {
    print_status "Setting up project directories..."

    local base_dir="$HOME/downloads/openLM"

    # Create main directories
    mkdir -p "$base_dir"/{app,sources,docling_service,marker_service}
    mkdir -p "$base_dir/app"/{models,chroma_db}

    # Create sample notebook directories
    mkdir -p "$base_dir/sources"/{sample_research,project_docs}

    print_success "Project directories created at $base_dir"
}

# Configure environment
setup_environment() {
    print_status "Setting up environment configuration..."

    local base_dir="$HOME/downloads/openLM"

    # Detect GPU architecture for RX 7700S (RDNA3)
    local gpu_arch="gfx1031"
    local hsa_override="10.3.0"

    # Check if we can detect the actual GPU
    if command -v rocm-smi &> /dev/null; then
        local gpu_info=$(rocm-smi --showproductname 2>/dev/null | grep -i "7700S" || true)
        if [[ -n "$gpu_info" ]]; then
            print_success "Detected AMD Radeon RX 7700S - using RDNA3 settings"
        else
            print_warning "Could not detect RX 7700S specifically, using default RDNA3 settings"
        fi
    fi

    cat > "$base_dir/podman.env" << EOF
# General Application Settings
APP_PORT=8000
SOURCE_DIR=/downloads/openLM/sources

# Granite LLM Settings using Transformers
# Choose your preferred Granite model from Hugging Face
# Recommended for RX 7700S (12GB VRAM):
# - ibm/granite-3b-code-instruct (fast, good for most tasks)
# - ibm/granite-7b-instruct (balanced performance)
# - ibm/granite-13b-instruct-v2 (requires model sharding)
GRANITE_MODEL_NAME=ibm/granite-3b-code-instruct

# Service Ports
DOCLING_API_PORT=8001
MARKER_API_PORT=8003

# ChromaDB Settings
CHROMA_PERSIST_DIR=/app/chroma_db

# ROCm/AMD GPU Settings for RX 7700S (RDNA3)
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm
PYTORCH_ROCM_ARCH=$gpu_arch
HSA_OVERRIDE_GFX_VERSION=$hsa_override
HIP_VISIBLE_DEVICES=0
EOF

    print_success "Environment configuration created"
}

# Create sample documents
create_samples() {
    print_status "Creating sample documents..."

    local base_dir="$HOME/downloads/openLM"

    # Sample research document
    cat > "$base_dir/sources/sample_research/research_notes.md" << 'EOF'
# AI Research Notes

## Introduction
This document contains notes on artificial intelligence research and applications.

## Machine Learning Fundamentals
Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

### Supervised Learning
- Uses labeled training data
- Common algorithms: Linear Regression, Decision Trees, Neural Networks

### Unsupervised Learning
- Works with unlabeled data
- Common techniques: Clustering, Dimensionality Reduction

## Deep Learning
Deep learning uses neural networks with multiple layers to learn complex patterns.

### Applications
- Computer Vision
- Natural Language Processing
- Speech Recognition

## Conclusion
AI continues to evolve rapidly with new breakthroughs in various domains.
EOF

    # Sample project document
    cat > "$base_dir/sources/project_docs/project_plan.txt" << 'EOF'
Project Alpha - Development Plan

Objective: Develop a local RAG system with GPU acceleration

Phase 1: Infrastructure Setup
- Container orchestration with Podman
- GPU driver installation and configuration
- Base service development

Phase 2: Model Integration
- Hugging Face Transformers integration
- ROCm optimization for AMD GPUs
- Model caching and optimization

Phase 3: User Interface
- NiceGUI frontend development
- Accessibility compliance (WCAG 2.x)
- Multi-notebook functionality

Phase 4: Testing and Optimization
- Performance benchmarking
- Memory usage optimization
- User experience testing

Timeline: 8-12 weeks
Resources: 1-2 developers, AMD GPU hardware
EOF

    print_success "Sample documents created"
}

# Main setup function
main() {
    echo
    print_status "Starting OpenNotebookLM setup for AMD ROCm..."
    echo

    check_dependencies
    echo

    check_rocm
    echo

    setup_directories
    echo

    setup_environment
    echo

    create_samples
    echo

    print_success "Setup completed successfully!"
    echo

    print_status "Next steps:"
    echo "1. Navigate to ~/downloads/openLM/"
    echo "2. Build containers: podman-compose build"
    echo "3. Start services: podman-compose up -d"
    echo "4. Access the application at: http://localhost:8000"
    echo

    print_warning "Note: First run will download Granite models from Hugging Face"
    print_warning "This may take several minutes depending on your internet connection"
    echo

    print_status "For troubleshooting, check the logs with:"
    echo "podman-compose logs [service_name]"
    echo

    print_success "Happy RAGing with your AMD GPU! 🚀"
}

# Run main function
main "$@"
