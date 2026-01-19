#!/bin/bash
# NVIDIA Nsight Systems Profiling Script
# Author: Deepak Soni
# Captures GPU profiling data for inference servers
# Usage: ./run-nsys-profile.sh <pod-name> <output-name>

set -e

POD_NAME=${1:-""}
OUTPUT_NAME=${2:-"profile"}
NAMESPACE="nim-bench"

if [ -z "$POD_NAME" ]; then
    echo "Usage: $0 <pod-name> <output-name>"
    echo ""
    echo "Examples:"
    echo "  $0 llama3-nim-xxx llama3_nim_profile"
    echo "  $0 sglang-llama-xxx sglang_llama_profile"
    echo "  $0 tgi-llama-xxx tgi_llama_profile"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

log_info "Starting nsys profiling for pod: $POD_NAME"

# Check if nsys is available, if not install it
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- bash -c '
    if ! command -v nsys &> /dev/null; then
        echo "Installing nsys from /tools..."
        if [ -f /tools/nsight-systems-cli-2025.6.1.deb ]; then
            dpkg --force-depends -i /tools/nsight-systems-cli-2025.6.1.deb 2>/dev/null || true
            export PATH="/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"
        else
            echo "ERROR: nsys package not found at /tools/"
            exit 1
        fi
    fi
    nsys --version
'

# Run profiling
log_info "Running nsys profile (30 second capture)..."
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- bash -c "
    export PATH=\"/opt/nvidia/nsight-systems-cli/2025.6.1/bin:\$PATH\"

    nsys profile \\
        --output=/results/${OUTPUT_NAME} \\
        --force-overwrite=true \\
        --trace=cuda,nvtx,cudnn,cublas \\
        --cuda-memory-usage=true \\
        --duration=30 \\
        --sample=process-tree \\
        curl -s http://localhost:8000/v1/completions \\
            -H 'Content-Type: application/json' \\
            -d '{\"model\":\"model\",\"prompt\":\"Write a detailed essay about AI\",\"max_tokens\":256}'

    echo ''
    echo 'Profile saved:'
    ls -la /results/${OUTPUT_NAME}*
"

log_info "Profiling complete!"
log_info "Profile file: /results/${OUTPUT_NAME}.nsys-rep"
