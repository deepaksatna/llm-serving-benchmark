#!/bin/bash
# Deploy All LLM Inference Servers
# Usage: ./deploy-all.sh [nim|sglang|tgi|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../deployments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Deploy common resources
deploy_common() {
    log_info "Deploying common resources (namespace, RBAC, benchmark pod)..."
    kubectl apply -f "$DEPLOY_DIR/common/00-namespace.yaml"
    kubectl apply -f "$DEPLOY_DIR/common/01-benchmark-pod.yaml"
    log_info "Common resources deployed successfully"
}

# Deploy NIM servers
deploy_nim() {
    log_info "Deploying NVIDIA NIM servers..."

    # Check for NGC API key secret
    if ! kubectl get secret ngc-api-secret -n nim-bench &>/dev/null; then
        log_error "NGC API key secret not found. Please create it first:"
        echo "kubectl create secret generic ngc-api-secret -n nim-bench --from-literal=NGC_API_KEY=<your-key>"
        exit 1
    fi

    kubectl apply -f "$DEPLOY_DIR/nim/01-nim-llama.yaml"
    kubectl apply -f "$DEPLOY_DIR/nim/02-nim-mistral.yaml"
    log_info "NIM servers deployed successfully"
}

# Deploy SGLang servers
deploy_sglang() {
    log_info "Deploying SGLang servers..."
    kubectl apply -f "$DEPLOY_DIR/sglang/01-sglang-llama.yaml"
    kubectl apply -f "$DEPLOY_DIR/sglang/02-sglang-mistral.yaml"
    log_info "SGLang servers deployed successfully"
}

# Deploy TGI servers
deploy_tgi() {
    log_info "Deploying TGI servers..."
    kubectl apply -f "$DEPLOY_DIR/tgi/01-tgi-llama.yaml"
    kubectl apply -f "$DEPLOY_DIR/tgi/02-tgi-mistral.yaml"
    log_info "TGI servers deployed successfully"
}

# Wait for pods to be ready
wait_for_pods() {
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l framework=$1 -n nim-bench --timeout=600s 2>/dev/null || true
    kubectl get pods -n nim-bench -o wide
}

# Main logic
case "${1:-all}" in
    common)
        deploy_common
        ;;
    nim)
        deploy_common
        deploy_nim
        wait_for_pods "nim"
        ;;
    sglang)
        deploy_common
        deploy_sglang
        wait_for_pods "sglang"
        ;;
    tgi)
        deploy_common
        deploy_tgi
        wait_for_pods "tgi"
        ;;
    all)
        deploy_common
        log_warn "Deploying all frameworks requires multiple GPUs"
        log_warn "Recommended: Deploy one framework at a time"
        echo ""
        echo "Choose which to deploy:"
        echo "  ./deploy-all.sh nim     - Deploy NIM only"
        echo "  ./deploy-all.sh sglang  - Deploy SGLang only"
        echo "  ./deploy-all.sh tgi     - Deploy TGI only"
        ;;
    *)
        echo "Usage: $0 [nim|sglang|tgi|common|all]"
        exit 1
        ;;
esac

log_info "Deployment complete!"
kubectl get pods -n nim-bench -o wide
