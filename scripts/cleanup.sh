#!/bin/bash
# Cleanup Script - Remove all deployments
# Author: Deepak Soni
# Usage: ./cleanup.sh [nim|sglang|tgi|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../deployments"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cleanup_nim() {
    log_info "Removing NIM deployments..."
    kubectl delete -f "$DEPLOY_DIR/nim/" --ignore-not-found=true 2>/dev/null || true
}

cleanup_sglang() {
    log_info "Removing SGLang deployments..."
    kubectl delete -f "$DEPLOY_DIR/sglang/" --ignore-not-found=true 2>/dev/null || true
}

cleanup_tgi() {
    log_info "Removing TGI deployments..."
    kubectl delete -f "$DEPLOY_DIR/tgi/" --ignore-not-found=true 2>/dev/null || true
}

cleanup_common() {
    log_info "Removing common resources..."
    kubectl delete -f "$DEPLOY_DIR/common/" --ignore-not-found=true 2>/dev/null || true
}

case "${1:-all}" in
    nim)
        cleanup_nim
        ;;
    sglang)
        cleanup_sglang
        ;;
    tgi)
        cleanup_tgi
        ;;
    all)
        cleanup_nim
        cleanup_sglang
        cleanup_tgi
        log_warn "Common resources (namespace) preserved. Run './cleanup.sh common' to remove."
        ;;
    common)
        cleanup_nim
        cleanup_sglang
        cleanup_tgi
        cleanup_common
        ;;
    *)
        echo "Usage: $0 [nim|sglang|tgi|all|common]"
        exit 1
        ;;
esac

log_info "Cleanup complete!"
kubectl get pods -n nim-bench 2>/dev/null || echo "Namespace removed or empty"
