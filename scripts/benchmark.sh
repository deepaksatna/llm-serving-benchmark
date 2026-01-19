#!/bin/bash
# LLM Inference Benchmark Script
# Runs latency benchmarks and generates results
# Usage: ./benchmark.sh <framework> <model> <service-url>

set -e

FRAMEWORK=${1:-"nim"}
MODEL=${2:-"llama"}
SERVICE_URL=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

# Determine service URL based on framework and model
get_service_url() {
    case "$FRAMEWORK-$MODEL" in
        nim-llama) echo "http://llama3-nim-svc:8000" ;;
        nim-mistral) echo "http://mistral-nim-svc:8000" ;;
        sglang-llama) echo "http://sglang-llama-svc:8000" ;;
        sglang-mistral) echo "http://sglang-mistral-svc:8000" ;;
        tgi-llama) echo "http://tgi-llama-svc:8000" ;;
        tgi-mistral) echo "http://tgi-mistral-svc:8000" ;;
        *) echo "$SERVICE_URL" ;;
    esac
}

# Get model name for API
get_model_name() {
    case "$FRAMEWORK-$MODEL" in
        nim-llama) echo "meta/llama-3.1-8b-instruct" ;;
        nim-mistral) echo "mistralai/mistral-7b-instruct-v03" ;;
        sglang-llama) echo "llama-3-8b-instruct" ;;
        sglang-mistral) echo "mistral-7b-instruct" ;;
        tgi-llama) echo "/models/llama-3-8b-instruct" ;;
        tgi-mistral) echo "/models/mistralai--Mistral-7B-Instruct-v0.3" ;;
    esac
}

URL=$(get_service_url)
MODEL_NAME=$(get_model_name)
OUTPUT_DIR="$RESULTS_DIR/$FRAMEWORK/$FRAMEWORK-$MODEL"

mkdir -p "$OUTPUT_DIR"

log_info "Starting benchmark for $FRAMEWORK $MODEL"
log_info "Service URL: $URL"
log_info "Model: $MODEL_NAME"

# Run benchmark using Python (executes inside benchmark pod)
kubectl exec nim-benchmark -n nim-bench -- python3 -c "
import time
import json
import subprocess
import os

results = []
latency_lines = ['=== $FRAMEWORK $MODEL Benchmark ===']
URL = '$URL'
MODEL_NAME = '$MODEL_NAME'

for tokens in [32, 64, 128, 256, 512]:
    start = time.time()
    result = subprocess.run([
        'curl', '-s', URL + '/v1/completions',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'model': MODEL_NAME,
            'prompt': 'Explain quantum computing in detail',
            'max_tokens': tokens
        })
    ], capture_output=True, text=True)
    elapsed = time.time() - start

    try:
        resp = json.loads(result.stdout)
        output_tokens = resp['usage']['completion_tokens']
        tps = output_tokens / elapsed
        line = f'max_tokens={tokens:3d}: {elapsed:.3f}s, {output_tokens} tokens, {tps:.1f} tok/s'
        latency_lines.append(line)
        results.append({
            'max_tokens': tokens,
            'latency_s': round(elapsed, 3),
            'output_tokens': output_tokens,
            'tokens_per_second': round(tps, 1)
        })
        print(line)
    except Exception as e:
        print(f'Error for {tokens}: {e}')

# Output JSON results
print('JSON_RESULTS_START')
print(json.dumps({
    'framework': '$FRAMEWORK',
    'model': MODEL_NAME,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'results': results
}, indent=2))
print('JSON_RESULTS_END')
"

log_info "Benchmark complete. Results saved to $OUTPUT_DIR"
