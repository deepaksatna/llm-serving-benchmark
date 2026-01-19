# GPU Profiling with NVIDIA Nsight Systems (nsys)

This directory contains Kubernetes Job definitions for profiling LLM inference servers using NVIDIA Nsight Systems.

## Overview

GPU profiling helps you understand:
- CUDA kernel execution times
- Memory transfer patterns
- GPU utilization efficiency
- Bottlenecks in inference pipeline

## Files

| File | Description |
|------|-------------|
| `00-nsys-configmap.yaml` | Scripts for manual profiling |
| `01-nsys-profiler-job.yaml` | Generic profiler job template |
| `02-nim-profiler.yaml` | NVIDIA NIM profiler |
| `03-sglang-profiler.yaml` | SGLang profiler |
| `04-tgi-profiler.yaml` | HuggingFace TGI profiler |
| `05-vllm-profiler.yaml` | vLLM profiler |

## Prerequisites

1. **nsys package**: The `nsight-systems-cli-2025.6.1.deb` file must be available in the tools volume:
   ```
   /mnt/coecommonfss/llmcore/2026-NIM-vLLM_LLM/tools/nsight-systems-cli-2025.6.1.deb
   ```

2. **Running inference server**: Deploy your inference server before running profiler:
   ```bash
   # Example: Deploy SGLang
   kubectl apply -f ../sglang/01-sglang-llama.yaml

   # Wait for it to be ready
   kubectl logs -f deployment/sglang-llama -n nim-bench
   ```

## Quick Start

### Step 1: Deploy Inference Server

```bash
# Choose one framework to deploy
kubectl apply -f ../sglang/01-sglang-llama.yaml
# or
kubectl apply -f ../tgi/01-tgi-llama.yaml
# or
kubectl apply -f ../nim/01-nim-llama.yaml
```

### Step 2: Run Profiler

```bash
# Profile SGLang
kubectl apply -f 03-sglang-profiler.yaml

# Watch progress
kubectl logs -f job/sglang-profiler -n nim-bench
```

### Step 3: Retrieve Results

```bash
# Copy profiles to local machine
kubectl cp nim-bench/sglang-profiler-xxx:/results/sglang/ ./profiles/

# Or access directly on FSS
ls /mnt/coecommonfss/llmcore/2026-NIM-vLLM_LLM/results/sglang/
```

### Step 4: Analyze with Nsight Systems GUI

1. Download [Nsight Systems](https://developer.nvidia.com/nsight-systems)
2. Open the `.nsys-rep` file
3. Analyze GPU timeline

## Using Generic Profiler

The `01-nsys-profiler-job.yaml` is a template you can customize:

```yaml
env:
- name: SERVICE_URL
  value: "http://your-service:8000"
- name: MODEL_NAME
  value: "your-model-name"
- name: OUTPUT_NAME
  value: "your_profile_name"
- name: PROFILE_DURATION
  value: "30"
```

Apply with your changes:
```bash
kubectl apply -f 01-nsys-profiler-job.yaml
```

## Manual Profiling

For more control, use the ConfigMap scripts:

```bash
# Apply ConfigMap
kubectl apply -f 00-nsys-configmap.yaml

# Exec into inference pod
kubectl exec -it <inference-pod> -n nim-bench -- bash

# Install nsys
/scripts/install-nsys.sh

# Run profiling
/scripts/profile-inference.sh my_profile http://localhost:8000 model-name
```

## Profile Analysis

### View in Nsight Systems GUI

Open `.nsys-rep` file in Nsight Systems for visual analysis:
- Timeline view of CUDA kernels
- Memory transfer patterns
- CPU-GPU synchronization

### Command-line Analysis

```bash
# Get summary statistics
nsys stats profile.nsys-rep

# Export to CSV
nsys stats profile.nsys-rep --format csv --output stats.csv

# Specific reports
nsys stats profile.nsys-rep --report cuda_kern_sum
nsys stats profile.nsys-rep --report cuda_api_sum
nsys stats profile.nsys-rep --report cuda_mem_size_sum
```

## Expected Results

Profile files will be saved to:
```
/mnt/coecommonfss/llmcore/2026-NIM-vLLM_LLM/results/
├── nim/
│   ├── llama3_nim_profile.nsys-rep
│   └── mistral_nim_profile.nsys-rep
├── sglang/
│   ├── sglang_llama_profile.nsys-rep
│   └── sglang_mistral_profile.nsys-rep
├── tgi/
│   ├── tgi_llama_profile.nsys-rep
│   └── tgi_mistral_profile.nsys-rep
└── vllm/
    ├── vllm_llama_profile.nsys-rep
    └── vllm_mistral_profile.nsys-rep
```

## Cleanup

```bash
# Delete profiler jobs
kubectl delete job nim-profiler sglang-profiler tgi-profiler vllm-profiler -n nim-bench

# Or delete all completed jobs
kubectl delete jobs --field-selector status.successful=1 -n nim-bench
```

## Troubleshooting

### nsys not found
```bash
# Install manually in pod
dpkg --force-depends -i /tools/nsight-systems-cli-2025.6.1.deb
export PATH="/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"
```

### Service not ready
```bash
# Check if inference server is running
kubectl get pods -n nim-bench
kubectl logs <inference-pod> -n nim-bench
```

### Permission denied
```bash
# Ensure securityContext allows root
securityContext:
  runAsUser: 0
```
