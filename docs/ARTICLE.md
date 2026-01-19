# Benchmarking LLM Inference Frameworks: A Complete Guide

**Comparing NVIDIA NIM, vLLM, SGLang, and HuggingFace TGI on GPU Infrastructure**

---

## Introduction

Deploying Large Language Models (LLMs) in production requires choosing the right inference framework. With multiple options available - NVIDIA NIM, vLLM, SGLang, and HuggingFace TGI - selecting the best one for your use case can be challenging.

This guide presents comprehensive benchmarks conducted on real GPU infrastructure, along with GPU profiling insights using NVIDIA Nsight Systems (nsys) to help you make informed decisions.

## Test Environment

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **GPU** | 2x NVIDIA A10 (24GB VRAM each) |
| **CPU** | 16 cores per node |
| **Memory** | 256GB per node |
| **Storage** | OCI File Storage Service (FSS) |
| **Platform** | Oracle Kubernetes Engine (OKE) |

### Software Versions

| Framework | Version | Backend |
|-----------|---------|---------|
| NVIDIA NIM | 1.8.4 | TensorRT-LLM |
| vLLM | 0.6.4 | PagedAttention |
| SGLang | 0.4.7 | RadixAttention + FlashInfer |
| HuggingFace TGI | 2.4.1 | FlashAttention |

### Models Tested

- **Llama-3-8B-Instruct** (Meta)
- **Mistral-7B-Instruct v0.3** (Mistral AI)

Both models run in FP16 precision on a single A10 GPU.

---

## Benchmark Results

### Throughput Comparison

We measured tokens per second (tok/s) across different output lengths:

#### Llama-3-8B Performance

| Framework | 32 tokens | 128 tokens | 512 tokens | Average |
|-----------|-----------|------------|------------|---------|
| **NIM** | 31.7 | 31.7 | 31.7 | **31.7 tok/s** |
| **SGLang** | 28.7 | 28.8 | 28.5 | **28.5 tok/s** |
| **TGI** | 28.0 | 28.2 | 28.1 | **28.1 tok/s** |
| **vLLM** | ~28 | ~28 | ~28 | **~28 tok/s** |

#### Mistral-7B Performance

| Framework | 32 tokens | 128 tokens | 512 tokens | Average |
|-----------|-----------|------------|------------|---------|
| **NIM** | 33.7 | 33.7 | 33.6 | **33.7 tok/s** |
| **SGLang** | 30.5 | 30.3 | 30.0 | **30.2 tok/s** |
| **TGI** | 29.7 | 29.7 | 29.6 | **29.6 tok/s** |
| **vLLM** | ~29 | ~29 | ~29 | **~29 tok/s** |

### Key Findings

1. **NVIDIA NIM is fastest** - 10-15% higher throughput than open-source alternatives
2. **SGLang edges out TGI and vLLM** - RadixAttention provides slight advantage
3. **Performance is consistent** - Throughput remains stable across output lengths
4. **Mistral-7B is faster** - Smaller parameter count yields higher tok/s

---

## Understanding Performance with GPU Profiling

### Why Use nsys Profiling?

Numbers tell only part of the story. To truly understand why NIM outperforms others, we used NVIDIA Nsight Systems (nsys) to capture GPU execution traces.

### What nsys Reveals

GPU profiling shows:

1. **Kernel Execution Patterns** - How efficiently operations utilize GPU
2. **Memory Operations** - Data transfer overhead
3. **Synchronization Points** - Where the GPU waits for CPU
4. **Kernel Fusion** - Optimized vs granular operations

### Profile Analysis

#### NIM (TensorRT-LLM)

```
Timeline Analysis:
├── Highly fused attention kernels
├── Minimal CPU-GPU synchronization
├── Pre-allocated memory pools
└── Optimized batch scheduling
```

**Key Observation:** NIM's TensorRT-LLM backend fuses multiple operations into optimized CUDA kernels, reducing kernel launch overhead and memory transfers.

#### SGLang (FlashInfer)

```
Timeline Analysis:
├── RadixAttention for prefix caching
├── FlashInfer optimized kernels
├── Automatic KV cache sharing
└── Efficient memory management
```

**Key Observation:** SGLang's RadixAttention shines when requests share common prefixes (chatbots, RAG systems).

#### TGI (FlashAttention)

```
Timeline Analysis:
├── FlashAttention-2 kernels
├── Rust-based server (low overhead)
├── Continuous batching
└── Stable, predictable performance
```

**Key Observation:** TGI provides consistent, reliable performance with battle-tested implementation.

### Profile Files Generated

Each benchmark run generated nsys profiles:

```
results/
├── nim/llama3-nim/llama3_nim_profile.nsys-rep (66KB)
├── nim/mistral-nim/mistral_nim_trtllm_profile.nsys-rep (68KB)
├── sglang/sglang-llama/sglang_llama_profile.nsys-rep (69KB)
├── sglang/sglang-mistral/sglang_mistral_profile.nsys-rep (70KB)
├── tgi/tgi-llama/tgi_llama_profile.nsys-rep (63KB)
└── tgi/tgi-mistral/tgi_mistral_profile.nsys-rep (58KB)
```

These profiles can be opened in Nsight Systems GUI for detailed analysis.

---

## Performance Across GPU Types

**Important:** Our benchmarks were conducted on NVIDIA A10 GPUs. Results will vary significantly on different hardware.

### Expected Scaling by GPU

| GPU | VRAM | Approx. Llama-3-8B Throughput |
|-----|------|------------------------------|
| **A10** | 24GB | 28-34 tok/s (this benchmark) |
| **A100 40GB** | 40GB | 50-70 tok/s (~2x) |
| **A100 80GB** | 80GB | 55-75 tok/s |
| **H100** | 80GB | 100-150 tok/s (~3-4x) |
| **H200** | 141GB | 120-180 tok/s |
| **B200** | 192GB | 150-250+ tok/s |

### Factors Affecting Performance

1. **Memory Bandwidth** - H100's HBM3 vs A10's GDDR6
2. **Tensor Cores** - FP8 support on H100/B200
3. **Compute Capability** - SM count and clock speed
4. **Batch Size** - Larger GPUs handle bigger batches

---

## Deployment Guide

### Step 1: Prepare Your Environment

```bash
# Clone the benchmark repository
git clone <repository>
cd llm-inference-benchmark

# Create namespace
kubectl apply -f deployments/common/00-namespace.yaml
```

### Step 2: Configure for Your Hardware

Edit `configs/environment.yaml`:

```yaml
gpu_nodes:
  - name: "Your GPU Node"
    ip: "YOUR_NODE_IP"
    gpu: "NVIDIA H100"  # Your GPU type

storage:
  paths:
    models:
      llama: "/your/path/to/llama-model"
      mistral: "/your/path/to/mistral-model"
```

Update deployment YAMLs with your node information:

```yaml
# In each deployment file
nodeName: your-gpu-node-ip
# Or use node selector
nodeSelector:
  gpu-type: h100
```

### Step 3: Deploy Inference Server

Choose your framework:

```bash
# Option 1: NVIDIA NIM (highest performance, requires NGC key)
kubectl create secret generic ngc-api-secret -n nim-bench \
  --from-literal=NGC_API_KEY=<your-key>
kubectl apply -f deployments/nim/

# Option 2: SGLang (great performance, free)
kubectl apply -f deployments/sglang/

# Option 3: TGI (easy deployment, free)
kubectl apply -f deployments/tgi/

# Option 4: vLLM (memory efficient, free)
kubectl apply -f deployments/vllm/
```

### Step 4: Run Benchmarks

```bash
# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l framework=sglang -n nim-bench

# Run benchmark
./scripts/benchmark.sh sglang llama
```

### Step 5: GPU Profiling (Optional)

```bash
# Get pod name
POD=$(kubectl get pods -n nim-bench -l app=sglang-llama -o jsonpath='{.items[0].metadata.name}')

# Run nsys profiling
./scripts/run-nsys-profile.sh $POD sglang_llama_profile
```

---

## Choosing the Right Framework

### Decision Matrix

| Your Priority | Recommended Framework |
|---------------|----------------------|
| Maximum Performance | NVIDIA NIM |
| Budget Conscious | SGLang or vLLM |
| Quick Deployment | TGI |
| Memory Efficiency | vLLM |
| Prefix-Heavy Workloads | SGLang |
| Enterprise Support | NVIDIA NIM |
| Research/Experimentation | SGLang |

### Cost-Performance Trade-off

```
Performance/Cost Ratio (normalized to TGI)

NIM     ████████████████████ 1.13x (best perf, license cost)
SGLang  ██████████████████   1.02x (good balance)
vLLM    █████████████████    1.00x (baseline)
TGI     █████████████████    1.00x (easiest setup)
```

---

## Troubleshooting Common Issues

### Container Won't Start

```bash
# Check pod status
kubectl describe pod <pod-name> -n nim-bench

# Common fixes:
# 1. GPU not allocated - check tolerations
# 2. Image pull error - verify credentials
# 3. Model not found - check volume mounts
```

### Low Throughput

```bash
# Profile to identify bottleneck
nsys profile --trace=cuda,nvtx ...

# Common causes:
# 1. Small batch size
# 2. CPU bottleneck
# 3. Memory fragmentation
```

### nsys Profiling Fails

```bash
# Install nsys if missing
dpkg --force-depends -i /tools/nsight-systems-cli-2025.6.1.deb
export PATH="/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"

# Verify
nsys --version
```

---

## Conclusion

### Summary

| Framework | Throughput | Ease of Use | Cost | Best For |
|-----------|------------|-------------|------|----------|
| **NIM** | Highest | Medium | License | Production |
| **SGLang** | High | Easy | Free | Research |
| **TGI** | Good | Very Easy | Free | Quick Deploy |
| **vLLM** | Good | Easy | Free | Memory-limited |

### Recommendations

1. **For Production at Scale:** Start with NVIDIA NIM for maximum performance
2. **For Startups/Cost-Sensitive:** SGLang or vLLM offer excellent free alternatives
3. **For Quick POCs:** TGI gets you running fastest
4. **Always Profile:** Use nsys to understand your specific workload's behavior

### Final Thoughts

The LLM inference landscape is rapidly evolving. What's optimal today may change tomorrow. The key is understanding your workload characteristics through profiling and benchmarking on your specific hardware.

This benchmark suite provides the tools and methodology to continuously evaluate as new framework versions are released.

---

## Resources

- **Project Repository:** Contains all YAML files, scripts, and results
- **NVIDIA Nsight Systems:** [Download](https://developer.nvidia.com/nsight-systems)
- **Framework Documentation:**
  - [NVIDIA NIM](https://docs.nvidia.com/nim/)
  - [vLLM](https://docs.vllm.ai/)
  - [SGLang](https://github.com/sgl-project/sglang)
  - [TGI](https://huggingface.co/docs/text-generation-inference)

---

*Benchmarks conducted January 2026 on OCI infrastructure with NVIDIA A10 GPUs.*
