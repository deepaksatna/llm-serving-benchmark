# GPU Profiling Guide with NVIDIA Nsight Systems

**Author:** Deepak Soni

This guide explains how to use NVIDIA Nsight Systems (nsys) to profile LLM inference workloads and analyze GPU performance.

## Why GPU Profiling?

GPU profiling with nsys helps you:

- **Identify bottlenecks:** Understand where time is spent (compute vs memory vs CPU)
- **Optimize performance:** Find opportunities for kernel fusion, memory optimization
- **Compare frameworks:** See how different inference engines utilize the GPU
- **Debug issues:** Identify CUDA errors, memory leaks, synchronization problems

## Profiling Setup

### Prerequisites

1. **NVIDIA Nsight Systems CLI:** Available in our tools volume or downloadable from NVIDIA
2. **Root access:** Required for detailed profiling
3. **GPU access:** Pod must have GPU allocated

### Installing nsys in Container

For containers without nsys pre-installed:

```bash
# Check if nsys is available
which nsys

# If not, install from our tools volume
dpkg --force-depends -i /tools/nsight-systems-cli-2025.6.1.deb

# Add to PATH
export PATH="/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"

# Verify installation
nsys --version
```

## Running GPU Profiles

### Basic Profiling Command

```bash
nsys profile \
    --output=/results/my_profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,cudnn,cublas \
    --duration=30 \
    <your-inference-command>
```

### Profiling Options Explained

| Option | Description |
|--------|-------------|
| `--output` | Output file path (without extension) |
| `--trace=cuda` | Capture CUDA API calls and kernel launches |
| `--trace=nvtx` | Capture NVTX markers (framework annotations) |
| `--trace=cudnn` | Capture cuDNN operations |
| `--trace=cublas` | Capture cuBLAS matrix operations |
| `--duration=30` | Profile for 30 seconds |
| `--cuda-memory-usage=true` | Track GPU memory allocations |
| `--cudabacktrace=kernel` | Capture backtraces for kernel launches |
| `--sample=process-tree` | Sample CPU execution |

### Example: Profile Inference Request

```bash
# Profile a single inference request
nsys profile \
    --output=/results/tgi_llama_profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    --duration=30 \
    curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"llama-3-8b","prompt":"Explain AI","max_tokens":256}'
```

## Profile Results

### Generated Files

Each profiling session generates:

| File | Description |
|------|-------------|
| `*.nsys-rep` | Main profile report (view in Nsight Systems GUI) |
| `*.sqlite` | SQLite database with detailed metrics |
| `*.qdstrm` | Intermediate stream file (temporary) |

### Our Benchmark Profiles

We captured nsys profiles for all framework/model combinations:

```
results/
├── nim/
│   ├── llama3-nim/
│   │   ├── llama3_nim_profile.nsys-rep      # 66KB
│   │   └── llama3_nim_trtllm_profile.nsys-rep
│   └── mistral-nim/
│       └── mistral_nim_trtllm_profile.nsys-rep  # 68KB
├── sglang/
│   ├── sglang-llama/
│   │   └── sglang_llama_profile.nsys-rep    # 69KB
│   └── sglang-mistral/
│       └── sglang_mistral_profile.nsys-rep  # 70KB
└── tgi/
    ├── tgi-llama/
    │   └── tgi_llama_profile.nsys-rep       # 63KB
    └── tgi-mistral/
        └── tgi_mistral_profile.nsys-rep     # 58KB
```

## Analyzing Profiles

### Using Nsight Systems GUI (Recommended)

1. **Download** Nsight Systems from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems)
2. **Open** the `.nsys-rep` file
3. **Analyze** the timeline view

### Key Metrics to Look For

#### 1. GPU Utilization Timeline

Look for gaps in GPU activity - these indicate:
- CPU bottlenecks (data preprocessing)
- Memory transfer overhead
- Synchronization delays

#### 2. Kernel Execution Time

Compare kernel execution times across frameworks:

```
NIM (TensorRT-LLM):
├── Fused attention kernels: ~5-10ms per layer
├── FFN kernels: ~3-5ms per layer
└── Total per token: ~30-32ms

SGLang/vLLM:
├── Flash attention kernels: ~8-12ms per layer
├── FFN kernels: ~5-8ms per layer
└── Total per token: ~35-40ms
```

#### 3. Memory Operations

Track memory transfers and allocations:
- High H2D (Host to Device) transfers indicate inefficient batching
- Frequent allocations suggest memory fragmentation

### Command-Line Analysis

```bash
# Export statistics to console
nsys stats my_profile.nsys-rep

# Export to CSV for further analysis
nsys stats my_profile.nsys-rep --format csv --output stats.csv

# Get kernel summary
nsys stats my_profile.nsys-rep --report cuda_kern_sum
```

### Example Output Analysis

```
CUDA Kernel Statistics:

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Kernel Name
 --------  ---------------  ---------  --------  -----------
    45.2%       12,345,678        256    48,225  ampere_fp16_s1688gemm_fp16_...
    22.1%        6,023,456        256    23,529  flash_fwd_kernel<...>
    15.3%        4,178,901        256    16,324  elementwise_kernel<...>
     8.9%        2,431,567        128    18,996  layernorm_kernel<...>
```

## Profile Comparison

### NIM vs Open Source Frameworks

**Key Differences Observed:**

1. **Kernel Fusion:**
   - NIM: Highly fused operations (attention + FFN in fewer kernels)
   - vLLM/SGLang/TGI: More granular kernels

2. **Memory Efficiency:**
   - NIM: Pre-allocated memory pools
   - Others: Dynamic allocation with some fragmentation

3. **Batch Processing:**
   - NIM: Optimized in-flight batching
   - SGLang: RadixAttention for prefix sharing
   - vLLM: PagedAttention for memory efficiency

### Throughput Correlation

Profile metrics correlate with benchmark results:

| Framework | Kernel Time/Token | Measured Throughput |
|-----------|-------------------|---------------------|
| NIM | ~31ms | 31.7 tok/s |
| SGLang | ~35ms | 28.5 tok/s |
| TGI | ~36ms | 28.1 tok/s |

## Best Practices

### 1. Profile Representative Workloads

```bash
# Don't just profile single requests
# Profile with realistic batch sizes and prompt lengths
for i in {1..10}; do
    curl -s http://localhost:8000/v1/completions \
        -d '{"prompt":"Long prompt here...","max_tokens":256}' &
done
wait
```

### 2. Warm Up Before Profiling

```bash
# Run warmup requests first
for i in {1..3}; do
    curl -s http://localhost:8000/v1/completions \
        -d '{"prompt":"warmup","max_tokens":10}'
done

# Then profile
nsys profile ...
```

### 3. Profile at Different Load Levels

- Single request (latency focused)
- Concurrent requests (throughput focused)
- Sustained load (stability focused)

## Troubleshooting Profiling

### Issue: "Permission denied"

```bash
# Run container as root
securityContext:
  runAsUser: 0
```

### Issue: "No CUDA activities captured"

```bash
# Ensure GPU is accessible
nvidia-smi

# Check CUDA visibility
echo $CUDA_VISIBLE_DEVICES
```

### Issue: Profile file too large

```bash
# Limit duration
nsys profile --duration=10 ...

# Reduce trace types
nsys profile --trace=cuda ...  # Only CUDA
```

## Automation Script

Use our provided script for easy profiling:

```bash
./scripts/run-nsys-profile.sh <pod-name> <output-name>

# Examples:
./scripts/run-nsys-profile.sh llama3-nim-xxx llama3_nim_profile
./scripts/run-nsys-profile.sh sglang-llama-xxx sglang_llama_profile
./scripts/run-nsys-profile.sh tgi-mistral-xxx tgi_mistral_profile
```

## Resources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [CUDA Profiling Best Practices](https://developer.nvidia.com/blog/cuda-pro-tip-profiling-mpi-applications/)
- [Understanding GPU Timelines](https://developer.nvidia.com/blog/understanding-gpu-timelines-with-nsight-systems/)
