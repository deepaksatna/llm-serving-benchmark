# LLM Inference Benchmark Performance Graphs

This directory contains professional visualizations of the benchmark results comparing NVIDIA NIM, vLLM, SGLang, and HuggingFace TGI inference frameworks.

## Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A10 (24GB VRAM) |
| Precision | FP16 |
| Platform | Oracle Cloud Infrastructure (OCI) Kubernetes |
| Models | Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 |

## Graph Descriptions

### 1. Throughput Comparison (`01_throughput_comparison.png`)

**Purpose**: Direct side-by-side comparison of token generation throughput across all frameworks.

**Key Metrics**:
- Y-axis: Tokens per second (tok/s)
- Grouped by model (Llama-3-8B and Mistral-7B)

**Key Findings**:
- NVIDIA NIM leads with 31.7-33.7 tok/s
- SGLang follows at 28.5-30.2 tok/s
- TGI and vLLM perform similarly at ~28-29 tok/s
- Mistral-7B shows ~6% higher throughput than Llama-3-8B across all frameworks

---

### 2. Latency Scaling (`02_latency_scaling.png`)

**Purpose**: Shows how latency scales with increasing token count.

**Key Metrics**:
- X-axis: Number of generated tokens (32, 64, 128, 256, 512)
- Y-axis: Latency in seconds
- Separate lines for each framework

**Key Findings**:
- All frameworks show linear latency scaling
- NIM maintains lowest latency at all token counts
- Latency difference becomes more pronounced at higher token counts
- At 512 tokens: NIM ~16s vs others ~18s

---

### 3. Relative Performance (`03_relative_performance.png`)

**Purpose**: Normalized comparison using NVIDIA NIM as 100% baseline.

**Key Metrics**:
- Bar chart showing percentage of NIM performance
- Highlights performance gap between frameworks

**Key Findings**:
- SGLang achieves 90-91% of NIM performance
- TGI achieves 88-89% of NIM performance
- vLLM achieves 88% of NIM performance
- Performance gap consistent across both models

---

### 4. Framework Radar Chart (`04_framework_radar.png`)

**Purpose**: Multi-dimensional comparison of framework capabilities.

**Dimensions**:
- **Throughput**: Token generation speed
- **Latency**: Response time (inverted - lower is better)
- **Features**: Available optimization features
- **Ease of Use**: Deployment and configuration simplicity
- **Community**: Documentation and community support

**Key Findings**:
- NIM leads in throughput and features (enterprise-grade)
- vLLM excels in ease of use and community support
- SGLang offers best balance of performance and openness
- TGI provides solid all-around performance

---

### 5. Latency Heatmap (`05_latency_heatmap.png`)

**Purpose**: Visual matrix showing latency across token counts and frameworks.

**Key Metrics**:
- Rows: Frameworks (NIM, SGLang, TGI, vLLM)
- Columns: Token counts (32, 64, 128, 256, 512)
- Color intensity: Latency value (lighter = faster)

**Key Findings**:
- Clear gradient showing linear scaling
- NIM row consistently lighter (lower latency)
- Useful for identifying sweet spots

---

### 6. Executive Summary (`06_executive_summary.png`)

**Purpose**: Single-page dashboard combining key metrics for presentations.

**Components**:
1. Performance ranking (bar chart)
2. Technology comparison table
3. Throughput summary table
4. Latency comparison (32 vs 512 tokens)
5. Performance gap analysis
6. Key findings summary

**Use Case**: Ideal for executive presentations and quick overview.

---

### 7. GPU Scaling Projection (`07_gpu_scaling_projection.png`)

**Purpose**: Projected performance on different NVIDIA GPU architectures.

**GPUs Compared**:
| GPU | Memory | Scaling Factor |
|-----|--------|----------------|
| A10 | 24GB | 1.0x (baseline) |
| A100-40GB | 40GB | 2.0x |
| A100-80GB | 80GB | 2.2x |
| H100 | 80GB | 4.0x |
| H200 | 141GB | 5.0x |
| B200 | 192GB | 7.0x |

**Key Findings**:
- Performance scales with memory bandwidth and compute
- H100 expected to deliver ~130 tok/s with NIM
- B200 projected at ~230 tok/s for NIM
- All frameworks maintain relative performance ratios

---

## File Formats

Each graph is available in two formats:

| Format | Use Case | Resolution |
|--------|----------|------------|
| PNG | Web, presentations, reports | 300 DPI |
| PDF | Print, publications, scaling | Vector |

## Regenerating Graphs

To regenerate graphs with updated data:

```bash
cd /path/to/llm-inference-benchmark/results
python3 generate_graphs.py
```

### Customizing Data

Edit the `BENCHMARK_DATA` dictionary in `generate_graphs.py`:

```python
BENCHMARK_DATA = {
    'Llama-3-8B': {
        'NIM': {
            'throughput': 31.70,  # tokens/second
            'latencies': [1.009, 2.017, 4.034, 8.089, 16.172]  # 32, 64, 128, 256, 512 tokens
        },
        # ... other frameworks
    },
    # ... other models
}
```

### Dependencies

```bash
pip install matplotlib numpy seaborn
```

## Color Scheme

| Framework | Color | Hex Code |
|-----------|-------|----------|
| NVIDIA NIM | Green | #76B900 |
| SGLang | Blue | #2E86AB |
| TGI | Orange | #F18F01 |
| vLLM | Purple | #A23B72 |

## Summary Statistics

### Throughput (tokens/second)

| Framework | Llama-3-8B | Mistral-7B | Average |
|-----------|------------|------------|---------|
| NIM | 31.70 | 33.72 | 32.71 |
| SGLang | 28.65 | 30.24 | 29.45 |
| TGI | 28.12 | 29.65 | 28.89 |
| vLLM | 28.00 | 29.00 | 28.50 |

### Performance Ranking

1. **NVIDIA NIM** - 10-18% faster (TensorRT-LLM optimized)
2. **SGLang** - Best open-source option (RadixAttention)
3. **TGI** - Solid production choice (FlashAttention)
4. **vLLM** - Great community support (PagedAttention)

## Notes

- Results may vary based on GPU model, driver version, and workload
- Projections for H100/H200/B200 are estimates based on hardware specifications
- For production deployments, conduct your own benchmarks
- nsys GPU profiles are available in individual framework result directories

## Related Files

- `../generate_graphs.py` - Python script to generate these graphs
- `../nim/` - NVIDIA NIM benchmark results
- `../sglang/` - SGLang benchmark results
- `../tgi/` - HuggingFace TGI benchmark results
- `../vllm/` - vLLM benchmark results
