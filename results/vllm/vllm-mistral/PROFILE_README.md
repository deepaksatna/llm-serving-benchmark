# vLLM Mistral-7B nsys Profile

## Status: Not Yet Generated

The nsys profile for vLLM Mistral was not captured in the current benchmark session.

## How to Generate

1. Deploy vLLM Mistral server:
   ```bash
   kubectl apply -f deployments/vllm/02-vllm-mistral.yaml
   ```

2. Wait for server to be ready:
   ```bash
   kubectl logs -f deployment/vllm-mistral -n nim-bench
   ```

3. Run the vLLM profiler job:
   ```bash
   kubectl apply -f deployments/profiling/05-vllm-profiler.yaml
   ```

4. Copy the profile:
   ```bash
   kubectl cp nim-bench/vllm-profiler-xxx:/results/vllm/vllm_mistral_profile.nsys-rep ./
   ```

## Expected Output

After profiling, you will have:
- `vllm_mistral_profile.nsys-rep` - Main profile file (open in Nsight Systems GUI)
- Profile size: ~60-70KB for 30-second capture

## Benchmark Results Available

The latency benchmark results ARE available in this directory:
- `benchmark_results.json` - JSON format results
- `latency.txt` - Human-readable latency data

Throughput: ~29 tok/s on NVIDIA A10
