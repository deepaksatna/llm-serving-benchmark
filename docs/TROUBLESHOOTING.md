# Troubleshooting Guide

Common issues and solutions when deploying LLM inference frameworks on Kubernetes.

## Table of Contents

1. [Container Image Issues](#container-image-issues)
2. [GPU and CUDA Issues](#gpu-and-cuda-issues)
3. [Memory Issues](#memory-issues)
4. [Storage and Model Loading](#storage-and-model-loading)
5. [Network and Service Issues](#network-and-service-issues)
6. [Framework-Specific Issues](#framework-specific-issues)
7. [Performance Issues](#performance-issues)

---

## Container Image Issues

### Issue: "short-name mode is enforcing"

**Symptom:**
```
Error: short-name "lmsys/sglang:v0.4.1" did not resolve to an alias
```

**Solution:** Use fully qualified image names:
```yaml
# Wrong
image: lmsys/sglang:v0.4.1

# Correct
image: docker.io/lmsysorg/sglang:v0.4.7-cu124
```

### Issue: "requested access denied" or "pull access denied"

**Symptom:** Image pull fails with authentication error.

**Solutions:**

1. **For public images:** Verify the correct image name
   ```bash
   # Check if image exists
   docker pull docker.io/lmsysorg/sglang:v0.4.7-cu124
   ```

2. **For NGC images (NIM):** Create image pull secret
   ```bash
   kubectl create secret docker-registry ocirsecret \
     -n nim-bench \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password=<NGC_API_KEY>
   ```

3. **Add imagePullSecrets to deployment:**
   ```yaml
   spec:
     imagePullSecrets:
     - name: ocirsecret
   ```

---

## GPU and CUDA Issues

### Issue: "CUDA out of memory"

**Symptom:** Pod crashes with OOM error during model loading.

**Solutions:**

1. **Reduce model precision:**
   ```yaml
   args:
   - --dtype=float16  # or bfloat16
   ```

2. **Enable quantization:**
   ```yaml
   args:
   - --quantization=awq  # or gptq, squeezellm
   ```

3. **Increase GPU memory (if available):**
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: "2"  # Use multiple GPUs
   ```

### Issue: "No GPU detected" or "CUDA not available"

**Symptom:** Container starts but doesn't detect GPU.

**Solutions:**

1. **Verify GPU node has proper labels:**
   ```bash
   kubectl get nodes -o wide
   kubectl describe node <gpu-node> | grep -i gpu
   ```

2. **Check NVIDIA device plugin:**
   ```bash
   kubectl get pods -n kube-system | grep nvidia
   kubectl logs -n kube-system <nvidia-device-plugin-pod>
   ```

3. **Add proper tolerations:**
   ```yaml
   tolerations:
   - key: nvidia.com/gpu
     operator: Exists
     effect: NoSchedule
   ```

4. **Verify node selector/affinity:**
   ```yaml
   nodeName: <your-gpu-node-ip>
   # or
   nodeSelector:
     nvidia.com/gpu: "true"
   ```

---

## Memory Issues

### Issue: Shared Memory Error

**Symptom:**
```
RuntimeError: DataLoader worker is killed by signal: Bus error
```

**Solution:** Add shared memory volume:
```yaml
volumes:
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: "16Gi"

volumeMounts:
- name: shm
  mountPath: /dev/shm
```

### Issue: Pod OOMKilled

**Symptom:** Pod gets killed with status OOMKilled.

**Solution:** Increase memory limits:
```yaml
resources:
  requests:
    memory: "32Gi"
  limits:
    memory: "64Gi"
```

---

## Storage and Model Loading

### Issue: "Model not found" or "No such file or directory"

**Symptom:** Container can't find the model files.

**Solutions:**

1. **Verify model path exists on host:**
   ```bash
   # SSH to GPU node
   ls -la /mnt/coecommonfss/llmcore/models/
   ```

2. **Check volume mount:**
   ```yaml
   volumeMounts:
   - name: fss-models
     mountPath: /models
     readOnly: true

   volumes:
   - name: fss-models
     hostPath:
       path: /mnt/coecommonfss/llmcore/models
       type: Directory  # Use Directory, not DirectoryOrCreate for existing paths
   ```

3. **Verify offline mode settings:**
   ```yaml
   env:
   - name: TRANSFORMERS_OFFLINE
     value: "1"
   - name: HF_HUB_OFFLINE
     value: "1"
   ```

### Issue: "Permission denied" accessing model files

**Symptom:** Container can't read model files due to permissions.

**Solution:** Add security context:
```yaml
spec:
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
```

---

## Network and Service Issues

### Issue: Service not reachable

**Symptom:** Can't connect to inference server.

**Solutions:**

1. **Check pod status:**
   ```bash
   kubectl get pods -n nim-bench -o wide
   kubectl logs <pod-name> -n nim-bench
   ```

2. **Check service endpoints:**
   ```bash
   kubectl get svc -n nim-bench
   kubectl get endpoints -n nim-bench
   ```

3. **Test from within cluster:**
   ```bash
   kubectl exec -it nim-benchmark -n nim-bench -- \
     curl http://<service-name>:8000/health
   ```

### Issue: Connection timeout

**Symptom:** Requests timeout during model loading.

**Solution:** Model loading can take 2-5 minutes. Check pod logs:
```bash
kubectl logs -f <pod-name> -n nim-bench
```

---

## Framework-Specific Issues

### NIM: "NIM cache is read-only"

**Symptom:**
```
Error: /opt/nim/.cache is read-only
```

**Solution:** Add init container to fix permissions:
```yaml
initContainers:
- name: fix-permissions
  image: busybox
  command: ['sh', '-c', 'mkdir -p /opt/nim/.cache && chmod -R 777 /opt/nim/.cache']
  volumeMounts:
  - name: nim-cache
    mountPath: /opt/nim/.cache
```

### NIM: NGC API Key Error

**Symptom:** Authentication failure with NGC.

**Solution:** Create secret correctly:
```bash
kubectl create secret generic ngc-api-secret \
  -n nim-bench \
  --from-literal=NGC_API_KEY=<your-actual-key>
```

### SGLang: "protobuf library not found"

**Symptom:**
```
ImportError: requires the protobuf library but it was not found
```

**Solution:** Install protobuf in container startup:
```yaml
command: ["/bin/bash", "-c"]
args:
- |
  pip install protobuf --quiet --break-system-packages &&
  python3 -m sglang.launch_server ...
```

### SGLang: "externally-managed-environment" (PEP 668)

**Symptom:** pip install fails with PEP 668 error.

**Solution:** Add `--break-system-packages` flag:
```bash
pip install protobuf --break-system-packages
```

### TGI: "Model architecture not supported"

**Symptom:** TGI fails to load certain models.

**Solution:** Check TGI supported models list:
- LLaMA, Llama 2, Llama 3
- Mistral, Mixtral
- Falcon
- GPT-NeoX
- [Full list](https://huggingface.co/docs/text-generation-inference/supported_models)

---

## Performance Issues

### Issue: Low throughput

**Possible Causes & Solutions:**

1. **Insufficient batch size:**
   ```yaml
   args:
   - --max-batch-prefill-tokens=8192
   - --max-concurrent-requests=64
   ```

2. **CUDA graphs not enabled:**
   - Check logs for "Cuda Graphs are enabled"
   - Some models don't support CUDA graphs

3. **Memory fragmentation:**
   - Restart the pod to clear memory
   - Consider using PagedAttention (vLLM)

### Issue: High latency on first request

**Cause:** Model warmup and CUDA kernel compilation.

**Solution:** This is normal. First request can take 10-30 seconds. Subsequent requests will be faster.

---

## Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n nim-bench

# Describe pod for events
kubectl describe pod <pod-name> -n nim-bench

# Get pod logs
kubectl logs <pod-name> -n nim-bench --tail=100

# Exec into pod
kubectl exec -it <pod-name> -n nim-bench -- bash

# Check GPU utilization
kubectl exec <pod-name> -n nim-bench -- nvidia-smi

# Check resource usage
kubectl top pod -n nim-bench
```

---

## Getting Help

If you encounter issues not covered here:

1. Check framework-specific documentation
2. Search GitHub issues for the framework
3. Include these details when reporting issues:
   - Kubernetes version
   - GPU type and driver version
   - Full pod logs
   - YAML configuration used
