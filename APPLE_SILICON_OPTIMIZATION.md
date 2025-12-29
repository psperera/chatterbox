# Apple Silicon Optimization Guide

## Current Implementation

The codebase has been updated to support Apple Silicon (M1/M2/M3/M4) GPUs using PyTorch's Metal Performance Shaders (MPS) backend.

### Features Implemented:
- ✅ Automatic MPS detection and device selection
- ✅ Proper model loading with CPU fallback for CUDA-saved models
- ✅ MPS validation with helpful error messages
- ✅ Inference mode optimizations (`@torch.inference_mode()`)

## MLX Alternative

For potentially better performance on Apple Silicon, consider using the MLX-optimized version:

**MLX Chatterbox Model**: [mlx-community/chatterbox-fp16](https://huggingface.co/mlx-community/chatterbox-fp16)

### Key Advantages of MLX:
- **Native Apple Silicon optimization**: MLX is Apple's native ML framework
- **FP16 precision**: More memory efficient (half precision)
- **Potentially faster**: Optimized specifically for Apple Silicon architecture
- **Lower memory usage**: FP16 reduces memory footprint by ~50%

### Installation:
```bash
pip install -U mlx-audio
```

### Usage:
```bash
# Voice Cloning
mlx_audio.tts.generate --model mlx-community/chatterbox-fp16 \
  --text "Your text here" \
  --ref_audio path_to_file.wav \
  --play

# Default Voice
mlx_audio.tts.generate --model mlx-community/chatterbox-fp16 \
  --text "Your text here" \
  --play
```

**Note**: The MLX version is currently for `chatterbox-turbo` (English only), not the multilingual version.

## Performance Comparison

To compare PyTorch MPS vs MLX performance on your M4 Max:

1. **PyTorch MPS** (current implementation):
   - Run: `./run_multilingual.sh`
   - Check GPU usage: Activity Monitor > Window > GPU History

2. **MLX** (alternative):
   - Install MLX and test with the commands above
   - Compare inference times and memory usage

## Current Optimizations

### 1. Inference Mode
The codebase extensively uses `@torch.inference_mode()` which:
- Disables gradient computation
- Reduces memory overhead
- Improves inference speed

### 2. Device Detection
```python
# Prioritizes: MPS (Mac) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
```

### 3. Model Loading
- Models are loaded to CPU first, then moved to target device
- This handles CUDA-saved models on non-CUDA devices correctly

## Potential Future Optimizations

1. **Mixed Precision (FP16)**: 
   - Currently models run in FP32
   - FP16 could reduce memory usage and potentially speed up inference
   - Note: Some operations may need to stay in FP32 for stability

2. **Model Compilation**:
   - PyTorch 2.0+ `torch.compile()` could provide additional speedups
   - Test with: `model = torch.compile(model)`

3. **Batch Processing**:
   - Process multiple requests in batches when possible
   - Better GPU utilization

4. **MPS-Specific Optimizations**:
   - Some operations may be faster on CPU for MPS
   - Profile and optimize hot paths

## Performance Tips

1. **Monitor GPU Usage**: Use Activity Monitor to verify MPS is being used
2. **Memory Management**: FP16 models use ~50% less memory
3. **Warm-up**: First inference may be slower due to model loading
4. **Batch Size**: Adjust based on available GPU memory

## References

- [MLX Chatterbox Model](https://huggingface.co/mlx-community/chatterbox-fp16)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
