# YuE Hardware Optimization Plan

## Overview

This document outlines a comprehensive plan to optimize YuE for running on various hardware configurations, including older GPUs and limited memory systems, without changing the default behavior or disrupting the normal workflow of the original codebase.

## Current Challenges

Based on the repository analysis, YuE faces several hardware-related challenges:

1. **High GPU Memory Requirements**: The Stage 1 model (7B parameters) requires significant VRAM
2. **Flash Attention Compatibility**: Older GPUs (like Quadro M6000) don't support Flash Attention
3. **Computational Intensity**: Post-processing, diffusion models, and stereo audio processing are resource-intensive
4. **Limited Fallback Options**: Current CPU fallback isn't optimized for actual production use

## Target Hardware Profile

The optimization plan targets the following hardware configuration:

-   1× RTX 2060 Super (6GB VRAM)
-   2× Quadro M6000 (24GB VRAM each, 48GB total)

## Optimization Goals

1. Enable running YuE on hardware without Flash Attention support
2. Efficiently utilize multiple GPUs with different capabilities
3. Provide graceful fallbacks for memory-constrained environments
4. Maintain full compatibility with the original workflow
5. Keep all optimizations optional through configuration

## Phased Implementation Plan

### Phase 1: Basic Compatibility

#### 1.1. Flash Attention Toggle

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--disable_flash_attention", action="store_true",
                   help="Disable Flash Attention for compatibility with older GPUs")

# When loading models
attn_implementation = "eager" if args.disable_flash_attention else "flash_attention_2"

# Add try/except with fallback when loading the model
try:
    stage1_model = AutoModelForCausalLM.from_pretrained(
        args.stage1_model,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        # other parameters...
    )
except Exception as e:
    if "flash_attention" in str(e).lower():
        print(f"Flash Attention error: {e}")
        print("Falling back to standard attention. Consider using --disable_flash_attention")
        # Try again with eager attention
        stage1_model = AutoModelForCausalLM.from_pretrained(
            args.stage1_model,
            device_map="auto" if args.device == "cuda" else None,
            torch_dtype=dtype,
            attn_implementation="eager",
            # other parameters...
        )
    else:
        # Re-raise if it's not a flash attention error
        raise
```

#### 1.2. Task-Based Device Allocation

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--transformer_device", type=str, default="auto",
                   help="Device to use for transformer models (e.g., 'cuda:0,1' for multi-GPU)")
parser.add_argument("--diffusion_device", type=str, default="auto",
                   help="Device to use for diffusion models (e.g., 'cuda:1')")
parser.add_argument("--codec_device", type=str, default="auto",
                   help="Device to use for codec model and audio processing (e.g., 'cuda:2')")

# In model loading code
def assign_model_to_device(model, device_spec, model_name="model"):
    """
    Assigns a model to specified device(s)
    - 'auto': Use CUDA device manager
    - 'cpu': Force CPU
    - 'cuda:N': Use specific GPU
    - 'cuda:N,M': Distribute across multiple GPUs
    """
    if device_spec == "auto":
        if torch.cuda.is_available():
            print(f"Assigning {model_name} to automatic device mapping")
            return model
        else:
            device_spec = "cpu"

    if device_spec == "cpu":
        print(f"Moving {model_name} to CPU")
        return model.to("cpu")

    if "," in device_spec:  # Multiple GPUs specified
        devices = [f"cuda:{idx}" for idx in device_spec.replace("cuda:", "").split(",")]
        print(f"Distributing {model_name} across devices: {devices}")

        # For transformers models, use device_map
        if hasattr(model, "device_map") or model_name == "transformer":
            # Create device map with either even or custom distribution
            num_layers = len([n for n in dict(model.named_modules()) if "layers" in n])
            layer_count = num_layers or 32  # Fallback if can't determine

            device_map = {}
            devices_count = len(devices)
            layers_per_device = layer_count // devices_count

            for i in range(devices_count):
                device_idx = int(devices[i].split(":")[-1])
                start_layer = i * layers_per_device
                end_layer = (i + 1) * layers_per_device if i < devices_count - 1 else layer_count

                for layer_idx in range(start_layer, end_layer):
                    device_map[layer_idx] = device_idx

            # Move model according to device map
            return model.to(devices[0])  # Move first to initial device, then let transformers handle the rest
        else:
            # Simple case - just move to first device for non-transformer models
            return model.to(devices[0])
    else:  # Single GPU
        print(f"Moving {model_name} to {device_spec}")
        return model.to(device_spec)

# Apply device assignment for each model type
stage1_model = assign_model_to_device(stage1_model, args.transformer_device, "transformer")
diffusion_model = assign_model_to_device(diffusion_model, args.diffusion_device, "diffusion") if diffusion_model else None
codec_model = assign_model_to_device(codec_model, args.codec_device, "codec")
```

#### 1.3. Multi-GPU Support Enhancement

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--device_map", type=str, default="auto",
                   help="Specify custom device mapping (auto, balanced, sequential, or specific mapping)")
parser.add_argument("--cuda_devices", type=str, default="",
                   help="Comma-separated list of CUDA device indices to use (e.g., '0,1')")
parser.add_argument("--model_split_strategy", type=str, choices=["layer", "model_type", "hybrid"], default="model_type",
                   help="How to split models across GPUs: by layer, by model type, or hybrid approach")

# In main function - examples of different splitting strategies
if args.model_split_strategy == "model_type" and torch.cuda.device_count() > 1:
    # Better strategy: Split different model types across GPUs
    print("Using model-type splitting strategy across GPUs")

    # For example, with 3 GPUs:
    if torch.cuda.device_count() >= 3:
        # Stage 1 model (large transformer) spans two Quadros
        args.transformer_device = "cuda:1,2"  # Both Quadros
        args.diffusion_device = "cuda:1"      # First Quadro
        args.codec_device = "cuda:0"          # RTX 2060 for codec
    elif torch.cuda.device_count() == 2:
        # With 2 GPUs
        args.transformer_device = "cuda:0,1"  # Span both
        args.diffusion_device = "cuda:1"      # Second GPU
        args.codec_device = "cuda:0"          # First GPU
elif args.model_split_strategy == "layer":
    # Original layer-based splitting within a single model
    if args.cuda_devices:
        devices = [int(x) for x in args.cuda_devices.split(",")]
        if args.device_map == "auto":
            device_map = "auto"
        elif args.device_map == "balanced":
            # Balanced allocation across specified devices
            device_map = {i: devices[i % len(devices)] for i in range(32)}  # Arbitrary layer count
        elif args.device_map == "sequential":
            # First half on first device, second half on second, etc.
            device_map = {}
            # This is a simplified example, actual implementation would inspect model structure
            num_layers = 32  # Placeholder, would detect actual number
            layers_per_device = num_layers // len(devices)
            for i in range(len(devices)):
                for j in range(i * layers_per_device, (i + 1) * layers_per_device):
                    device_map[j] = devices[i]
        else:
            # Custom mapping specified in format "layer1:device1,layer2:device2"
            device_map = {}
            for mapping in args.device_map.split(","):
                if ":" in mapping:
                    layer, device = mapping.split(":")
                    device_map[int(layer)] = int(device)
    else:
        device_map = "auto" if args.device == "cuda" else None
elif args.model_split_strategy == "hybrid":
    # Combine both approaches
    # 1. Split stage1 model by layers across multiple GPUs
    # 2. Assign other models to specific GPUs by type
    # This requires custom implementation based on hardware profile
    pass
```

#### 1.4. PyTorch Compile Optimization

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--enable_torch_compile", action="store_true",
                   help="Enable PyTorch 2.0+ compilation for improved performance on all hardware")
parser.add_argument("--torch_compile_mode", type=str, choices=["default", "reduce-overhead", "max-autotune"],
                   default="reduce-overhead", help="Compilation mode for PyTorch 2.0+")
parser.add_argument("--torch_compile_fullgraph", action="store_true",
                   help="Enable full graph compilation in torch.compile (may increase compilation time)")

# After model loading
def apply_torch_compile(model, mode="reduce-overhead", fullgraph=False):
    """Apply torch.compile if available (PyTorch 2.0+)"""
    if not hasattr(torch, "compile"):
        print("PyTorch version does not support compilation (requires 2.0+)")
        return model

    try:
        print(f"Applying PyTorch compilation with mode: {mode}")
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph
        )
        return compiled_model
    except Exception as e:
        print(f"Error during model compilation: {e}")
        print("Continuing with uncompiled model")
        return model

# Apply compilation to models when appropriate
if args.enable_torch_compile:
    if not args.supports_flash_attn:
        print("Flash Attention not supported - torch.compile will be applied as an alternative optimization")

    stage1_model = apply_torch_compile(stage1_model, args.torch_compile_mode, args.torch_compile_fullgraph)

    if diffusion_model:
        diffusion_model = apply_torch_compile(diffusion_model, args.torch_compile_mode, args.torch_compile_fullgraph)
```

#### 1.5. Memory-Efficient Quantization Options

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit", "4bit_nf4"], default="none",
                   help="Model quantization level (lower bits = less memory, slightly lower quality)")

# When loading models
if args.quantization == "none":
    quantization_config = None
elif args.quantization == "8bit":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.quantization == "4bit":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype
    )
elif args.quantization == "4bit_nf4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

# Use this config when loading models
stage1_model = AutoModelForCausalLM.from_pretrained(
    args.stage1_model,
    device_map=device_map,
    torch_dtype=dtype,
    quantization_config=quantization_config,
    # other parameters...
)
```

### Phase 2: Performance Optimizations

#### 2.1. Selective Audio Processing Levels

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--audio_processing_level", type=str,
                   choices=["minimal", "standard", "full"], default="full",
                   help="Level of audio post-processing to apply (minimal = fastest, full = best quality)")

# In enhanced_audio_mix function (audio_mixing.py)
def enhanced_audio_mix(vocal, instrumental, mix_params=None, sr=44100, processing_level="full"):
    """Enhanced audio mixing with configurable processing levels"""

    # Default parameters based on processing level
    if processing_level == "minimal":
        # Override parameters to disable most effects
        if mix_params is None:
            mix_params = {}
        mix_params = deep_merge_dicts({
            'phase_alignment': {'enabled': True, 'multiband': False},
            'normalization': {'enabled': True},
            'multiband_compression': {'enabled': False},
            'vocal_compression': {'enabled': False},
            'instrumental_compression': {'enabled': False},
            'stereo_width': {'enabled': False},
            'vocal_enhancement': {'enabled': False},
            'vocal_space_carving': {'enabled': False},
            'instrumental_saturation': {'enabled': False},
            'exciter': {'enabled': False},
            # Disable other CPU-intensive effects
        }, mix_params)
    elif processing_level == "standard":
        # Enable core effects but disable the most CPU-intensive ones
        if mix_params is None:
            mix_params = {}
        mix_params = deep_merge_dicts({
            'phase_alignment': {'enabled': True, 'multiband': True},
            'normalization': {'enabled': True},
            'multiband_compression': {'enabled': True},
            'vocal_compression': {'enabled': True},
            'instrumental_compression': {'enabled': False},
            'stereo_width': {'enabled': True, 'width': 1.1},  # Less aggressive
            'vocal_enhancement': {'enabled': True, 'level': 0.5},  # Lower level
            'vocal_space_carving': {'enabled': True, 'level': 0.4},  # Lower level
            'instrumental_saturation': {'enabled': False},
            'exciter': {'enabled': False},
            # Moderate settings for other effects
        }, mix_params)

    # Continue with existing function...
```

#### 2.2. Diffusion Model Optimizations

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--diffusion_optimization", type=str,
                   choices=["none", "faster", "memory_efficient"], default="none",
                   help="Optimization strategy for diffusion models")

# In diffusion_models.py
def denoise(self, x, condition=None, steps=None, sampling_method=None, guidance_scale=None):
    """Enhanced diffusion sampling with optimization strategies"""

    # Override parameters based on optimization strategy
    if self.optimization_strategy == "faster":
        # Use faster sampling method with fewer steps
        sampling_method = sampling_method or "ddim"  # DDIM is faster than DDPM
        steps = steps or max(20, self.diffusion_steps // 2)  # Use fewer steps
    elif self.optimization_strategy == "memory_efficient":
        # Use methods that consume less memory
        sampling_method = sampling_method or "plms"  # PLMS is memory efficient
        steps = steps or max(30, self.diffusion_steps // 2)

    # Existing sampling logic...
```

#### 2.3. Chunked Processing

**Implementation Details:**

```python
# In stage2_inference function (generation.py)
def stage2_inference(model, tokens, codectool, mmtokenizer, device, batch_size=16, chunk_size=None):
    """
    Enhanced Stage 2 inference with chunked processing
    """
    # If chunk_size is provided, process in chunks to reduce memory usage
    if chunk_size and tokens.shape[0] > chunk_size:
        # Process in chunks
        results = []
        for i in range(0, tokens.shape[0], chunk_size):
            end_idx = min(i + chunk_size, tokens.shape[0])
            chunk_tokens = tokens[i:end_idx]

            # Process this chunk
            chunk_result = stage2_inference(
                model, chunk_tokens, codectool, mmtokenizer,
                device, batch_size
            )
            results.append(chunk_result)

        # Combine results
        return torch.cat(results, dim=0)

    # Original processing for the full batch or a single chunk
    # ...existing code...
```

#### 2.4. Parallel Processing Pipeline

**Implementation Details:**

```python
# In main.py
parser.add_argument("--enable_parallel_processing", action="store_true",
                   help="Enable parallel processing across multiple GPUs")

# Example implementation of parallel processing across GPUs
def run_parallel_pipeline(args):
    """
    Run audio generation pipeline with parallelized execution across GPUs
    """
    # Use Python's multiprocessing or threading for non-blocking execution
    import threading

    results = {}
    active_threads = {}

    # Stage 1: Run transformer on GPU 1+2 (Quadros)
    def run_stage1():
        # Move stage1 model to appropriate devices
        results['stage1_tokens'] = stage1_inference(...)
        print("Stage 1 complete")

    # Simultaneously: Run codec preloading on GPU 0 (RTX 2060)
    def prepare_codec():
        # Preload codec model and prepare for encoding/decoding
        results['codec_ready'] = True
        print("Codec preparation complete")

    # Start parallel execution
    t1 = threading.Thread(target=run_stage1)
    t2 = threading.Thread(target=prepare_codec)

    t1.start()
    t2.start()

    # Wait for both to complete
    t1.join()
    t2.join()

    # Stage 2: Run on appropriate GPUs
    # ...continue with sequential or further parallel processing...

    return results
```

### Phase 3: Advanced Optimizations

#### 3.1. Smart Resource Detection

**Implementation Details:**

```python
def detect_hardware_capabilities():
    """Detect hardware capabilities and recommend settings"""
    capabilities = {
        'has_cuda': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [],
        'gpu_memory': [],
        'supports_flash_attn': False,
        'supports_bfloat16': False,
        'cpu_count': os.cpu_count(),
        'recommended_settings': {}
    }

    if capabilities['has_cuda']:
        # Check each GPU
        for i in range(capabilities['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            capabilities['gpu_names'].append(props.name)
            capabilities['gpu_memory'].append(props.total_memory // (1024**2))  # Convert to MB

            # Check Flash Attention compatibility (compute capability >= 8.0)
            if props.major >= 8:
                capabilities['supports_flash_attn'] = True

            # Check bfloat16 support
            # Ampere (compute capability 8.x) and newer support bfloat16
            if props.major >= 8:
                capabilities['supports_bfloat16'] = True

    # Generate recommended settings
    if not capabilities['has_cuda']:
        capabilities['recommended_settings'] = {
            'device': 'cpu',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'minimal',
            'diffusion_optimization': 'faster',
            'disable_flash_attention': True
        }
    elif not capabilities['supports_flash_attn']:
        capabilities['recommended_settings'] = {
            'device': 'cuda',
            'disable_flash_attention': True,
            'enable_torch_compile': True,
            'quantization': '8bit' if max(capabilities['gpu_memory']) >= 12000 else '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'faster'
        }
    elif min(capabilities['gpu_memory']) < 8000:
        # Low memory GPUs
        capabilities['recommended_settings'] = {
            'device': 'cuda',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'memory_efficient'
        }
    else:
        # High-end GPUs
        capabilities['recommended_settings'] = {
            'device': 'cuda',
            'quantization': 'none',
            'audio_processing_level': 'full',
            'diffusion_optimization': 'none'
        }

    return capabilities
```

#### 3.2. Checkpoint and Resume Capability

**Implementation Details:**

```python
# In argument parser (main.py)
parser.add_argument("--enable_checkpointing", action="store_true",
                   help="Enable checkpointing of intermediate results for possible resumption")
parser.add_argument("--resume_from_checkpoint", type=str, default="",
                   help="Resume generation from a saved checkpoint file")

# In main.py
def save_checkpoint(stage, data, output_dir, session_id):
    """Save checkpoint data for potential resumption"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{session_id}_{stage}.pt")
    torch.save(data, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path):
    """Load checkpoint data for resumption"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        data = torch.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
        return data
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
```

#### 3.3. Adaptive Batch Sizing

**Implementation Details:**

```python
def calculate_optimal_batch_size(model, device, input_shape=(1, 1000)):
    """Determine optimal batch size based on available memory"""
    if device == "cpu":
        return 1  # Default to minimal batch size on CPU

    # Start with a small batch size
    batch_size = 1
    max_batch_size = 32  # Upper limit to prevent excessive testing

    # Get initial free memory
    torch.cuda.empty_cache()
    initial_free_memory = torch.cuda.mem_get_info(device)[0]

    # Create an example input
    example_input = torch.zeros((batch_size,) + input_shape[1:], device=device)

    try:
        # Do a test forward pass to account for any lazy initialization
        with torch.no_grad():
            _ = model(example_input)

        # Get memory usage after first forward pass
        memory_after_first = initial_free_memory - torch.cuda.mem_get_info(device)[0]

        # Estimate memory per sample
        memory_per_sample = memory_after_first / batch_size

        # Calculate safe batch size (using 80% of free memory)
        safe_memory = 0.8 * initial_free_memory
        estimated_batch_size = int(safe_memory / memory_per_sample)

        # Constrain to reasonable bounds
        optimal_batch_size = max(1, min(estimated_batch_size, max_batch_size))

        return optimal_batch_size
    except Exception as e:
        print(f"Error during batch size estimation: {e}")
        return 1  # Default to 1 on error
```

## Recommended Configuration for Target Hardware

For the specific hardware configuration (1× RTX 2060 Super + 2× Quadro M6000), we recommend:

1. **Model Distribution**:

    - Stage 1 transformer model: Distributed across both Quadro M6000 GPUs
    - Diffusion model: Single Quadro M6000
    - Codec model and audio processing: RTX 2060 Super

2. **Optimization Settings**:

    ```
    --disable_flash_attention
    --enable_torch_compile
    --torch_compile_mode=reduce-overhead
    --quantization=8bit
    --model_split_strategy=model_type
    --transformer_device=cuda:1,2  # Both Quadros
    --diffusion_device=cuda:1      # First Quadro
    --codec_device=cuda:0          # RTX 2060
    --enable_parallel_processing
    --audio_processing_level=standard
    --diffusion_optimization=faster
    ```

3. **Advanced Configuration**:
    - If the transformer model is still too large for the Quadros even with 8-bit quantization, fall back to 4-bit quantization
    - For maximum throughput, process the audio codec and enhancement operations in parallel with model inference
    - Use PLMS sampling method for diffusion models to balance quality and speed

## Expected Benefits

1. **Broader Hardware Compatibility**: Run YuE on older GPUs without Flash Attention support
2. **Efficient Resource Utilization**: Better distribution of workload across heterogeneous GPUs
3. **Memory Efficiency**: Reduced VRAM requirements through quantization and chunking
4. **Configurable Performance**: Users can choose appropriate tradeoffs between quality and speed
5. **Improved Stability**: Graceful fallbacks and recovery from errors

## Implementation Timeline

1. **Phase 1 (Immediate)**: Basic compatibility options to make YuE run on older hardware
2. **Phase 2 (Short-term)**: Performance optimizations for better efficiency on limited hardware
3. **Phase 3 (Medium-term)**: Advanced optimizations for optimal performance across all configurations

## Future Considerations

1. **Dynamic Parameter Adjustment**: Runtime adjustment of parameters based on generation progress
2. **Progressive Quality Refinement**: Start with low-quality fast generation and incrementally improve
3. **Distributed Processing**: Extend to multi-machine setups for even larger scale processing
4. **Integration with Specialized Libraries**: Support for libraries like vLLM, TensorRT, or ONNX Runtime
