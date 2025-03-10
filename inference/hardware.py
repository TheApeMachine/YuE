import os
import platform
import torch
import gc
import psutil
from transformers import BitsAndBytesConfig

def is_wsl():
    """Check if running under Windows Subsystem for Linux"""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if "microsoft" in f.read().lower():
                return True
    return False

def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == "windows"

def get_system_memory_info():
    """Get current system memory information"""
    mem_info = psutil.virtual_memory()
    return {
        'total': mem_info.total / (1024**3),  # GB
        'available': mem_info.available / (1024**3),  # GB
        'percent_used': mem_info.percent
    }

def _check_gpu_capabilities(gpu_idx):
    """Check capabilities of a specific GPU"""
    props = torch.cuda.get_device_properties(gpu_idx)
    name = props.name
    memory_mb = props.total_memory // (1024**2)  # Convert to MB
    compute_capability = f"{props.major}.{props.minor}"
    supports_flash_attn = props.major >= 8
    supports_bfloat16 = props.major >= 8
    
    # More detailed capability detection
    tensorcore_support = props.major >= 7  # Volta and newer
    mma_support = props.major >= 8  # Ampere and newer (MMA = matrix multiply accumulate)
    
    return {
        'name': name,
        'memory_mb': memory_mb,
        'compute_capability': compute_capability,
        'supports_flash_attn': supports_flash_attn,
        'supports_bfloat16': supports_bfloat16,
        'supports_tensor_cores': tensorcore_support,
        'supports_mma': mma_support
    }

def get_gpu_memory_usage():
    """Get current GPU memory usage for all GPUs"""
    if not torch.cuda.is_available():
        return []
    
    result = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
        
        # Get total memory for percentage calculations
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        
        result.append({
            'device': i,
            'allocated_gb': allocated,
            'reserved_gb': reserved, 
            'total_gb': total,
            'percent_used': (allocated / total) * 100
        })
    
    return result

def _get_recommended_settings(capabilities):
    """Generate recommended settings based on detected hardware capabilities"""
    recommended = {}
    
    # Get system memory information for informed decisions
    system_memory = get_system_memory_info()
    low_system_memory = system_memory['available'] < 8  # Less than 8GB available
    
    # CPU-only recommendations
    if not capabilities['has_cuda']:
        recommended = {
            'device': 'cpu',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'minimal',
            'diffusion_optimization': 'faster',
            'disable_flash_attention': True,
            'enable_checkpointing': True,
            'auto_batch_size': True,
            'chunk_size': 300,  # Process in small chunks
            'safe_mode': True
        }
    # Special handling for Quadro-era GPUs (Maxwell/Pascal, compute capability 5.x-6.x)
    elif any(5.0 <= float(cc.split('.')[0]) <= 6.0 for cc in capabilities['gpu_compute_capabilities']):
        print("Detected Quadro-era GPU (Maxwell/Pascal architecture)")
        print("Applying special optimizations for older architectures")
        recommended = {
            'device': 'cuda',
            'disable_flash_attention': True,  # Must disable flash attention
            'enable_torch_compile': False,    # Can be unstable on older architectures
            'quantization': '4bit_nf4',       # More aggressive memory optimization
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'memory_efficient',
            'auto_batch_size': True,
            'chunk_size': 600,               # Enable chunking for better stability
            'max_new_tokens': 400,           # Reduce token count for safety
            'gradient_checkpointing': True,  # Save memory with gradient checkpointing
            'repetition_penalty': 1.05       # More conservative repetition penalty
        }
    # GPUs without Flash Attention support
    elif not capabilities['supports_flash_attn']:
        recommended = {
            'device': 'cuda',
            'disable_flash_attention': True,
            'enable_torch_compile': True,
            'torch_compile_mode': 'reduce-overhead',
            'quantization': '8bit' if max(capabilities['gpu_memory']) >= 12000 else '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'faster',
            'auto_batch_size': True,
            'chunk_size': 600 if max(capabilities['gpu_memory']) < 10000 else None,
            'gradient_checkpointing': True
        }
    # Low memory GPUs (<8GB VRAM)
    elif min(capabilities['gpu_memory']) < 8000:
        recommended = {
            'device': 'cuda',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'memory_efficient',
            'auto_batch_size': True,
            'chunk_size': 600,
            'enable_checkpointing': True,
            'max_new_tokens': 500,  # Limit token generation
            'gradient_checkpointing': True
        }
    # Multi-GPU setup
    elif capabilities['gpu_count'] > 1:
        # If we have multiple GPUs, recommend distributing workload
        recommended = {
            'device': 'cuda',
            'quantization': 'none' if max(capabilities['gpu_memory']) >= 24000 else '8bit',
            'audio_processing_level': 'full',
            'diffusion_optimization': 'none',
            'model_split_strategy': 'model_type',
            'transformer_device': 'cuda:0,1' if capabilities['gpu_count'] >= 2 else 'auto',
            'diffusion_device': 'cuda:1' if capabilities['gpu_count'] >= 2 else 'auto',
            'codec_device': 'cuda:0' if capabilities['gpu_count'] >= 2 else 'auto',
            'enable_parallel_processing': True
        }
    # Single high-end GPU
    else:
        recommended = {
            'device': 'cuda',
            'quantization': 'none' if max(capabilities['gpu_memory']) >= 24000 else '8bit',
            'audio_processing_level': 'full',
            'diffusion_optimization': 'none',
            'auto_batch_size': True
        }
    
    # Add common recommendations based on other detected capabilities
    if capabilities['supports_bfloat16']:
        recommended['no_bfloat16'] = False
    else:
        recommended['no_bfloat16'] = True
    
    # For Windows/WSL environments, recommend safer settings
    if is_wsl() or is_windows():
        recommended.update({
            'enable_checkpointing': True,
            'audio_processing_level': 'standard',
            'safe_mode': True if low_system_memory else False
        })
    
    # If we're on low system memory, adjust settings further
    if low_system_memory:
        recommended.update({
            'max_new_tokens': min(recommended.get('max_new_tokens', 3000), 500),
            'run_n_segments': min(recommended.get('run_n_segments', 2), 1),
            'audio_processing_level': 'minimal',
            'low_memory_mode': True
        })
        
    return recommended

def detect_hardware_capabilities():
    """Detect hardware capabilities and recommend settings"""
    capabilities = {
        'has_cuda': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [],
        'gpu_memory': [],
        'gpu_compute_capabilities': [],
        'supports_flash_attn': False,
        'supports_bfloat16': False,
        'supports_tensor_cores': False,
        'cpu_count': os.cpu_count(),
        'system_memory': get_system_memory_info(),
        'recommended_settings': {}
    }
    
    # Check GPU capabilities if available
    if capabilities['has_cuda']:
        for i in range(capabilities['gpu_count']):
            gpu_info = _check_gpu_capabilities(i)
            capabilities['gpu_names'].append(gpu_info['name'])
            capabilities['gpu_memory'].append(gpu_info['memory_mb'])
            capabilities['gpu_compute_capabilities'].append(gpu_info['compute_capability'])
            
            # If any GPU supports flash attention, mark as supported
            if gpu_info['supports_flash_attn']:
                capabilities['supports_flash_attn'] = True
                
            # If any GPU supports bfloat16, mark as supported
            if gpu_info['supports_bfloat16']:
                capabilities['supports_bfloat16'] = True
                
            # If any GPU supports tensor cores, mark as supported
            if gpu_info['supports_tensor_cores']:
                capabilities['supports_tensor_cores'] = True
    
    # Get recommended settings based on detected capabilities
    capabilities['recommended_settings'] = _get_recommended_settings(capabilities)
    
    return capabilities

def configure_settings_from_hardware(args):
    """Apply hardware-specific configuration settings"""
    capabilities = detect_hardware_capabilities()
    print(f"Detected hardware: {len(capabilities['gpu_names'])} GPUs")
    
    # Print system memory info
    sys_mem = capabilities['system_memory']
    print(f"System memory: {sys_mem['total']:.1f}GB total, {sys_mem['available']:.1f}GB available ({sys_mem['percent_used']}% used)")
    
    for i, (name, memory, cc) in enumerate(zip(
        capabilities['gpu_names'], 
        capabilities['gpu_memory'],
        capabilities['gpu_compute_capabilities']
    )):
        print(f"  GPU {i}: {name} with {memory} MB VRAM (Compute Capability {cc})")
    
    # Run diagnostics if requested
    if getattr(args, 'run_diagnostics', False):
        run_system_diagnostics()
    
    print("\nRecommended settings for your hardware:")
    for setting, value in capabilities['recommended_settings'].items():
        print(f"  --{setting}={value}")
        
        # Apply the recommended settings to args
        if hasattr(args, setting):
            setattr(args, setting, value)
            print(f"  Applied: {setting}={value}")
    
    print("\nContinuing with auto-configured settings...")
    return args

def run_system_diagnostics():
    """Run comprehensive system diagnostics to verify GPU setup"""
    import torch
    import subprocess
    import pkg_resources
    
    print("\n--- RUNNING SYSTEM DIAGNOSTICS ---")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("⚠️ CUDA is not available. This suggests driver issues or incompatible PyTorch build.")
        return
    
    # Check PyTorch CUDA version
    try:
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
    except AttributeError:
        print("⚠️ Could not determine PyTorch CUDA version")
    
    # Get NVIDIA driver version
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
        driver_line = [line for line in nvidia_smi.split('\n') if 'Driver Version' in line][0]
        driver_version = driver_line.split('Driver Version:')[1].strip().split(' ')[0]
        print(f"NVIDIA driver version: {driver_version}")
        
        # Check for known problematic driver versions
        if cuda_version.startswith('11.') and not driver_version.startswith('4'):
            print("⚠️ Potential driver/CUDA version mismatch. CUDA 11.x works best with 470.x drivers for older GPUs")
    except (subprocess.SubprocessError, IndexError, AttributeError) as e:
        print(f"⚠️ Could not determine NVIDIA driver version: {e}")
    
    # Check for limited VRAM
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem < 6 * (1024**3):  # Less than 6GB
            print(f"⚠️ Limited VRAM detected: {total_mem / (1024**3):.1f}GB - This may cause segmentation faults")
    except RuntimeError as e:
        print(f"⚠️ Could not check GPU memory: {e}")
    
    # Test basic CUDA operations
    try:
        # Clear existing allocations
        torch.cuda.empty_cache()
        
        # Try a basic tensor operation
        print("Testing basic CUDA tensor operations...")
        x = torch.ones(1000, 1000, device='cuda')
        y = x + x
        del y
        
        print("Testing matrix multiplication...")
        a = torch.randn(2000, 2000, device='cuda')
        b = torch.randn(2000, 2000, device='cuda')
        c = torch.matmul(a, b)
        del a, b, c
        
        print("Basic CUDA operations successful ✓")
    except Exception as e:
        print(f"⚠️ CUDA operation test failed: {e}")
        print("This suggests driver/hardware issues that may cause segmentation faults.")
    
    # Check library conflicts
    try:
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        critical_packages = {
            'torch': installed.get('torch', 'Not installed'),
            'transformers': installed.get('transformers', 'Not installed'),
            'bitsandbytes': installed.get('bitsandbytes', 'Not installed'),
            'accelerate': installed.get('accelerate', 'Not installed')
        }
        
        print("\nCritical package versions:")
        for pkg, ver in critical_packages.items():
            print(f"  {pkg}: {ver}")
            
        # Look for known conflicts
        if 'bitsandbytes' in installed:
            bnb_version = installed['bitsandbytes']
            if bnb_version < '0.39.0' and cuda_version.startswith('11.8'):
                print("⚠️ Older bitsandbytes versions (<0.39.0) have issues with CUDA 11.8")
    except (ImportError, pkg_resources.DistributionNotFound) as e:
        print(f"Could not check library dependencies: {e}")
    
    print("--- DIAGNOSTICS COMPLETE ---\n")

def apply_safe_mode_settings(args):
    """Apply conservative settings for safe mode"""
    print("Running in safe mode with conservative settings")
    # Override settings with safe defaults
    args.max_new_tokens = min(args.max_new_tokens, 500)
    args.run_n_segments = min(args.run_n_segments, 1)
    args.stage2_batch_size = 1
    args.low_memory_mode = True
    args.force_cpu = True if is_wsl() or is_windows() else args.force_cpu
    args.quantization = "4bit_nf4" if args.quantization == "none" else args.quantization
    args.audio_processing_level = "minimal"
    args.disable_flash_attention = True
    return args

def initialize_device(args):
    """Initialize the appropriate device based on settings"""
    if args.force_cpu:
        device = torch.device("cpu")
        args.device = "cpu"  # Override device setting
        print("Forcing CPU-only operation for maximum stability")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_idx}")
        print(f"Using device: {device}")
        
        # Print current memory usage
        mem_info = torch.cuda.mem_get_info(device)
        free_memory = mem_info[0] / (1024**3)  # GB
        total_memory = mem_info[1] / (1024**3)  # GB
        print(f"  Memory: {free_memory:.2f} GB free / {total_memory:.2f} GB total")
        
        # Test device with small operations if diagnostics are enabled
        if getattr(args, 'test_device', False):
            try:
                # Quick tensor test
                test_tensor = torch.zeros(100, 100, device=device)
                test_tensor = test_tensor + 1
                print(f"  Device test successful on {device}")
                del test_tensor
            except Exception as e:
                print(f"⚠️ Device test failed: {e}")
                print("  This may indicate driver/hardware issues causing segmentation faults")
                print("  Consider using --force_cpu if issues persist")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device

def configure_memory_settings(args):
    """Configure memory settings based on low-memory mode"""
    if args.low_memory_mode:
        print("Running in low memory mode - performance may be slower but more stable")
        # Increase memory garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # In low memory mode, explicitly set low quantization config
        if torch.cuda.is_available() and args.device == "cuda":
            print("Setting up quantization with reduced precision in low memory mode")
            
            # Ensure quantization is set in low memory mode
            if args.quantization == "none":
                args.quantization = "8bit"
                print("Enabling 8-bit quantization for low memory mode")
            elif args.quantization == "8bit" and torch.cuda.get_device_properties(0).total_memory < 6 * (1024**3):
                # For very constrained GPUs (less than 6GB), force 4-bit
                args.quantization = "4bit_nf4" 
                print("Enabling 4-bit NF4 quantization for very limited memory mode")
            
            # Reduce max_new_tokens in low memory mode
            if args.max_new_tokens > 1000:
                max_tokens = min(1000, args.max_new_tokens)
                print(f"Reducing max_new_tokens from {args.max_new_tokens} to {max_tokens} in low memory mode")
                args.max_new_tokens = max_tokens
                
            # Activate additional memory-saving measures
            args.gradient_checkpointing = True
            
            # Monitor memory usage after configuration
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_info = torch.cuda.mem_get_info(i)
                    free_memory_gb = memory_info[0] / (1024**3)
                    total_memory_gb = memory_info[1] / (1024**3)
                    print(f"GPU {i} memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")

def prepare_model_dtype(args):
    """Determine model data type based on settings"""
    if args.no_bfloat16:
        dtype = torch.float16
        compute_dtype = torch.float16
        print("Using float16 precision instead of bfloat16 for better compatibility")
    else:
        # Check for hardware support of bfloat16
        if torch.cuda.is_available() and args.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            supports_bf16 = props.major >= 8  # Ampere and newer support bf16
            
            if supports_bf16:
                dtype = torch.bfloat16
                compute_dtype = torch.bfloat16
                print("Using bfloat16 precision for better numerical stability")
            else:
                dtype = torch.float16
                compute_dtype = torch.float16
                print("GPU doesn't support bfloat16, using float16 instead")
        else:
            dtype = torch.float32
            compute_dtype = torch.float32
            print("Using float32 precision for CPU inference")
    
    return dtype, compute_dtype

def prepare_quantization_config(args, compute_dtype):
    """Prepare quantization configuration based on settings"""
    if args.quantization == "none":
        print("Not using quantization - using full precision model")
        return None
    elif args.quantization == "8bit":
        print("Using 8-bit quantization (memory efficient with good quality)")
        return BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        print("Using 4-bit quantization (very memory efficient)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
    elif args.quantization == "4bit_nf4":
        print("Using 4-bit NF4 quantization (extremely memory efficient)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )

def monitor_and_clean_memory(threshold=0.9, force_gc=False):
    """Monitor memory usage and clean if above threshold
    
    Args:
        threshold: fraction of GPU memory that triggers cleaning (0.0-1.0)
        force_gc: if True, always run garbage collection regardless of usage
        
    Returns:
        Dictionary with memory statistics 
    """
    stats = {'cleaned': False}
    
    # Always check system memory
    sys_mem = get_system_memory_info()
    stats['system'] = sys_mem
    
    # If we have CUDA, check GPU memory
    if torch.cuda.is_available():
        gpu_memory = get_gpu_memory_usage()
        stats['gpu'] = gpu_memory
        
        # Check if any GPU is above threshold
        needs_cleanup = force_gc or any(gpu['percent_used'] > threshold * 100 for gpu in gpu_memory)
        
        if needs_cleanup:
            # Perform memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            stats['cleaned'] = True
    
    return stats
