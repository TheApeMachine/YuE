import os
import platform
import torch
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

def _check_gpu_capabilities(gpu_idx):
    """Check capabilities of a specific GPU"""
    props = torch.cuda.get_device_properties(gpu_idx)
    name = props.name
    memory_mb = props.total_memory // (1024**2)  # Convert to MB
    supports_flash_attn = props.major >= 8
    supports_bfloat16 = props.major >= 8
    
    return {
        'name': name,
        'memory_mb': memory_mb,
        'supports_flash_attn': supports_flash_attn,
        'supports_bfloat16': supports_bfloat16
    }

def _get_recommended_settings(capabilities):
    """Generate recommended settings based on detected hardware capabilities"""
    recommended = {}
    
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
    # GPUs without Flash Attention support
    elif not capabilities['supports_flash_attn']:
        recommended = {
            'device': 'cuda',
            'disable_flash_attention': True,
            'enable_torch_compile': True,
            'quantization': '8bit' if max(capabilities['gpu_memory']) >= 12000 else '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'faster',
            'auto_batch_size': True,
            'chunk_size': 600 if max(capabilities['gpu_memory']) < 10000 else None
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
            'enable_checkpointing': True
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
            'safe_mode': True
        })
        
    return recommended

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
    
    # Check GPU capabilities if available
    if capabilities['has_cuda']:
        for i in range(capabilities['gpu_count']):
            gpu_info = _check_gpu_capabilities(i)
            capabilities['gpu_names'].append(gpu_info['name'])
            capabilities['gpu_memory'].append(gpu_info['memory_mb'])
            
            # If any GPU supports flash attention, mark as supported
            if gpu_info['supports_flash_attn']:
                capabilities['supports_flash_attn'] = True
                
            # If any GPU supports bfloat16, mark as supported
            if gpu_info['supports_bfloat16']:
                capabilities['supports_bfloat16'] = True
    
    # Get recommended settings based on detected capabilities
    capabilities['recommended_settings'] = _get_recommended_settings(capabilities)
    
    return capabilities

def configure_settings_from_hardware(args):
    """Apply hardware-specific configuration settings"""
    capabilities = detect_hardware_capabilities()
    print(f"Detected hardware: {len(capabilities['gpu_names'])} GPUs")
    for i, (name, memory) in enumerate(zip(capabilities['gpu_names'], capabilities['gpu_memory'])):
        print(f"  GPU {i}: {name} with {memory} MB VRAM")
    
    print("\nRecommended settings for your hardware:")
    for setting, value in capabilities['recommended_settings'].items():
        print(f"  --{setting}={value}")
        
        # Apply the recommended settings to args
        if hasattr(args, setting):
            setattr(args, setting, value)
            print(f"  Applied: {setting}={value}")
    
    print("\nContinuing with auto-configured settings...")
    return args

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
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device

def configure_memory_settings(args):
    """Configure memory settings based on low-memory mode"""
    if args.low_memory_mode:
        print("Running in low memory mode - performance may be slower but more stable")
        # Increase memory garbage collection
        torch.cuda.empty_cache()
        
        # In low memory mode, explicitly set low quantization config
        if torch.cuda.is_available() and args.device == "cuda":
            print("Setting up quantization with reduced precision in low memory mode")
            
            # Ensure quantization is set in low memory mode
            if args.quantization == "none":
                args.quantization = "8bit"
                print("Enabling 8-bit quantization for low memory mode")
            
            # Reduce max_new_tokens in low memory mode
            if args.max_new_tokens > 1000:
                print(f"Reducing max_new_tokens from {args.max_new_tokens} to 1000 in low memory mode")
                args.max_new_tokens = 1000

def prepare_model_dtype(args):
    """Determine model data type based on settings"""
    if args.no_bfloat16:
        dtype = torch.float16
        compute_dtype = torch.float16
        print("Using float16 precision instead of bfloat16 for better compatibility")
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        compute_dtype = torch.float16 if args.device == "cuda" else torch.float32
    return dtype, compute_dtype

def prepare_quantization_config(args, compute_dtype):
    """Prepare quantization configuration based on settings"""
    if args.quantization == "none":
        return None
    elif args.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
    elif args.quantization == "4bit_nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
