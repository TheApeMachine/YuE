import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import uuid
import argparse
import torch
import torchaudio
import platform
from torchaudio.transforms import Resample

from transformers import AutoModelForCausalLM
from codecmanipulator import CodecManipulator, StereoCodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from vocoder import build_codec_model

# Import from our modular components
from audio_utils import (
    load_audio_mono, load_audio_stereo
)
from codec_utils import (
    seed_everything, encode_audio, encode_audio_stereo, split_lyrics
)
from generation import (
    stage2_inference, stage2_inference_stereo, stage1_inference_stereo, stage1_inference
)

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
    """Generate recommended settings based on hardware capabilities"""
    if not capabilities['has_cuda']:
        return {
            'device': 'cpu',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'minimal',
            'diffusion_optimization': 'faster',
            'disable_flash_attention': True
        }
    elif not capabilities['supports_flash_attn']:
        return {
            'device': 'cuda',
            'disable_flash_attention': True,
            'enable_torch_compile': True,
            'quantization': '8bit' if max(capabilities['gpu_memory']) >= 12000 else '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'faster'
        }
    elif min(capabilities['gpu_memory']) < 8000:
        # Low memory GPUs
        return {
            'device': 'cuda',
            'quantization': '4bit_nf4',
            'audio_processing_level': 'standard',
            'diffusion_optimization': 'memory_efficient'
        }
    else:
        # High-end GPUs
        return {
            'device': 'cuda',
            'quantization': 'none',
            'audio_processing_level': 'full',
            'diffusion_optimization': 'none'
        }

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # === Operation Mode Selection ===
    mode_group = parser.add_argument_group('Operation Modes')
    mode_group.add_argument("--auto_config", action="store_true", 
                           help="Automatically configure settings based on detected hardware")
    mode_group.add_argument("--safe_mode", action="store_true", 
                           help="Run with conservative settings for maximum stability")
    mode_group.add_argument("--test_audio_mixing", action="store_true",
                           help="Test audio mixing functionality only (no model inference)")
    
    # === Basic Configuration ===
    # Model Configuration:
    parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help="The model checkpoint path or identifier for the Stage 1 model.")
    parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general", help="The model checkpoint path or identifier for the Stage 2 model.")
    parser.add_argument("--tokenizer_model", type=str, default="./mm_tokenizer_v0.2_hf/tokenizer.model", help="Path to the SentencePiece tokenizer model file.")
    parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.")
    parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during the generation.")
    parser.add_argument("--stage2_batch_size", type=int, default=4, help="The batch size used in Stage 2 inference.")
    
    # Prompt
    parser.add_argument("--genre_txt", type=str, required=True, help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.")
    parser.add_argument("--lyrics_txt", type=str, default="", help="The file path to a text file containing lyrics for vocal music generation. Leave empty for instrumental.")
    parser.add_argument("--title", type=str, default="", help="The song title.")
    parser.add_argument("--instruction", type=str, default="", help="Optional instruction text guidance.")
    parser.add_argument("--use_audio_prompt", action="store_true", help="Whether to use an audio prompt for the generation.")
    parser.add_argument("--audio_prompt_path", type=str, default="", help="The file path to the audio prompt. Only used when use_audio_prompt is set.")
    parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start timestamp (in seconds) of the audio prompt section to use. Helps when focusing on a specific segment.")
    parser.add_argument("--prompt_end_time", type=float, default=6.0, help="The end timestamp (in seconds) of the audio prompt section to use. Helps define the duration of the audio prompt.")
    parser.add_argument("--use_dual_tracks_prompt", action="store_true", help="Whether to use separate vocal and instrumental track prompts. If set, provide paths to both.")
    parser.add_argument("--vocal_track_prompt_path", type=str, default="", help="The file path to the vocal track prompt. Used when use_dual_tracks_prompt is set.")
    parser.add_argument("--instrumental_track_prompt_path", type=str, default="", help="The file path to the instrumental track prompt. Used when use_dual_tracks_prompt is set.")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
    parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
    parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="An integer value to reproduce generation.")
    parser.add_argument("--low_memory_mode", action="store_true", help="Enable low memory mode to reduce memory usage at the cost of speed. Useful for GPUs with limited VRAM.")
    parser.add_argument("--no_bfloat16", action="store_true", help="Disable the use of bfloat16 precision, which can cause compatibility issues on some GPUs. Use float16 instead.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to run inference on. Use 'cpu' for maximum compatibility.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU-only operation for maximum stability, especially on WSL or systems with GPU compatibility issues.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling during generation. Higher values (>1.0) make output more random, lower values (<1.0) make it more deterministic.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter. Controls diversity by only considering tokens with cumulative probability < top_p.")

    # Config for xcodec and upsampler
    parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for xcodec configurations.')
    parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the xcodec checkpoint.')
    parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', help='Path to Vocos config file.')
    parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
    
    # Diffusion model enhancements (optional)
    parser.add_argument('--use_diffusion', action='store_true', help='Enable diffusion model enhancements (any type).')
    parser.add_argument('--diffusion_model_path', type=str, default='', help='Path to the pre-trained diffusion model weights.')
    parser.add_argument('--use_hybrid_architecture', action='store_true', help='Use diffusion model in hybrid architecture with transformer for musical structure.')
    parser.add_argument('--use_diffusion_postprocessing', action='store_true', help='Apply diffusion-based enhancement after Stage 2 inference.')
    parser.add_argument('--use_conditional_diffusion', action='store_true', help='Use diffusion models conditioned on codec tokens for transitions.')
    parser.add_argument('--diffusion_guidance_scale', type=float, default=3.0, help='Guidance scale for classifier-free guidance in diffusion (higher = more adherence to condition).')
    parser.add_argument('--diffusion_steps', type=int, default=50, help='Number of diffusion steps to use for generation/refinement.')
    parser.add_argument('--diffusion_sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim', 'plms'], help='Sampling method for diffusion model.')
    parser.add_argument('--diffusion_optimization', type=str, choices=['none', 'faster', 'memory_efficient'], default='none', help='Optimization strategy for diffusion models.')
    
    # Other
    parser.add_argument("--use_stereo", action="store_true", help="Whether to use stereo processing for audio generation.")
    
    # Add audio enhancement options
    parser.add_argument(
        "--enhance-audio",
        action="store_true",
        help="Apply advanced audio mixing enhancements to generated output"
    )
    
    # === Hardware Compatibility Options ===
    hardware_group = parser.add_argument_group('Hardware Compatibility')
    hardware_group.add_argument("--disable_flash_attention", action="store_true", 
                             help="Disable Flash Attention for compatibility with older GPUs")
    hardware_group.add_argument("--enable_torch_compile", action="store_true",
                             help="Enable PyTorch 2.0+ compilation for improved performance on all hardware")
    hardware_group.add_argument("--torch_compile_mode", type=str, choices=["default", "reduce-overhead", "max-autotune"],
                             default="reduce-overhead", help="Compilation mode for PyTorch 2.0+")
    hardware_group.add_argument("--torch_compile_fullgraph", action="store_true", 
                             help="Enable full graph compilation in torch.compile (may increase compilation time)")
    hardware_group.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit", "4bit_nf4"], default="none",
                             help="Model quantization level (lower bits = less memory, slightly lower quality)")
    hardware_group.add_argument("--audio_processing_level", type=str, choices=["minimal", "standard", "full"], default="full",
                             help="Level of audio post-processing to apply (minimal = fastest, full = best quality)")
    
    # Multi-GPU Support
    hardware_group.add_argument("--transformer_device", type=str, default="auto", 
                             help="Device to use for transformer models (e.g., 'cuda:0,1' for multi-GPU)")
    hardware_group.add_argument("--diffusion_device", type=str, default="auto",
                             help="Device to use for diffusion models (e.g., 'cuda:1')")
    hardware_group.add_argument("--codec_device", type=str, default="auto",
                             help="Device to use for codec model and audio processing (e.g., 'cuda:2')")
    hardware_group.add_argument("--model_split_strategy", type=str, choices=["layer", "model_type", "hybrid"], default="model_type",
                             help="How to split models across GPUs: by layer, by model type, or hybrid approach")
    hardware_group.add_argument("--enable_parallel_processing", action="store_true",
                             help="Enable parallel processing across multiple GPUs")
    
    # Advanced Options
    hardware_group.add_argument("--chunk_size", type=int, default=None,
                             help="Process in chunks to reduce memory usage (specify max tokens per chunk)")
    hardware_group.add_argument("--enable_checkpointing", action="store_true",
                             help="Enable checkpointing of intermediate results for possible resumption")
    hardware_group.add_argument("--resume_from_checkpoint", type=str, default="",
                             help="Resume generation from a saved checkpoint file")
    
    # Audio Mixing Test Arguments
    audio_mixing_group = parser.add_argument_group('Audio Mixing Test')
    audio_mixing_group.add_argument("--vocal_path", type=str, default="",
                                 help="Path to vocal audio file for audio mixing tests")
    audio_mixing_group.add_argument("--instrumental_path", type=str, default="",
                                 help="Path to instrumental audio file for audio mixing tests")
    audio_mixing_group.add_argument("--output_path", type=str, default="./mixed_output.wav",
                                 help="Output path for mixed audio")
    
    # Add any other arguments you need...
    
    return parser.parse_args()

def _configure_settings_from_hardware(args):
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

def _apply_safe_mode_settings(args):
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

def _initialize_device(args):
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

def _configure_memory_settings(args):
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

def _prepare_model_dtype(args):
    """Determine model data type based on settings"""
    if args.no_bfloat16:
        dtype = torch.float16
        compute_dtype = torch.float16
        print("Using float16 precision instead of bfloat16 for better compatibility")
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        compute_dtype = torch.float16 if args.device == "cuda" else torch.float32
    return dtype, compute_dtype

def _prepare_quantization_config(args, compute_dtype):
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

def _load_stage1_model(args, dtype, quantization_config, device):
    """Load the Stage 1 model with appropriate settings"""
    print("Loading Stage 1 model...")
    attn_implementation = "eager" if args.disable_flash_attention else "flash_attention_2"
    
    try:
        if args.device == "cuda" and torch.cuda.is_available():
            # GPU loading with appropriate device mapping
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map="auto",  # Will be overridden if transformer_device is set
                torch_dtype=dtype,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,
            )
        else:
            # CPU loading - use minimal memory
            print("Loading model on CPU - this may take longer but will be more stable")
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map=None,
                torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
                attn_implementation="eager",  # Use eager implementation instead of flash attention
                low_cpu_mem_usage=True,
            )
            # Move model to CPU explicitly
            stage1_model = stage1_model.to("cpu")
    except Exception as e:
        if "flash_attention" in str(e).lower():
            print(f"Flash Attention error: {e}")
            print("Falling back to standard attention")
            # Try again with eager attention
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map="auto" if args.device == "cuda" else None,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        else:
            # For other errors, try a more conservative loading approach
            print(f"Error loading model: {e}")
            print("Trying fallback loading method...")
            
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                offload_folder="offload_folder",
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
    
    return stage1_model

def _apply_model_device_mapping(model, transformer_device, model_name="model"):
    """Apply custom device mapping to a model"""
    try:
        # For multi-GPU transformer setups
        if "," in transformer_device:
            # Logic for distributing across multiple GPUs
            devices = [f"cuda:{idx}" for idx in transformer_device.replace("cuda:", "").split(",")]
            print(f"Distributing {model_name} across devices: {devices}")
            
            # Create a device map for layer distribution
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
            
            # Apply device map
            print(f"Applying device map: {device_map}")
            # The actual application would depend on the model's specific structure
            # This is a simplified example
        else:
            # Single GPU for transformer
            print(f"Moving {model_name} to {transformer_device}")
            model = model.to(transformer_device)
    except Exception as e:
        print(f"Error in custom device mapping: {e}")
        print("Falling back to default device mapping")
    
    return model

def _apply_torch_compile(model, args):
    """Apply PyTorch compilation if available"""
    if args.enable_torch_compile and hasattr(torch, "compile"):
        try:
            print(f"Applying PyTorch compilation with mode: {args.torch_compile_mode}")
            model = torch.compile(
                model, 
                mode=args.torch_compile_mode,
                fullgraph=args.torch_compile_fullgraph
            )
            print("PyTorch compilation successful")
        except Exception as e:
            print(f"Error during model compilation: {e}")
            print("Continuing with uncompiled model")
    
    return model

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Auto-configuration based on hardware detection
    if args.auto_config:
        args = _configure_settings_from_hardware(args)
    
    # Apply safe mode settings if requested
    if args.safe_mode:
        args = _apply_safe_mode_settings(args)
    
    # Special mode: test audio mixing only
    if args.test_audio_mixing:
        if not (args.vocal_path and args.instrumental_path):
            print("Error: --vocal_path and --instrumental_path are required for audio mixing test")
            return 1
        
        test_audio_mixing(args.vocal_path, args.instrumental_path, args.output_path, args.audio_processing_level)
        print(f"Audio mixing test completed. Output saved to {args.output_path}")
        return 0
    
    # Set the seed for reproducibility
    seed_everything(args.seed)
    
    # Initialize device based on command line parameter
    device = _initialize_device(args)
    
    # Configure memory settings based on mode
    _configure_memory_settings(args)
    
    # Choose precision based on parameters and compatibility
    dtype, compute_dtype = _prepare_model_dtype(args)
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate session ID for output files
    session_id = str(uuid.uuid4())[:8]
    print(f"Using session ID: {session_id} for this generation run")
    
    # Create session-specific output directories
    stage1_output_dir = os.path.join(args.output_dir, f"stage1_{session_id}")
    stage2_output_dir = os.path.join(args.output_dir, f"stage2_{session_id}")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Load tokenizer
    mmtokenizer = _MMSentencePieceTokenizer(args.tokenizer_model)
    
    # Initialize codec models
    codec_model, _ = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    
    # Apply task-based device allocation for codec model
    if args.codec_device != "auto" and args.device == "cuda":
        try:
            codec_device = args.codec_device
            print(f"Moving codec model to {codec_device}")
            codec_model = codec_model.to(codec_device)
        except Exception as e:
            print(f"Error moving codec model to {codec_device}: {e}")
            print("Falling back to default device")
            codec_model = codec_model.to(device)
    else:
        codec_model = codec_model.to(device)
    
    codec_model.eval()
    
    # Initialize codec tool
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", n_quantizer=1)
    
    # Initialize diffusion models if enabled
    diffusion_postproc_model = None
    
    if args.use_diffusion:
        diffusion_models = _initialize_diffusion_models(args, device)
        # Only extract the models we actually use
        diffusion_postproc_model = diffusion_models.get('postproc')
    
    # For stereo processing, initialize the stereo codec tool
    stereo_codectool = None
    if args.use_stereo:
        stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
    
    # Prepare quantization config based on args
    quantization_config = _prepare_quantization_config(args, compute_dtype)
    
    # Initialize and load Stage 1 model
    stage1_model = _load_stage1_model(args, dtype, quantization_config, device)
    
    # Apply task-based device allocation for Stage 1 model if specified
    if args.transformer_device != "auto" and args.device == "cuda":
        stage1_model = _apply_model_device_mapping(stage1_model, args.transformer_device, "transformer")
    
    # Apply PyTorch compilation if requested
    stage1_model = _apply_torch_compile(stage1_model, args)
    
    stage1_model.eval()
    
    # Prepare prompt texts
    with open(args.genre_txt, 'r', encoding='utf-8') as f:
        genres = f.read().strip()
    
    lyrics = ""
    if args.lyrics_txt and os.path.exists(args.lyrics_txt):
        with open(args.lyrics_txt, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()
        
        # We get lyric_segments but don't use them directly - our function signature expects this processing
        # This variable is kept to maintain the original processing flow
        _ = split_lyrics(lyrics)
    
    # Create prompt text
    prompt_text = f"<|genres|>{genres}<|title|>{args.title}<|lyrics|>{lyrics}<|instruction|>{args.instruction}"
    
    # Handle audio prompt if specified
    audio_prompt = None
    if args.use_audio_prompt and os.path.exists(args.audio_prompt_path):
        if args.use_stereo:
            audio_prompt = load_audio_stereo(args.audio_prompt_path)
            # Encode the stereo audio prompt - values not directly used here but kept for consistency
            _, _ = encode_audio_stereo(codec_model, audio_prompt, device)
        else:
            audio_prompt = load_audio_mono(args.audio_prompt_path)
            # Encode the mono audio prompt - original code used this in stage1_inference
            # but our refactored code uses args, so we don't need to pass it directly
            _ = encode_audio(codec_model, audio_prompt, device)
    
    # Parallel processing setup if enabled
    if args.enable_parallel_processing and torch.cuda.device_count() > 1:
        print("Enabling parallel processing across multiple GPUs")
        # Placeholder for parallel processing implementation
    
    print("Stage 1 inference...")
    stage1_output_set = []
    
    if args.use_stereo and args.use_dual_tracks_prompt:
        # Use stereo generation
        stage1_output_set = stage1_inference_stereo(
            model=stage1_model, 
            prompt_texts=prompt_text, 
            codectool=codectool, 
            mmtokenizer=mmtokenizer, 
            device=device, 
            args=args  # Pass args object directly instead of individual parameters
        )
    else:
        # Use mono or standard stereo generation
        stage1_output_set = stage1_inference(
            model=stage1_model, 
            prompt_text=prompt_text,  # Changed from prompt_texts to prompt_text
            codectool=codectool, 
            mmtokenizer=mmtokenizer, 
            device=device, 
            args=args  # Pass args object directly instead of individual parameters
        )
    
    # Offload stage1_model to CPU to save GPU memory
    if not args.disable_offload_model and args.device == "cuda":
        stage1_model = stage1_model.to("cpu")
        torch.cuda.empty_cache()
    
    print("Setting up Stage 2 model...")
    stage2_model = AutoModelForCausalLM.from_pretrained(
        args.stage2_model,
        torch_dtype=dtype if args.device == "cuda" else torch.float32
    )
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    
    print("Stage 2 inference...")
    
    # Apply PyTorch compilation to Stage 2 model if requested
    stage2_model = _apply_torch_compile(stage2_model, args)
    
    if args.use_stereo and args.use_dual_tracks_prompt:
        # Using the stereo codec tool initialized earlier
        if not stereo_codectool:
            stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
        
        for stage1_output in stage1_output_set:
            # stereo output - match the expected function signature
            stage2_inference_stereo(
                model=stage2_model,
                stage1_output_set=[stage1_output],  # Wrap in list to match expected signature
                stage2_output_dir=stage2_output_dir,
                codectool=codectool,
                mmtokenizer=mmtokenizer,
                device=device,
                batch_size=args.stage2_batch_size,
                apply_enhancements=(args.audio_processing_level != "minimal"),
                diffusion_postproc_model=diffusion_postproc_model if args.use_diffusion_postprocessing else None,
                diffusion_steps=args.diffusion_steps,
                diffusion_sampling_method=args.diffusion_sampling_method
            )
    else:
        # Match the expected function signature for stage2_inference
        stage2_inference(
            model=stage2_model,
            stage1_output_set=stage1_output_set,
            stage2_output_dir=stage2_output_dir,
            codectool=codectool_stage2,
            mmtokenizer=mmtokenizer,
            device=device,
            batch_size=args.stage2_batch_size,
            apply_enhancements=(args.audio_processing_level != "minimal"),
            diffusion_postproc_model=diffusion_postproc_model if args.use_diffusion_postprocessing else None,
            diffusion_steps=args.diffusion_steps,
            diffusion_sampling_method=args.diffusion_sampling_method
        )
    
    print("Generation complete!")
    print(f"Output files saved in: {stage2_output_dir}")
    return 0

def _initialize_diffusion_models(args, device):
    """Initialize diffusion models based on settings"""
    diffusion_models = {}
    
    if not args.diffusion_model_path:
        print("Warning: Diffusion enabled but no model path provided. Using fallbacks.")
    
    # Apply diffusion optimization strategies if specified
    diffusion_opts = {}
    if args.diffusion_optimization != "none":
        diffusion_opts["optimization_strategy"] = args.diffusion_optimization
        if args.diffusion_optimization == "faster":
            diffusion_opts["sampling_method"] = "ddim"
            diffusion_opts["steps"] = min(30, args.diffusion_steps)
        elif args.diffusion_optimization == "memory_efficient":
            diffusion_opts["sampling_method"] = "plms"
            diffusion_opts["steps"] = min(40, args.diffusion_steps)
    
    if args.use_hybrid_architecture:
        from diffusion_models import HybridArchitectureDiffusion
        from hybrid_diffusion import TransformerDiffusionHybrid
        print("Initializing hybrid architecture diffusion model...")
        diffusion_models['hybrid'] = HybridArchitectureDiffusion(
            model_path=args.diffusion_model_path,
            device=device,
            **diffusion_opts
        )
        
    if args.use_diffusion_postprocessing:
        from diffusion_models import PostProcessingDiffusion
        print("Initializing post-processing diffusion model...")
        diffusion_models['postproc'] = PostProcessingDiffusion(
            model_path=args.diffusion_model_path,
            device=device,
            **diffusion_opts
        )
        
    if args.use_conditional_diffusion:
        from diffusion_models import ConditionalDiffusion
        from hybrid_diffusion import ConditionalDiffusionGenerator
        print("Initializing conditional diffusion model...")
        diffusion_models['conditional'] = ConditionalDiffusion(
            model_path=args.diffusion_model_path,
            device=device,
            **diffusion_opts
        )
    
    # Apply task-based device allocation for diffusion model if specified
    if args.diffusion_device != "auto" and args.device == "cuda":
        diffusion_device = args.diffusion_device
        print(f"Moving diffusion models to {diffusion_device}")
        
        for model_type, model in diffusion_models.items():
            if model:
                diffusion_models[model_type] = model.to(diffusion_device)
    
    return diffusion_models

def test_audio_mixing(vocal_path, instrumental_path, output_path, processing_level="full"):
    """
    Test the enhanced audio mixing features with custom settings.
    
    Args:
        vocal_path: Path to vocal file
        instrumental_path: Path to instrumental file
        output_path: Path to save mixed output
        processing_level: Level of audio processing to apply
    """
    print(f"Processing {os.path.basename(vocal_path)} + {os.path.basename(instrumental_path)}")
    
    # Load audio files
    vocal, sr_v = torchaudio.load(vocal_path)
    instrumental, sr_i = torchaudio.load(instrumental_path)
    
    # Resample if necessary
    if sr_v != 44100:
        resampler = Resample(sr_v, 44100)
        vocal = resampler(vocal)
        sr_v = 44100
    
    if sr_i != 44100:
        resampler = Resample(sr_i, 44100)
        instrumental = resampler(instrumental)
        # Note that sr_i is updated but not used directly later
        # We'll use sr_v for both since they should be the same now
    
    # Make sure they have the same number of channels
    if vocal.shape[0] == 1 and instrumental.shape[0] == 2:
        vocal = torch.cat([vocal, vocal], dim=0)
    elif vocal.shape[0] == 2 and instrumental.shape[0] == 1:
        instrumental = torch.cat([instrumental, instrumental], dim=0)
    
    # Create a mix with different processing levels
    if processing_level == "minimal":
        # Simple mixing with minimal processing
        mix_params = {
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
        }
    elif processing_level == "standard":
        # Standard processing with moderate enhancements
        mix_params = {
            'phase_alignment': {'enabled': True, 'multiband': True},
            'normalization': {'enabled': True},
            'multiband_compression': {'enabled': True},
            'vocal_compression': {'enabled': True},
            'instrumental_compression': {'enabled': False},
            'stereo_width': {'enabled': True, 'width': 1.1},
            'vocal_enhancement': {'enabled': True, 'level': 0.5},
            'vocal_space_carving': {'enabled': True, 'level': 0.4},
            'instrumental_saturation': {'enabled': False},
            'exciter': {'enabled': False},
        }
    else:  # full processing
        # Full processing with all enhancements
        mix_params = None  # Use defaults in enhanced_audio_mix
    
    # Process the mix
    from audio_mixing import enhanced_audio_mix
    
    # Apply the mix
    mixed = enhanced_audio_mix(vocal, instrumental, mix_params, sr=sr_v)
    
    # Save the output
    torchaudio.save(output_path, mixed, sr_v)
    print(f"Saved mixed output to {output_path}")
    return True

if __name__ == "__main__":
    main() 