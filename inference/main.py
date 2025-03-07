import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import uuid
import argparse
import numpy as np
import torch
import torchaudio

from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig, GPTNeoXTokenizerFast
from transformers.trainer_pt_utils import is_sagemaker_mp_enabled
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator, StereoCodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from vocoder import build_codec_model
from post_process_audio import replace_low_freq_with_energy_matched, replace_low_freq_with_energy_matched_stereo

# Import from our modular components
from audio_utils import (
    load_audio_mono, load_audio_stereo, save_audio, save_audio_stereo,
    process_stereo_mix, mix_tracks
)
from codec_utils import (
    seed_everything, encode_audio, encode_audio_stereo,
    decode_audio, decode_stereo_audio, split_lyrics
)
from generation import (
    stage2_inference, stage2_inference_stereo, stage1_inference_stereo,
    stage1_inference, post_process_generated_audio, process_and_save_audio
)

from transformers import BitsAndBytesConfig

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
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
    
    # Other
    parser.add_argument("--use_stereo", action="store_true", help="Whether to use stereo processing for audio generation.")
    
    # Add audio enhancement options
    parser.add_argument(
        "--enhance-audio",
        action="store_true",
        help="Apply advanced audio mixing enhancements to generated output"
    )
    parser.add_argument(
        "--stereo-width",
        type=float,
        default=1.3,
        help="Stereo width enhancement (1.0 = normal, >1.0 = wider)"
    )
    parser.add_argument(
        "--apply-compression",
        action="store_true",
        help="Apply multiband compression to enhance dynamics"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set the seed for reproducibility
    seed_everything(args.seed)
    
    # Initialize device based on command line parameter
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
    
    # Configure memory settings based on mode
    if args.low_memory_mode:
        print("Running in low memory mode - performance may be slower but more stable")
        # Increase memory garbage collection
        torch.cuda.empty_cache()
        
        # In low memory mode, explicitly set low quantization config
        if torch.cuda.is_available() and args.device == "cuda":
            print("Setting up 8-bit quantization with reduced precision in low memory mode")
            
            # Reduce max_new_tokens in low memory mode
            if args.max_new_tokens > 1000:
                print(f"Reducing max_new_tokens from {args.max_new_tokens} to 1000 in low memory mode")
                args.max_new_tokens = 1000
    
    # Choose precision based on parameters and compatibility
    if args.no_bfloat16:
        dtype = torch.float16
        compute_dtype = torch.float16
        print("Using float16 precision instead of bfloat16 for better compatibility")
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        compute_dtype = torch.float16 if args.device == "cuda" else torch.float32
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    stage1_output_dir = os.path.join(args.output_dir, "stage1")
    stage2_output_dir = os.path.join(args.output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Set random seed
    random_id = str(uuid.uuid4())[:8]
    
    # Load tokenizer
    mmtokenizer = _MMSentencePieceTokenizer(args.tokenizer_model)
    
    # Initialize codec models
    codec_model, _ = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    codec_model = codec_model.to(device)
    codec_model.eval()
    
    # Initialize codec tool
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", n_quantizer=1)
    
    # Initialize diffusion models if enabled
    diffusion_hybrid_model = None
    diffusion_postproc_model = None
    diffusion_conditional_model = None
    
    if args.use_diffusion:
        if not args.diffusion_model_path:
            print("Warning: Diffusion enabled but no model path provided. Using fallbacks.")
        
        if args.use_hybrid_architecture:
            from diffusion_models import HybridArchitectureDiffusion
            from hybrid_diffusion import TransformerDiffusionHybrid
            print("Initializing hybrid architecture diffusion model...")
            diffusion_hybrid_model = HybridArchitectureDiffusion(
                model_path=args.diffusion_model_path,
                device=device
            )
            
        if args.use_diffusion_postprocessing:
            from diffusion_models import PostProcessingDiffusion
            print("Initializing post-processing diffusion model...")
            diffusion_postproc_model = PostProcessingDiffusion(
                model_path=args.diffusion_model_path,
                device=device
            )
            
        if args.use_conditional_diffusion:
            from diffusion_models import ConditionalDiffusion
            from hybrid_diffusion import ConditionalDiffusionGenerator
            print("Initializing conditional diffusion model...")
            diffusion_conditional_model = ConditionalDiffusion(
                model_path=args.diffusion_model_path,
                device=device
            )
    
    # For stereo processing, initialize the stereo codec tool
    if args.use_stereo:
        stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
    
    # Initialize and load Stage 1 model
    print("Loading Stage 1 model...")
    
    # Low memory config
    if args.low_memory_mode and args.device == "cuda" and torch.cuda.is_available():
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
        stage1_model = AutoModelForCausalLM.from_pretrained(
            args.stage1_model,
            device_map="auto" if args.device == "cuda" else None,
            torch_dtype=dtype,
            quantization_config=config,
            attn_implementation="eager",  # Use eager implementation instead of flash attention
            low_cpu_mem_usage=True,
        )
    elif args.device == "cuda" and torch.cuda.is_available():
        # Standard loading on GPU
        stage1_model = AutoModelForCausalLM.from_pretrained(
            args.stage1_model,
            device_map="auto",
            torch_dtype=dtype,
            load_in_8bit=True,
            attn_implementation="eager",  # Use eager implementation instead of flash attention
            low_cpu_mem_usage=True,
        )
    else:
        # CPU loading - use minimal memory
        print("Loading model on CPU - this may take longer but will be more stable")
        try:
            # First attempt with standard loading
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map=None,
                torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
                attn_implementation="eager",  # Use eager implementation instead of flash attention
                low_cpu_mem_usage=True,
            )
            # Move model to CPU explicitly
            stage1_model = stage1_model.to("cpu")
        except Exception as cpu_load_error:
            print(f"Standard CPU model loading failed: {cpu_load_error}")
            print("Trying fallback CPU loading method...")
            
            # Fallback method with conservative settings
            stage1_model = AutoModelForCausalLM.from_pretrained(
                args.stage1_model,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                offload_folder="offload_folder",
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
    
    stage1_model.eval()
    
    # Prepare prompt texts
    with open(args.genre_txt, 'r', encoding='utf-8') as f:
        genres = f.read().strip()
    
    lyrics = ""
    if args.lyrics_txt and os.path.exists(args.lyrics_txt):
        with open(args.lyrics_txt, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()
        
        # Process lyrics into segments if needed
        lyric_segments = split_lyrics(lyrics)
    
    # Create prompt text
    prompt_text = f"<|genres|>{genres}<|title|>{args.title}<|lyrics|>{lyrics}<|instruction|>{args.instruction}"
    
    # Handle audio prompt if specified
    audio_prompt = None
    if args.use_audio_prompt and os.path.exists(args.audio_prompt_path):
        if args.use_stereo:
            audio_prompt = load_audio_stereo(args.audio_prompt_path)
            # Encode the stereo audio prompt
            prompt_tokens_left, prompt_tokens_right = encode_audio_stereo(codec_model, audio_prompt, device)
        else:
            audio_prompt = load_audio_mono(args.audio_prompt_path)
            # Encode the mono audio prompt
            prompt_tokens = encode_audio(codec_model, audio_prompt, device)
    
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
            args=args
        )
    else:
        # Use regular mono generation
        stage1_output_set = stage1_inference(
            model=stage1_model, 
            prompt_text=prompt_text, 
            codectool=codectool, 
            mmtokenizer=mmtokenizer, 
            device=device, 
            args=args
        )
    
    # Unload Stage 1 model to save memory if not disabled
    if not args.disable_offload_model:
        print("Offloading Stage 1 model to save memory...")
        del stage1_model
        torch.cuda.empty_cache()
    
    # Load Stage 2 model
    print("Loading Stage 2 model...")
    stage2_model = AutoModelForCausalLM.from_pretrained(
        args.stage2_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        load_in_8bit=True,
        use_cache=True,
    )
    
    # Stage 2 inference
    if args.use_stereo:
        # Stage 2 inference with stereo
        stage2_output_set = stage2_inference_stereo(
            model=stage2_model,
            stage1_output_set=stage1_output_set,
            stage2_output_dir=stage2_output_dir,
            codectool=codectool_stage2,
            mmtokenizer=mmtokenizer,
            batch_size=args.stage2_batch_size,
            apply_enhancements=True,
            diffusion_postproc_model=diffusion_postproc_model,
            diffusion_steps=args.diffusion_steps,
            diffusion_sampling_method=args.diffusion_sampling_method
        )
    else:
        # Stage 2 inference
        stage2_output_set = stage2_inference(
            model=stage2_model,
            stage1_output_set=stage1_output_set,
            stage2_output_dir=stage2_output_dir,
            codectool=codectool_stage2,
            mmtokenizer=mmtokenizer,
            device=device,
            batch_size=args.stage2_batch_size,
            apply_enhancements=True,
            diffusion_postproc_model=diffusion_postproc_model,
            diffusion_steps=args.diffusion_steps,
            diffusion_sampling_method=args.diffusion_sampling_method
        )
    
    print(stage2_output_set)
    print('Stage 2 DONE.\n')
    
    # Reconstruct tracks
    recons_output_dir = os.path.join(args.output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, 'mix')
    os.makedirs(recons_mix_dir, exist_ok=True)
    
    tracks = []
    for npy in stage2_output_set:
        codec_result = np.load(npy)
        decodec_rlt = []
        
        with torch.no_grad():
            if args.use_stereo:
                # For stereo, decode left and right channels
                left_codes = codec_result[:, 0, :]
                right_codes = codec_result[:, 1, :]
                decoded_waveform = decode_stereo_audio(codec_model, left_codes, right_codes, device)
            else:
                # For mono
                decoded_waveform = decode_audio(codec_model, codec_result, device)
        
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        
        save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
        tracks.append(save_path)
        
        # Save using regular or stereo function based on configuration
        if args.use_stereo:
            save_audio_stereo(decodec_rlt, save_path, 16000)
        else:
            save_audio(decodec_rlt, save_path, 16000)
    
    # Mix tracks if we have multiple stems (vocals and instrumentals)
    if len(tracks) >= 2 and ('_vtrack' in tracks[0] or '_itrack' in tracks[0]):
        # Process stereo mix for higher quality when we have separate stems
        vocal_track = next((t for t in tracks if '_vtrack' in t), None)
        inst_track = next((t for t in tracks if '_itrack' in t), None)
        
        if vocal_track and inst_track:
            mix_path = os.path.join(recons_mix_dir, f"mixed_{random_id}.mp3")
            process_stereo_mix(vocal_track, inst_track, mix_path)
            tracks.append(mix_path)
    else:
        # Standard mixing for other track combinations
        mixed_tracks = mix_tracks(tracks, recons_mix_dir)
    
    # Vocoder to upsample audios
    vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
    vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
    vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    
    # Additional post-processing if needed
    if args.enhance_audio:
        print("Applying audio enhancements to the final output...")
        
        # Apply both audio enhancements and frequency blending
        for output_path in stage2_output_set:
            audio_path = output_path.replace('.npy', '.wav')
            if os.path.exists(audio_path):
                # First apply enhancements
                audio, sr = torchaudio.load(audio_path)
                enhanced_audio = post_process_generated_audio(
                    audio, 
                    sr=sr,
                    apply_enhancements=True
                )
                
                # Save enhanced audio
                enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                process_and_save_audio(enhanced_audio, enhanced_path, sr, apply_enhancements=False)
                
                # Then apply frequency blending for improved low frequencies
                for mixed_track in mixed_tracks:
                    hi_res_path = os.path.join(vocoder_output_dir, os.path.basename(mixed_track))
                    final_output_path = os.path.join(args.output_dir, os.path.basename(mixed_track).replace('.mp3', '_final.mp3'))
                    
                    # Use appropriate function based on stereo configuration
                    if args.use_stereo:
                        replace_low_freq_with_energy_matched_stereo(
                            a_file=mixed_track,  # 16kHz
                            b_file=hi_res_path,  # 44.1kHz 
                            c_file=final_output_path,
                            cutoff_freq=5500.0
                        )
                    else:
                        replace_low_freq_with_energy_matched(
                            a_file=mixed_track,  # 16kHz
                            b_file=hi_res_path,  # 44.1kHz
                            c_file=final_output_path,
                            cutoff_freq=5500.0
                        )
                    
                    print(f"Final enhanced output saved to {final_output_path}")

    print("Generation complete!")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main() 