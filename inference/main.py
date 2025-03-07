import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import uuid
import argparse
import numpy as np
import torch
import torchaudio

from transformers import AutoModelForCausalLM
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

    # Config for xcodec and upsampler
    parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for xcodec configurations.')
    parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the xcodec checkpoint.')
    parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', help='Path to Vocos config file.')
    parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
    
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
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    stage1_output_dir = os.path.join(args.output_dir, "stage1")
    stage2_output_dir = os.path.join(args.output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Set random seed
    seed_everything(args.seed)
    random_id = str(uuid.uuid4())[:8]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    mmtokenizer = _MMSentencePieceTokenizer(args.tokenizer_model)
    
    # Initialize codec models
    codec_model, _ = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    codec_model = codec_model.to(device)
    codec_model.eval()
    
    # Initialize codec tool
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", n_quantizer=1)
    
    # For stereo processing, initialize the stereo codec tool
    if args.use_stereo:
        stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
    
    # Load Stage 1 model
    print("Loading Stage 1 model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.stage1_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)
    
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
        stage1_output_set = stage1_inference_stereo(model, prompt_text, codectool, mmtokenizer, device, args)
    else:
        # Use regular mono generation
        stage1_output_set = stage1_inference(model, prompt_text, codectool, mmtokenizer, device, args)
    
    # Offload model to save memory
    if not args.disable_offload_model:
        model.cpu()
        del model
        torch.cuda.empty_cache()
    
    # Stage 2 inference
    print("Stage 2 inference...")
    model_stage2 = AutoModelForCausalLM.from_pretrained(
        args.stage2_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model_stage2.to(device)
    model_stage2.eval()
    
    if torch.__version__ >= "2.0.0":
        model_stage2 = torch.compile(model_stage2)
    
    # Run Stage 2 inference
    if args.use_stereo:
        stage2_result = stage2_inference_stereo(
            model_stage2, stage1_output_set, stage2_output_dir, 
            codectool_stage2, mmtokenizer, batch_size=args.stage2_batch_size
        )
    else:
        stage2_result = stage2_inference(
            model_stage2, stage1_output_set, stage2_output_dir, 
            codectool_stage2, batch_size=args.stage2_batch_size,
            apply_enhancements=args.enhance_audio
        )
    
    print(stage2_result)
    print('Stage 2 DONE.\n')
    
    # Reconstruct tracks
    recons_output_dir = os.path.join(args.output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, 'mix')
    os.makedirs(recons_mix_dir, exist_ok=True)
    
    tracks = []
    for npy in stage2_result:
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
        for output_path in stage2_result:
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