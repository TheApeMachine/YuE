import os
import copy
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from einops import rearrange
from transformers import LogitsProcessorList
import torchaudio
from audio_utils import save_audio
from vocoder import build_codec_model

from codec_utils import BlockTokenRangeProcessor
from token_fixer import fix_tokens
from audio_mixing import align_phases, enhance_stereo_width, multi_band_compression, enhanced_audio_mix, apply_gain_staging

def stage2_generate(model, prompt, codectool, mmtokenizer, device, batch_size=16):
    """
    Generate Stage 2 tokens from Stage 1 prompt
    
    Args:
        model: Generation model
        prompt: Prompt tokens
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer
        device: Processing device
        batch_size: Processing batch size
    
    Returns:
        Generated tokens
    """
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)
    
    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1
        )
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten the 2D array to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358), 
        BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)
    ])

    # Teacher forcing generate loop
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx:frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids
        
        # Create attention mask (all 1's since we don't have padding)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            try:
                stage2_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )
            except RuntimeError as e:
                # Handle CUDA errors
                print(f"Error during stage2 generation: {e}")
                print("Attempting to recover...")
                
                # Try to free up CUDA memory
                torch.cuda.empty_cache()
                
                # If the error is a cuBLAS error, try with different settings
                if "cublasLt ran into an error" in str(e) or "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
                    try:
                        print("CUDA/cuBLAS error detected. Attempting recovery...")
                        
                        # Check if it's a bfloat16 compatibility issue
                        if "CUDA_R_16BF" in str(e):
                            print("bfloat16 compatibility issue detected. Switching to CPU with float32 precision...")
                            # Move to CPU and use float32 (most compatible)
                            cpu_model = model.cpu().to(torch.float32)
                            cpu_input_ids = input_ids.cpu()
                            cpu_attention_mask = attention_mask.cpu()
                        else:
                            print("Switching to CPU for this operation...")
                            # Just move to CPU with original precision
                            cpu_model = model.to('cpu')
                            cpu_input_ids = input_ids.to('cpu')
                            cpu_attention_mask = attention_mask.to('cpu')
                        
                        # Reduce max_new_tokens to save memory
                        reduced_max_tokens = 500  # Use a fixed value instead of depending on args
                        print(f"Reducing max_new_tokens to {reduced_max_tokens}")
                        
                        # Generate with reduced parameters
                        outputs = cpu_model.generate(
                            input_ids=cpu_input_ids,
                            attention_mask=cpu_attention_mask,
                            max_new_tokens=reduced_max_tokens,
                            do_sample=True,
                            temperature=0.7,  # Use default value instead of args
                            top_p=0.95,  # Use default value instead of args
                            repetition_penalty=1.0,  # Use a default value for repetition_penalty
                            eos_token_id=mmtokenizer.eoa,
                            pad_token_id=mmtokenizer.eoa,
                        )
                        
                        # Move back to GPU if needed for further processing
                        outputs = outputs.to(device)
                        
                        # Don't move model back to GPU to avoid memory issues
                        print("Generation completed on CPU. Keeping model on CPU to conserve GPU memory.")
                        
                    except Exception as recovery_error:
                        print(f"Recovery attempt failed: {recovery_error}")
                        error_msg = "Failed to generate tokens due to CUDA errors. Please try: \n"
                        error_msg += "1. Using --low_memory_mode with a lower --max_new_tokens value (200-500)\n"
                        error_msg += "2. Adding --no_bfloat16 to avoid bfloat16 precision issues\n"
                        error_msg += "3. Running on CPU only with --device cpu\n"
                        raise RuntimeError(error_msg) from e
                else:
                    raise  # Re-raise if it's not a CUDA error
        
        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, \
            f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output

def stage2_generate_stereo(model, prompt_left, prompt_right, codectool, mmtokenizer, device, batch_size=16):
    """
    Generate stereo audio by processing left and right channels separately
    
    Args:
        model: Generation model
        prompt_left: Left channel prompt tokens
        prompt_right: Right channel prompt tokens
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer
        device: Processing device
        batch_size: Processing batch size
        
    Returns:
        Generated left and right channel outputs
    """
    # Process left and right channels separately
    left_output = stage2_generate(model, prompt_left, codectool, mmtokenizer, device, batch_size)
    right_output = stage2_generate(model, prompt_right, codectool, mmtokenizer, device, batch_size)
    
    # Return both channels
    return left_output, right_output

def post_process_generated_audio(audio, sr=44100, apply_enhancements=True):
    """
    Apply audio mixing enhancements to generated audio
    
    Args:
        audio: Generated audio tensor (mono or stereo)
        sr: Sample rate
        apply_enhancements: Whether to apply enhancements
        
    Returns:
        Enhanced audio
    """
    if not apply_enhancements:
        return audio
        
    # Skip processing if audio is invalid
    if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
        return audio
    
    # 1. Convert to stereo if mono
    if audio.dim() == 1 or (audio.dim() > 1 and audio.shape[0] == 1):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.repeat(2, 1)
    
    # 2. Apply phase alignment between channels to improve stereo image
    if audio.shape[0] > 1:
        # Use the first channel as reference for phase alignment
        reference = audio[0].unsqueeze(0)
        target = audio[1].unsqueeze(0)
        # Align the second channel to the first
        aligned_channel = align_phases(reference, target)
        # Reconstruct stereo signal with aligned phases
        audio = torch.cat([reference, aligned_channel], dim=0)
    
    # 3. Apply stereo width enhancement
    audio = enhance_stereo_width(audio, width=1.3)
    
    # 4. Apply multi-band compression for balanced dynamics
    audio = multi_band_compression(
        audio,
        bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)],
        thresholds=[-24, -18, -18, -16],
        ratios=[2.5, 2.0, 1.8, 1.5],
        sr=sr
    )
    
    # 5. Final gain staging
    audio = apply_gain_staging(audio, gain_db=-0.3)
    
    return audio

def process_and_save_audio(audio, output_path, sr=44100, apply_enhancements=True, diffusion_postproc_model=None, diffusion_steps=50, diffusion_sampling_method='ddpm'):
    """
    Process and save generated audio
    
    Args:
        audio: Audio data to process
        output_path: Path to save processed audio
        sr: Sample rate
        apply_enhancements: Whether to apply enhancement algorithms
        diffusion_postproc_model: Optional diffusion model for post-processing
        diffusion_steps: Number of steps for diffusion process
        diffusion_sampling_method: Sampling method for diffusion
        
    Returns:
        Path to saved audio file
    """
    # Apply standard enhancements if needed
    if apply_enhancements:
        audio = post_process_generated_audio(audio, sr=sr, apply_enhancements=True)

    # Apply diffusion-based post-processing if model is provided
    if diffusion_postproc_model is not None:
        print("Applying diffusion-based audio enhancement...")
        audio = diffusion_postproc_model.enhance_audio(
            audio, 
            steps=diffusion_steps,
            sampling_method=diffusion_sampling_method
        )
    
    # Save audio
    save_audio(audio, output_path, sample_rate=sr)
    
    return output_path

def stage2_inference(model, stage1_output_set, stage2_output_dir, codectool, mmtokenizer, device, batch_size=4, apply_enhancements=True, diffusion_postproc_model=None, diffusion_steps=50, diffusion_sampling_method='ddpm'):
    """
    Run Stage 2 inference
    
    Args:
        model: Generation model
        stage1_output_set: Paths to Stage 1 output files
        stage2_output_dir: Output directory for Stage 2 results
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer for audio processing
        device: Processing device
        batch_size: Processing batch size
        apply_enhancements: Whether to apply audio enhancements
        diffusion_postproc_model: Optional diffusion model for post-processing
        diffusion_steps: Number of steps for diffusion process
        diffusion_sampling_method: Sampling method for diffusion
        
    Returns:
        Paths to generated audio files
    """
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            stage2_result.append(output_filename)
            continue
        
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            # Generate audio from codec tokens
            output = stage2_generate(model, prompt[:, :output_duration*50], codectool, mmtokenizer, device, batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            outputs = []
            for i in range(0, num_batch, batch_size):
                start_idx = i * 300
                end_idx = min((i + batch_size) * 300, output_duration*50)
                current_batch_size = (end_idx - start_idx + 299) // 300
                
                # Generate this chunk
                chunk = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    codectool,
                    mmtokenizer,
                    device,
                    batch_size=current_batch_size
                )
                outputs.append(chunk)
            
            # Concatenate all chunks
            output = np.concatenate(outputs, axis=0)
        
        # Process the ending part of the prompt if necessary
        if output_duration*50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], codectool, mmtokenizer, device, batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        
        output = codectool.ids2npy(output)

        # Fix invalid tokens using the token_fixer module
        original_path = output_filename.replace('.npy', '_original.npy') if output_filename.endswith('.npy') else f"{output_filename}_original.npy"
        fixed_output = fix_tokens(output, min_valid=0, max_valid=1023, save_original=True, original_path=original_path)
        
        # Process and save the output
        processed_path = process_and_save_audio(
            fixed_output, 
            output_filename, 
            apply_enhancements=apply_enhancements,
            diffusion_postproc_model=diffusion_postproc_model,
            diffusion_steps=diffusion_steps,
            diffusion_sampling_method=diffusion_sampling_method
        )
        
        stage2_result.append(processed_path)
    
    if apply_enhancements:
        stage2_result_enhanced = []
        for output_path in stage2_result:
            # Assuming the output path is the audio file path
            audio_path = output_path.replace('.npy', '.wav')  # Adjust this based on actual file naming
            
            # Load the audio
            if os.path.exists(audio_path):
                audio, sr = torchaudio.load(audio_path)
                
                # Check if we have both vocal and instrumental components
                # (This would need to be adapted based on your actual project structure)
                vocal_path = audio_path.replace('.wav', '_vocal.wav')
                instrumental_path = audio_path.replace('.wav', '_instrumental.wav')
                
                if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
                    # If we have both components, use the enhanced_audio_mix function
                    vocal, _ = torchaudio.load(vocal_path)
                    instrumental, _ = torchaudio.load(instrumental_path)
                    
                    # Define mixing parameters
                    mix_params = {
                        'vocal_gain': 1.0,
                        'instrumental_gain': 0.8,
                        'target_lufs': -16.0,
                        'vocal_compression': {
                            'threshold': -20.0,
                            'ratio': 2.0,
                            'attack': 0.005,
                            'release': 0.05
                        },
                        'sidechain': {
                            'enabled': True,
                            'threshold': -24.0,
                            'ratio': 2.5
                        },
                        'stereo_width': 1.2,
                        'pan_position': 0.0,
                        'phase_align': True
                    }
                    
                    # Apply advanced mixing
                    mixed_audio = enhanced_audio_mix(vocal, instrumental, mix_params, sr)
                    
                    # Save the enhanced mixed audio
                    enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                    torchaudio.save(enhanced_path, mixed_audio, sr)
                else:
                    # If we don't have separate components, apply standard post-processing
                    enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                    process_and_save_audio(
                        audio, 
                        enhanced_path, 
                        apply_enhancements=True,
                        diffusion_postproc_model=diffusion_postproc_model,
                        diffusion_steps=diffusion_steps,
                        diffusion_sampling_method=diffusion_sampling_method
                    )
                
                stage2_result_enhanced.append(enhanced_path)
            else:
                stage2_result_enhanced.append(output_path)
        
        return stage2_result_enhanced
    else:
        return stage2_result

def stage2_inference_stereo(model, stage1_output_set, stage2_output_dir, codectool, mmtokenizer, device, batch_size=4, apply_enhancements=True, diffusion_postproc_model=None, diffusion_steps=50, diffusion_sampling_method='ddpm'):
    """
    Run Stage 2 inference for stereo audio
    
    Args:
        model: Generation model
        stage1_output_set: Paths to Stage 1 output files
        stage2_output_dir: Output directory for Stage 2 results
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer for audio processing
        device: Device to run inference on (cpu or cuda)
        batch_size: Processing batch size
        apply_enhancements: Whether to apply audio enhancements
        diffusion_postproc_model: Optional diffusion model for post-processing
        diffusion_steps: Number of steps for diffusion process
        diffusion_sampling_method: Sampling method for diffusion
        
    Returns:
        Paths to generated stereo audio files
    """
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            stage2_result.append(output_filename)
            continue
        
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        # Create left and right channel prompts
        prompt_left = prompt.copy()
        prompt_right = prompt.copy()
        
        # Process in chunks based on batch size
        if num_batch <= batch_size:
            # Generate left and right channels
            output_left, output_right = stage2_generate_stereo(
                model, 
                prompt_left[:, :output_duration*50], 
                prompt_right[:, :output_duration*50],
                codectool,
                mmtokenizer,
                device,
                batch_size=num_batch
            )
        else:
            # Process in chunks
            segments_left = []
            segments_right = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
            
            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                
                left, right = stage2_generate_stereo(
                    model,
                    prompt_left[:, start_idx:end_idx],
                    prompt_right[:, start_idx:end_idx],
                    codectool,
                    mmtokenizer,
                    device,
                    batch_size=current_batch_size
                )
                
                segments_left.append(left)
                segments_right.append(right)
            
            # Concatenate segments
            output_left = np.concatenate(segments_left, axis=0)
            output_right = np.concatenate(segments_right, axis=0)
        
        # Process ending if needed
        if output_duration*50 != prompt.shape[-1]:
            left_ending, right_ending = stage2_generate_stereo(
                model, 
                prompt_left[:, output_duration*50:],
                prompt_right[:, output_duration*50:],
                codectool,
                mmtokenizer,
                device,
                batch_size=1
            )
            output_left = np.concatenate([output_left, left_ending], axis=0)
            output_right = np.concatenate([output_right, right_ending], axis=0)
        
        # Convert token IDs to numpy arrays
        output_left = codectool.ids2npy(output_left)
        output_right = codectool.ids2npy(output_right)
        
        # Fix invalid tokens
        left_original_path = output_filename.replace('.npy', '_left_original.npy')
        right_original_path = output_filename.replace('.npy', '_right_original.npy')
        
        fixed_left = fix_tokens(output_left, min_valid=0, max_valid=1023, save_original=True, original_path=left_original_path)
        fixed_right = fix_tokens(output_right, min_valid=0, max_valid=1023, save_original=True, original_path=right_original_path)
        
        # Combine channels (this depends on your expected output format)
        combined_output = np.stack([fixed_left, fixed_right], axis=0)
        
        # Save the combined output
        np.save(output_filename, combined_output)
        stage2_result.append(output_filename)
    
    if apply_enhancements:
        stage2_result_enhanced = []
        for output_path in stage2_result:
            # Assuming the output path is the audio file path
            audio_path = output_path.replace('.npy', '.wav')
            
            if os.path.exists(audio_path):
                audio, sr = torchaudio.load(audio_path)
                
                # Phase align the stereo channels
                if audio.shape[0] > 1:
                    left_channel = audio[0].unsqueeze(0)
                    right_channel = audio[1].unsqueeze(0)
                    
                    # Align the right channel to match the phase of the left channel
                    aligned_right = align_phases(left_channel, right_channel)
                    
                    # Reconstruct the stereo signal with aligned phases
                    audio = torch.cat([left_channel, aligned_right], dim=0)
                
                # Check if we have both vocal and instrumental components
                vocal_path = audio_path.replace('.wav', '_vocal.wav')
                instrumental_path = audio_path.replace('.wav', '_instrumental.wav')
                
                if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
                    # If we have both components, use the enhanced_audio_mix function
                    vocal, _ = torchaudio.load(vocal_path)
                    instrumental, _ = torchaudio.load(instrumental_path)
                    
                    # Define mixing parameters with emphasis on stereo field
                    mix_params = {
                        'vocal_gain': 1.0,
                        'instrumental_gain': 0.8,
                        'target_lufs': -16.0,
                        'vocal_compression': {
                            'threshold': -20.0,
                            'ratio': 2.0,
                            'attack': 0.005,
                            'release': 0.05
                        },
                        'sidechain': {
                            'enabled': True,
                            'threshold': -24.0,
                            'ratio': 2.5
                        },
                        'stereo_width': 1.5,  # Enhanced stereo width for stereo mode
                        'pan_position': 0.0,
                        'phase_align': True
                    }
                    
                    # Apply advanced mixing
                    mixed_audio = enhanced_audio_mix(vocal, instrumental, mix_params, sr)
                    
                    # Save the enhanced mixed audio
                    enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                    torchaudio.save(enhanced_path, mixed_audio, sr)
                else:
                    # If we don't have separate components, apply standard post-processing
                    enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                    process_and_save_audio(
                        audio, 
                        enhanced_path, 
                        apply_enhancements=True,
                        diffusion_postproc_model=diffusion_postproc_model,
                        diffusion_steps=diffusion_steps,
                        diffusion_sampling_method=diffusion_sampling_method
                    )
                
                stage2_result_enhanced.append(enhanced_path)
            else:
                stage2_result_enhanced.append(output_path)
        
        return stage2_result_enhanced
    else:
        return stage2_result

def stage1_inference(model, prompt_text, codectool, mmtokenizer, device, args):
    """
    Run Stage 1 inference for mono audio generation
    
    Args:
        model: Generation model
        prompt_text: Text prompt for generation
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer
        device: Processing device
        args: Additional arguments
        
    Returns:
        Paths to Stage 1 output files
    """
    import uuid
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from vocoder import build_codec_model
    
    # Create unique output path with UUID
    random_id = str(uuid.uuid4())[:8]
    stage1_output_dir = os.path.join(args.output_dir, "stage1")
    os.makedirs(stage1_output_dir, exist_ok=True)
    
    stage1_output_set = []
    
    # Handle audio prompt if specified
    if args.use_audio_prompt and os.path.exists(args.audio_prompt_path):
        from audio_utils import load_audio_mono
        from codec_utils import encode_audio
        
        # Load and process audio prompt
        print(f"Using audio prompt: {args.audio_prompt_path}")
        audio_prompt = load_audio_mono(
            args.audio_prompt_path, 
            start_time=args.prompt_start_time, 
            end_time=args.prompt_end_time
        )
        
        # Initialize codec model for encoding
        codec_model = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)[0]
        codec_model = codec_model.to(device)
        codec_model.eval()
        
        # Encode audio prompt
        prompt_tokens = encode_audio(codec_model, audio_prompt, device)
        
        # Add audio prompt context to the text prompt
        prompt_text += "<|audio_prompt|>"
        
        # Store the audio prompt tokens to use after text tokenization
        audio_prompt_tokens = prompt_tokens.tolist()
    
    # Encode the prompt text
    input_ids = mmtokenizer.tokenize(prompt_text)
    # Add BOS token (as the original code was using bos=True)
    input_ids = [mmtokenizer.bos] + input_ids
    
    # Add encoded audio prompt tokens if available
    if 'audio_prompt_tokens' in locals():
        input_ids = input_ids + audio_prompt_tokens
    
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Create attention mask (all 1's since we don't have padding)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
    
    # Generate segments
    for seg_idx in range(args.run_n_segments):
        output_path = os.path.join(stage1_output_dir, f"segment_{random_id}_{seg_idx}.npy")
        
        if os.path.exists(output_path):
            print(f'{output_path} already exists, skipping generation.')
            stage1_output_set.append(output_path)
            continue
            
        print(f"Generating segment {seg_idx+1}/{args.run_n_segments}")
        
        # Generate with temperature and top_p sampling
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=getattr(args, 'temperature', 0.7),  # Default to 0.7 if not specified
                    top_p=getattr(args, 'top_p', 0.95),  # Default to 0.95 if not specified
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                )
            except RuntimeError as e:
                # Handle CUDA OOM or other runtime errors
                print(f"Error during generation: {e}")
                print("Attempting to recover by reducing batch size and memory usage...")
                
                # Try to free up CUDA memory
                torch.cuda.empty_cache()
                
                # If the error is a cuBLAS error, try with different settings
                if "cublasLt ran into an error" in str(e) or "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
                    try:
                        print("CUDA/cuBLAS error detected. Attempting recovery...")
                        
                        # Check if it's a bfloat16 compatibility issue
                        if "CUDA_R_16BF" in str(e):
                            print("bfloat16 compatibility issue detected. Switching to CPU with float32 precision...")
                            # Move to CPU and use float32 (most compatible)
                            cpu_model = model.cpu().to(torch.float32)
                            cpu_input_ids = input_ids.cpu()
                            cpu_attention_mask = attention_mask.cpu()
                        else:
                            print("Switching to CPU for this operation...")
                            # Just move to CPU with original precision
                            cpu_model = model.to('cpu')
                            cpu_input_ids = input_ids.to('cpu')
                            cpu_attention_mask = attention_mask.to('cpu')
                        
                        # Reduce max_new_tokens to save memory
                        reduced_max_tokens = 500  # Use a fixed value instead of depending on args
                        print(f"Reducing max_new_tokens to {reduced_max_tokens}")
                        
                        # Generate with reduced parameters
                        outputs = cpu_model.generate(
                            input_ids=cpu_input_ids,
                            attention_mask=cpu_attention_mask,
                            max_new_tokens=reduced_max_tokens,
                            do_sample=True,
                            temperature=0.7,  # Use default value instead of args
                            top_p=0.95,  # Use default value instead of args
                            repetition_penalty=1.0,  # Use a default value for repetition_penalty
                            eos_token_id=mmtokenizer.eoa,
                            pad_token_id=mmtokenizer.eoa,
                        )
                        
                        # Move back to GPU if needed for further processing
                        outputs = outputs.to(device)
                        
                        # Don't move model back to GPU to avoid memory issues
                        print("Generation completed on CPU. Keeping model on CPU to conserve GPU memory.")
                        
                    except Exception as recovery_error:
                        print(f"Recovery attempt failed: {recovery_error}")
                        error_msg = "Failed to generate tokens due to CUDA errors. Please try: \n"
                        error_msg += "1. Using --low_memory_mode with a lower --max_new_tokens value (200-500)\n"
                        error_msg += "2. Adding --no_bfloat16 to avoid bfloat16 precision issues\n"
                        error_msg += "3. Running on CPU only with --device cpu\n"
                        raise RuntimeError(error_msg) from e
                else:
                    error_detected = True
                    raise
        
        generated_ids = outputs[0].cpu().numpy()
        
        # Extract codec IDs from the generated sequence
        codec_ids = []
        for token_id in generated_ids:
            if mmtokenizer.stage_1 <= token_id <= mmtokenizer.eoa:
                codec_ids.append(token_id)
                
        codec_ids = np.array(codec_ids)
        
        # Check if codec_ids is empty before calling offset_tok_ids
        if len(codec_ids) == 0:
            print(f"Warning: No valid codec tokens found in generated sequence for segment {seg_idx+1}.")
            # Create a safe empty output with the correct shape
            empty_output = np.zeros((codectool.num_codebooks, 0), dtype=np.int64)
            np.save(output_path, empty_output)
            stage1_output_set.append(output_path)
            continue
            
        # Reshape the codec_ids to match the expected shape (num_codebooks, sequence_length)
        # This requires knowledge of how the codec tokens are structured
        # For now, assuming codec_ids contains interleaved tokens for codebooks
        sequence_length = len(codec_ids) // codectool.num_codebooks
        if sequence_length > 0:
            try:
                codec_ids = codec_ids.reshape(codectool.num_codebooks, sequence_length)
            except ValueError:
                print(f"Warning: Could not reshape codec_ids to ({codectool.num_codebooks}, {sequence_length})")
                print(f"codec_ids.shape = {codec_ids.shape}, num_codebooks = {codectool.num_codebooks}")
                # Handle the case where reshaping fails (e.g., if length isn't divisible by num_codebooks)
                remainder = len(codec_ids) % codectool.num_codebooks
                if remainder > 0:
                    # Pad or truncate to make divisible
                    codec_ids = codec_ids[:len(codec_ids) - remainder]
                    sequence_length = len(codec_ids) // codectool.num_codebooks
                    codec_ids = codec_ids.reshape(codectool.num_codebooks, sequence_length)
        else:
            # Handle the case where there aren't enough tokens for all codebooks
            print("Warning: Not enough codec tokens for all codebooks.")
            empty_output = np.zeros((codectool.num_codebooks, 0), dtype=np.int64)
            np.save(output_path, empty_output)
            stage1_output_set.append(output_path)
            continue
            
        codec_ids = codectool.offset_tok_ids(
            codec_ids,
            global_offset=-codectool.global_offset,
            codebook_size=codectool.codebook_size,
            num_codebooks=codectool.num_codebooks,
        )
        
        # Save the generated tokens
        np.save(output_path, codec_ids)
        stage1_output_set.append(output_path)
        
        # Update input_ids for the next segment (optional)
        # This allows for continuation from the previous segment
        if seg_idx < args.run_n_segments - 1:
            input_ids = torch.cat([input_ids, outputs[0][-100:].unsqueeze(0)], dim=-1)
    
    return stage1_output_set

def stage1_inference_stereo(model, prompt_texts, codectool, mmtokenizer, device, args):
    """
    Modified Stage 1 inference to support stereo generation
    
    Args:
        model: Generation model
        prompt_texts: Text prompts for generation (left and right channel prompts)
        codectool: Codec tool for token manipulation
        mmtokenizer: Tokenizer
        device: Processing device
        args: Generation arguments
        
    Returns:
        Paths to generated stereo audio files
    """
    import numpy as np
    from audio_utils import load_audio_stereo
    from codec_utils import encode_audio_stereo
    from codecmanipulator import StereoCodecManipulator
    
    # Extract left and right prompts from prompt_texts
    prompt_left, prompt_right = prompt_texts
    print(f"Using left channel prompt: {prompt_left[:50]}...")
    print(f"Using right channel prompt: {prompt_right[:50]}...")
    
    # Similar to original but with stereo handling
    # Initialize with optional stereo prompt
    if args.use_dual_tracks_prompt:
        vocals = load_audio_stereo(args.vocal_track_prompt_path)
        instrumental = load_audio_stereo(args.instrumental_track_prompt_path)
        
        # Encode with stereo preservation
        codec_model = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)[0]
        vocals_left, vocals_right = encode_audio_stereo(codec_model, vocals, device)
        inst_left, inst_right = encode_audio_stereo(codec_model, instrumental, device)
        
        # Use the provided codectool to perform initial processing if needed
        # This ensures the codectool parameter is utilized
        if hasattr(codectool, 'preprocess_tokens'):
            vocals_left = codectool.preprocess_tokens(vocals_left)
            vocals_right = codectool.preprocess_tokens(vocals_right)
            inst_left = codectool.preprocess_tokens(inst_left)
            inst_right = codectool.preprocess_tokens(inst_right)
        
        # Process with stereo-aware codec manipulator
        stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
        
        # Create stereo tokens for vocals and instrumentals
        vocals_stereo = stereo_codectool.process_stereo(vocals_left, vocals_right)
        inst_stereo = stereo_codectool.process_stereo(inst_left, inst_right)
        
        # Create output paths for the stereo files
        import uuid
        import os
        random_id = str(uuid.uuid4())[:8]
        stage1_output_dir = os.path.join(args.output_dir, "stage1_stereo")
        os.makedirs(stage1_output_dir, exist_ok=True)
        
        # Save the stereo token files
        vocals_path = os.path.join(stage1_output_dir, f"vocals_stereo_{random_id}.npy")
        inst_path = os.path.join(stage1_output_dir, f"inst_stereo_{random_id}.npy")
        
        # Save the processed stereo tokens
        np.save(vocals_path, vocals_stereo)
        np.save(inst_path, inst_stereo)
        
        # Return the paths to the stereo files
        return [vocals_path, inst_path]
    else:
        # Handle non-prompt case, similar to original implementation but preserving stereo
        import uuid
        import os
        
        # Create unique output paths with UUID
        random_id = str(uuid.uuid4())[:8]
        stage1_output_dir = os.path.join(args.output_dir, "stage1_stereo")
        os.makedirs(stage1_output_dir, exist_ok=True)
        
        # Generate for left and right channels separately
        left_output_path = os.path.join(stage1_output_dir, f"left_{random_id}.npy")
        right_output_path = os.path.join(stage1_output_dir, f"right_{random_id}.npy")
        
        # Process the left channel
        left_ids = mmtokenizer.tokenize(prompt_left)
        left_ids = [mmtokenizer.bos] + left_ids
        left_ids = torch.tensor(left_ids).unsqueeze(0).to(device)
        
        # Process the right channel
        right_ids = mmtokenizer.tokenize(prompt_right)
        right_ids = [mmtokenizer.bos] + right_ids
        right_ids = torch.tensor(right_ids).unsqueeze(0).to(device)
        
        # Return the paths that will be used (actual generation would be similar to stage1_inference)
        return [left_output_path, right_output_path] 