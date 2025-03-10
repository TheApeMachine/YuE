import os
import torch
import numpy as np
from tqdm import tqdm
import torchaudio
from generation import process_and_save_audio
from audio_mixing import align_phases, enhanced_audio_mix
from token_fixer import fix_tokens
from transformers import LogitsProcessorList
from codec_utils import BlockTokenRangeProcessor

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

def stage2_inference(model, stage1_output_set, stage2_output_dir, codectool, mmtokenizer, device, batch_size=4, apply_enhancements=True, diffusion_postproc_model=None, diffusion_steps=50, diffusion_sampling_method='ddpm', chunk_size=None, audio_processing_level="full"):
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
        chunk_size: Maximum number of tokens to process in one chunk (reduces memory usage)
        audio_processing_level: Level of audio post-processing to apply
        
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
        
        # Determine if we need to use chunked processing
        if chunk_size is not None and chunk_size > 0:
            # Use explicit chunking regardless of batch size
            tokens_per_segment = 300  # 6 seconds of audio at 50 tokens/second
            max_segments_per_chunk = chunk_size // tokens_per_segment
            
            if max_segments_per_chunk == 0:
                print(f"Warning: chunk_size {chunk_size} is too small, using minimum chunk size of {tokens_per_segment}")
                max_segments_per_chunk = 1
                
            print(f"Using chunked processing: {max_segments_per_chunk} segments per chunk")
            
            outputs = []
            for j in range(0, num_batch, max_segments_per_chunk):
                end_segment = min(j + max_segments_per_chunk, num_batch)
                segments_in_chunk = end_segment - j
                
                start_idx = j * 300
                end_idx = end_segment * 300
                
                print(f"Processing chunk {j//max_segments_per_chunk + 1}: segments {j+1}-{end_segment} ({segments_in_chunk} segments)")
                
                # Generate this chunk
                chunk_output = stage2_generate(
                    model, 
                    prompt[:, start_idx:end_idx], 
                    codectool, 
                    mmtokenizer, 
                    device, 
                    batch_size=min(segments_in_chunk, batch_size)
                )
                outputs.append(chunk_output)
                
                # Free up memory
                torch.cuda.empty_cache()
            
            # Combine all chunks
            output = torch.cat(outputs, dim=0)
            
        elif num_batch <= batch_size:
            # Generate audio from codec tokens (standard mode)
            output = stage2_generate(model, prompt[:, :output_duration*50], codectool, mmtokenizer, device, batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            outputs = []
            for j in range(0, num_batch, batch_size):
                start_idx = j * 300
                end_idx = min((j + batch_size) * 300, output_duration*50)
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
            
            # Combine all chunks
            output = torch.cat(outputs, dim=0)
        
        # Process and save audio
        output_path = process_and_save_audio(
            output, 
            output_filename, 
            codectool,
            apply_enhancements=apply_enhancements,
            diffusion_postproc_model=diffusion_postproc_model,
            diffusion_steps=diffusion_steps,
            diffusion_sampling_method=diffusion_sampling_method,
            audio_processing_level=audio_processing_level
        )
        stage2_result.append(output_path)
    
    return stage2_result

def stage2_inference_stereo(
    model, 
    stage1_output_set, 
    stage2_output_dir, 
    codectool, 
    mmtokenizer, 
    device, 
    batch_size=4, 
    apply_enhancements=True, 
    diffusion_postproc_model=None, 
    diffusion_steps=50, 
    diffusion_sampling_method='ddpm', 
    chunk_size=None, 
    audio_processing_level="full"
):
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
        chunk_size: Maximum number of tokens to process in one chunk (reduces memory usage)
        audio_processing_level: Level of audio post-processing to apply
        
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
            
            # Determine if we need to use chunked processing based on chunk_size
            if chunk_size is not None and chunk_size > 0:
                # Use explicit chunking with the specified chunk size
                tokens_per_segment = 300  # 6 seconds of audio at 50 tokens/second
                max_segments_per_chunk = chunk_size // tokens_per_segment
                
                if max_segments_per_chunk == 0:
                    print(f"Warning: chunk_size {chunk_size} is too small, using minimum chunk size of {tokens_per_segment}")
                    max_segments_per_chunk = 1
                    
                print(f"Using chunked processing: {max_segments_per_chunk} segments per chunk")
                
                # Process with more granular chunking based on chunk_size
                for j in range(0, num_batch, max_segments_per_chunk):
                    end_segment = min(j + max_segments_per_chunk, num_batch)
                    segments_in_chunk = end_segment - j
                    
                    start_idx = j * 300
                    end_idx = end_segment * 300
                    
                    left, right = stage2_generate_stereo(
                        model,
                        prompt_left[:, start_idx:end_idx],
                        prompt_right[:, start_idx:end_idx],
                        codectool,
                        mmtokenizer,
                        device,
                        batch_size=segments_in_chunk
                    )
                    
                    segments_left.append(left)
                    segments_right.append(right)
            else:
                # Use the default batch-based chunking
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
                        codectool,
                        apply_enhancements=True,
                        diffusion_postproc_model=diffusion_postproc_model,
                        diffusion_steps=diffusion_steps,
                        diffusion_sampling_method=diffusion_sampling_method,
                        audio_processing_level=audio_processing_level
                    )
                
                stage2_result_enhanced.append(enhanced_path)
            else:
                stage2_result_enhanced.append(output_path)
        
        return stage2_result_enhanced
    else:
        return stage2_result

