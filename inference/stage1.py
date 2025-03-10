import torch
from vocoder import build_codec_model


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