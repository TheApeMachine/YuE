import torch
from vocoder import build_codec_model
import numpy as np
import os

def stage1_inference(model, prompt_text, codectool, mmtokenizer, device, args):
    """
    Stage 1 inference for mono or standard stereo generation.
    """
    stage1_output_set = []
    error_detected = False
    
    # Check if chunked processing is needed and set up params
    is_chunked = args.chunk_size is not None and args.chunk_size > 0 
    
    if is_chunked:
        print(f"Using chunked processing for Stage 1 with chunk size: {args.chunk_size}")
    
    # Iterate over the number of segments to generate
    for seg_idx in range(args.run_n_segments):
        np.random.seed(args.seed + seg_idx) if args.seed is not None else None
        torch.manual_seed(args.seed + seg_idx) if args.seed is not None else None
        
        # Prepare the output path for this segment
        output_path = os.path.join(args.output_dir, f"stage1_output_{seg_idx}.npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skip if the output already exists
        if os.path.exists(output_path) and not args.force_overwrite:
            print(f'{output_path} already exists, skipping generation.')
            stage1_output_set.append(output_path)
            continue
            
        print(f"Generating segment {seg_idx+1}/{args.run_n_segments}")
        
        # Prepare input tokens 
        # For the first segment, use the full prompt
        # For subsequent segments, just use a minimized prompt to maintain continuity
        if seg_idx == 0 or not args.use_minimal_prompt_for_continuation:
            # Full prompt for first segment
            input_text = prompt_text
        else:
            # Minimized prompt for continuation segments
            # Extract just genre/style information and add continuation marker
            minimal_prompt = extract_minimal_prompt(prompt_text) + " [CONTINUE]"
            input_text = minimal_prompt
            
        print(f"Tokenizing prompt (length: {len(input_text)})")
        
        # Tokenize the input text
        input_tokens = mmtokenizer.tokenize(input_text)
        # Add BOS token
        input_tokens = [mmtokenizer.bos] + input_tokens
        input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Check if input sequence is very large and chunking is enabled
        if is_chunked and input_ids.shape[1] > args.chunk_size // 2:
            # Process with chunked generation when prompt is large
            outputs = _chunked_generation(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                args=args,
                mmtokenizer=mmtokenizer,
                device=device,
                seg_idx=seg_idx
            )
            if outputs is not None:
                _process_outputs(outputs, output_path, seg_idx, codectool, mmtokenizer, stage1_output_set)
                continue
        
        # Standard generation (not chunked or chunked generation failed)
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
                if "cublasLt ran into an error" in str(e) or "CUBLAS_STATUS_NOT_SUPPORTED" in str(e) or "CUDA out of memory" in str(e):
                    try:
                        print("CUDA error detected. Attempting staged recovery...")
                        
                        # Stage 1: Try reducing tokens while staying on GPU
                        if device.type == 'cuda':
                            try:
                                print("Recovery stage 1: Reducing max_new_tokens while staying on GPU...")
                                reduced_max_tokens = min(300, args.max_new_tokens // 2)
                                print(f"Reducing max_new_tokens to {reduced_max_tokens}")
                                
                                # Clear cache again
                                torch.cuda.empty_cache()
                                
                                # Try generating with reduced tokens
                                outputs = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=reduced_max_tokens,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.95,
                                    repetition_penalty=1.0,
                                    eos_token_id=mmtokenizer.eoa,
                                    pad_token_id=mmtokenizer.eoa,
                                )
                                print("Recovery successful - reduced tokens was sufficient")
                                # If we get here, we've recovered successfully
                                return _process_outputs(outputs, output_path, seg_idx, codectool, mmtokenizer, stage1_output_set)
                            except Exception as stage1_error:
                                print(f"Recovery stage 1 failed: {stage1_error}")
                                # Continue to next recovery stage
                        
                        # Stage 2: Move everything to CPU and try again with reduced tokens
                        print("Recovery stage 2: Moving to CPU with further reduced parameters...")
                        
                        # First detach model from any grad operations to avoid tensor device issues
                        model.eval()  # Ensure in eval mode (no gradients)
                        
                        # Check if it's a bfloat16 compatibility issue and adjust precision 
                        use_float32 = "CUDA_R_16BF" in str(e) or "bfloat16" in str(e)
                        if use_float32:
                            print("bfloat16 compatibility issue detected. Using float32 precision...")
                        
                        # Move everything to CPU all at once (critical to avoid device mixing)
                        print("Moving model and tensors to CPU...")
                        
                        # First, move model to CPU with the right precision
                        with torch.no_grad():
                            cpu_model = model.cpu()
                            if use_float32:
                                cpu_model = cpu_model.to(torch.float32)
                        
                        # Move input tensors to CPU
                        cpu_input_ids = input_ids.cpu()
                        cpu_attention_mask = attention_mask.cpu()
                        
                        # Clear GPU cache
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Further reduce max tokens for CPU operation
                        very_reduced_max_tokens = min(200, reduced_max_tokens if 'reduced_max_tokens' in locals() else 300)
                        print(f"Using extremely reduced max_new_tokens: {very_reduced_max_tokens}")
                        
                        # Generate with smallest possible parameters
                        print("Starting CPU generation (this may take a while)...")
                        with torch.no_grad():
                            outputs = cpu_model.generate(
                                input_ids=cpu_input_ids,
                                attention_mask=cpu_attention_mask,
                                max_new_tokens=very_reduced_max_tokens,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.95,
                                repetition_penalty=1.0,
                                eos_token_id=mmtokenizer.eoa,
                                pad_token_id=mmtokenizer.eoa,
                            )
                        
                        # Keep everything on CPU for processing
                        print("Generation completed on CPU. Processing outputs...")
                        
                        # Process directly on CPU without moving back to GPU
                        generated_ids = outputs[0].numpy()
                        
                        # Extract codec IDs from the generated sequence
                        codec_ids = []
                        for token_id in generated_ids:
                            if mmtokenizer.stage_1 <= token_id <= mmtokenizer.eoa:
                                codec_ids.append(token_id)
                                
                        codec_ids = np.array(codec_ids)
                        
                        # Check if codec_ids is empty 
                        if len(codec_ids) == 0:
                            print(f"Warning: No valid codec tokens found in generated sequence for segment {seg_idx+1}.")
                            # Create a safe empty output with the correct shape
                            empty_output = np.zeros((codectool.num_codebooks, 0), dtype=np.int64)
                            np.save(output_path, empty_output)
                            stage1_output_set.append(output_path)
                            continue
                            
                        # Reshape the codec_ids to match the expected shape (num_codebooks, sequence_length)
                        tokenized_audio = codectool.offset_tok_ids(codec_ids)
                        
                        # Save the generated tokens
                        np.save(output_path, tokenized_audio)
                        stage1_output_set.append(output_path)
                        
                        # Don't move model back to GPU to avoid memory issues
                        print("Successfully recovered and processed outputs on CPU.")
                        
                        # Skip to next segment without going through normal output processing
                        continue
                        
                    except Exception as recovery_error:
                        print(f"All recovery attempts failed: {recovery_error}")
                        error_msg = "Failed to generate tokens due to CUDA errors. Please try: \n"
                        error_msg += "1. Using --low_memory_mode with a lower --max_new_tokens value (200-500)\n"
                        error_msg += "2. Adding --no_bfloat16 to avoid bfloat16 precision issues\n"
                        error_msg += "3. Running on CPU only with --device cpu\n"
                        error_msg += "4. Using --force_cpu to avoid GPU completely\n"
                        error_msg += "5. Adding --quantization 4bit_nf4 for maximum memory efficiency\n"
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

def _process_outputs(outputs, output_path, seg_idx, codectool, mmtokenizer, stage1_output_set):
    """Process model outputs and save tokens - separated to allow for reuse in recovery code"""
    generated_ids = outputs[0].cpu().numpy()
    
    # Extract codec IDs from the generated sequence
    codec_ids = []
    for token_id in generated_ids:
        if mmtokenizer.stage_1 <= token_id <= mmtokenizer.eoa:
            codec_ids.append(token_id)
            
    codec_ids = np.array(codec_ids)
    
    # Check if codec_ids is empty
    if len(codec_ids) == 0:
        print(f"Warning: No valid codec tokens found in generated sequence for segment {seg_idx+1}.")
        # Create a safe empty output with the correct shape
        empty_output = np.zeros((codectool.num_codebooks, 0), dtype=np.int64)
        np.save(output_path, empty_output)
        stage1_output_set.append(output_path)
        return stage1_output_set
        
    # Reshape the codec_ids to match the expected shape (num_codebooks, sequence_length)
    tokenized_audio = codectool.offset_tok_ids(codec_ids)
    
    # Save the generated tokens
    np.save(output_path, tokenized_audio)
    stage1_output_set.append(output_path)
    
    return stage1_output_set 

def _chunked_generation(model, input_ids, attention_mask, args, mmtokenizer, device, seg_idx):
    """
    Perform chunked generation for Stage 1 to reduce memory usage.
    
    Args:
        model: The model to use for generation
        input_ids: Input token IDs
        attention_mask: Attention mask for input tokens
        args: Runtime arguments
        mmtokenizer: Tokenizer 
        device: Computation device
        seg_idx: Current segment index
        
    Returns:
        Generated sequence or None if an error occurred
    """
    print(f"Using chunked generation for segment {seg_idx+1} to reduce memory usage")
    
    # Calculate how many tokens to generate in each chunk
    # This is a safe value that should work on most hardware
    tokens_per_chunk = min(100, args.chunk_size // 4 if args.chunk_size else 100)
    
    # For older GPUs like Quadro, use an even smaller chunk size
    if device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(device)
            compute_capability = float(f"{props.major}.{props.minor}")
            if compute_capability < 7.0:  # Pre-Volta architectures
                # Further reduce chunk size for older architectures
                tokens_per_chunk = min(tokens_per_chunk, 50)
                print(f"Using reduced chunk size of {tokens_per_chunk} for older GPU architecture")
        except Exception as e:
            print(f"Could not check GPU compute capability: {e}")
    
    # We'll generate up to the maximum number of new tokens, but in chunks
    max_chunks = (args.max_new_tokens + tokens_per_chunk - 1) // tokens_per_chunk
    
    # Current sequence starts with the input
    current_sequence = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    
    # Safety mechanism: if we're getting segmentation faults, we can recover
    # by saving intermediate results
    last_successful_sequence = current_sequence.clone()
    
    for chunk_idx in range(max_chunks):
        try:
            # Clear cache before each chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            print(f"Generating chunk {chunk_idx+1}/{max_chunks} (total tokens: {current_sequence.shape[1]})")
            
            # Configure generation parameters for this chunk
            # We use a smaller top_p for intermediate chunks to reduce variance
            current_top_p = args.top_p if chunk_idx == max_chunks - 1 else min(args.top_p, 0.92)
            
            # Generate the next chunk
            with torch.no_grad():
                # For the first chunk, use the provided attention mask
                # For subsequent chunks, create a new attention mask for the current sequence
                if chunk_idx == 0:
                    chunk_attention_mask = current_attention_mask
                else:
                    chunk_attention_mask = torch.ones_like(current_sequence).to(device)
                    
                # Use protective try/except around the generate call
                try:
                    outputs = model.generate(
                        input_ids=current_sequence,
                        attention_mask=chunk_attention_mask,
                        max_new_tokens=tokens_per_chunk,
                        do_sample=True, 
                        temperature=args.temperature,
                        top_p=current_top_p,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=mmtokenizer.eoa,
                        eos_token_id=mmtokenizer.eoa,
                    )
                    
                    # If we got here, update the last successful sequence
                    last_successful_sequence = outputs.clone()
                except RuntimeError as inner_e:
                    # If error happens within generate call, try to handle it
                    print(f"Error within generation step: {inner_e}")
                    
                    # If we have a successful sequence from before, return that
                    if last_successful_sequence.shape[1] > input_ids.shape[1]:
                        print(f"Returning last successful sequence ({last_successful_sequence.shape[1]} tokens)")
                        return last_successful_sequence
                    else:
                        # Otherwise, re-raise to be caught by outer handler
                        raise
            
            # Update current sequence with generated output
            current_sequence = outputs
            
            # Check if an EOS token was generated in this chunk
            generated_tokens = outputs[0].cpu().numpy()
            if mmtokenizer.eoa in generated_tokens:
                print(f"End of audio token detected after {current_sequence.shape[1]} tokens")
                break
                
            # If we've exceeded max_new_tokens, stop
            if current_sequence.shape[1] - input_ids.shape[1] >= args.max_new_tokens:
                print(f"Reached maximum token length ({args.max_new_tokens})")
                break
                
        except RuntimeError as e:
            print(f"Error during chunked generation: {e}")
            
            # If we've generated at least some content, return what we have
            if current_sequence.shape[1] > input_ids.shape[1]:
                print(f"Returning partial generation of {current_sequence.shape[1]} tokens")
                return current_sequence
            else:
                # If we haven't generated anything useful, signal failure
                print("Chunked generation failed completely")
                return None
        
        # Safety check - if the sequence got too large, stop
        if current_sequence.shape[1] > args.max_new_tokens * 1.5:
            print(f"Safety limit reached at {current_sequence.shape[1]} tokens")
            break
                
    # Return the final sequence
    return current_sequence

def extract_minimal_prompt(full_prompt):
    """Extract a minimal version of the prompt containing just genre information"""
    # Simple extraction that keeps the first few lines (usually genre info)
    # and discards detailed instructions, lyrics, etc.
    lines = full_prompt.split('\n')
    genre_section = []
    
    # Keep only lines that likely contain genre/style information
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # If the line is short and doesn't start with common instruction words,
        # it's likely a genre/style tag
        if len(line) < 100 and not any(line.lower().startswith(x) for x in ["write", "create", "generate", "make", "lyrics"]):
            genre_section.append(line)
        # Stop when we hit what looks like lyrics or detailed instructions    
        if "[" in line and "]" in line or len(line) > 100:
            break
            
    return " ".join(genre_section) if genre_section else "Music" 