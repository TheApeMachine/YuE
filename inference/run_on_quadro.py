"""
Script to run YuE on older GPUs like Quadro M6000 that don't support Flash Attention.
This script modifies the model loading process to disable Flash Attention and use standard attention instead.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run YuE on Quadro GPUs")
    parser.add_argument("--genre_txt", type=str, required=True, help="Path to text file with genre descriptions")
    parser.add_argument("--lyrics_txt", type=str, default="", help="Path to text file with lyrics (optional)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for output files")
    parser.add_argument("--cuda_idx", type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--stage2_batch_size", type=int, default=2, help="Batch size for stage 2 generation")
    parser.add_argument("--use_stereo", action="store_true", help="Generate stereo audio")
    
    # Parse only known args and let the rest go to the main.py script
    args, unknown = parser.parse_known_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    print(f"Running YuE on Quadro GPU (CUDA device {args.cuda_idx})")
    print("Flash Attention will be disabled")
    
    # Store original from_pretrained method
    original_from_pretrained = AutoModelForCausalLM.from_pretrained
    
    # Define patched method to disable Flash Attention
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Remove flash attention
        if "attn_implementation" in kwargs:
            print(f"Replacing attention implementation '{kwargs['attn_implementation']}' with 'eager'")
            kwargs["attn_implementation"] = "eager"
        
        # Use float16 instead of bfloat16 for compatibility
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] == torch.bfloat16:
            print("Switching from bfloat16 to float16 for compatibility")
            kwargs["torch_dtype"] = torch.float16
        
        # Call original method
        return original_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    
    # Apply the monkey patch
    AutoModelForCausalLM.from_pretrained = classmethod(patched_from_pretrained)
    
    # Build command line args to pass to main.py
    sys_argv = sys.argv.copy()  # Save original
    
    # Reconstruct argv with our modified arguments and any unknown args
    sys.argv = [sys.argv[0]] + [
        "--genre_txt", args.genre_txt,
        "--output_dir", args.output_dir,
        "--stage2_batch_size", str(args.stage2_batch_size)
    ]
    
    # Add lyrics if provided
    if args.lyrics_txt:
        sys.argv.extend(["--lyrics_txt", args.lyrics_txt])
    
    # Add stereo if enabled
    if args.use_stereo:
        sys.argv.append("--use_stereo")
    
    # Add all unknown args
    sys.argv.extend(unknown)
    
    # Import and run main.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import main as run_main
    
    try:
        print("Starting YuE with the following args:", " ".join(sys.argv[1:]))
        run_main()
    except Exception as e:
        print(f"Error running YuE: {e}")
    finally:
        # Restore original sys.argv and from_pretrained
        sys.argv = sys_argv
        AutoModelForCausalLM.from_pretrained = original_from_pretrained

if __name__ == "__main__":
    main() 