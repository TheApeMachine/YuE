#!/usr/bin/env python
"""
Example script for running YuE in safe mode with conservative defaults.
This demonstrates how to use the main entry point with safe settings for systems with limited resources.
"""

import os
import sys
import argparse
import subprocess

# Add parent directory to path to allow importing from inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run YuE in safe mode"""
    parser = argparse.ArgumentParser(description="Run YuE in safe mode")
    parser.add_argument("--genre_txt", type=str, default="../../prompt_egs/genre.txt",
                       help="Path to genre text file")
    parser.add_argument("--lyrics_txt", type=str, default="../../prompt_egs/lyrics.txt",
                       help="Path to lyrics text file")
    parser.add_argument("--output_dir", type=str, default="../../output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Prepare command-line arguments for safe mode
    cmd = [
        "python", "../infer.py",
        "--genre_txt", args.genre_txt,
        "--lyrics_txt", args.lyrics_txt,
        "--output_dir", args.output_dir,
        "--safe_mode",                      # Enable safe mode
        "--max_new_tokens", "500",          # Use fewer tokens
        "--run_n_segments", "1",            # Only generate one segment
        "--stage2_batch_size", "1",         # Minimal batch size for safety
        "--quantization", "4bit_nf4",       # Use 4-bit quantization to reduce memory
        "--audio_processing_level", "minimal",  # Minimal audio processing
        "--diffusion_optimization", "faster",   # Faster diffusion (if enabled)
        "--disable_flash_attention",        # Disable flash attention for compatibility
    ]
    
    # Run the command
    print("Running YuE in safe mode with command: " + " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 