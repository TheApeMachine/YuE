#!/usr/bin/env python
"""
Example script for running YuE on older GPUs like Quadro M6000 that don't support Flash Attention.
This example demonstrates how to use the main entry point with appropriate flags for Quadro compatibility.
"""

import os
import sys
import subprocess

# Add parent directory to path to allow importing from inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Example command
def main():
    """Run YuE with Quadro-compatible settings"""
    print("Running YuE on Quadro GPUs using the main entry point")
    
    # Prepare command-line arguments
    cmd = [
        "python", "../infer.py",
        "--genre_txt", "../../prompt_egs/genre.txt",
        "--lyrics_txt", "../../prompt_egs/lyrics.txt",
        "--output_dir", "../../output",
        "--disable_flash_attention",  # Key flag for Quadro compatibility
        "--enable_torch_compile",     # Use PyTorch 2.0+ compilation as alternative optimization
        "--quantization", "8bit",     # Use 8-bit quantization to reduce memory usage
        "--audio_processing_level", "standard",  # Use standard audio processing (faster)
        "--stage2_batch_size", "2",   # Use a smaller batch size
        "--run_n_segments", "1"       # Start with one segment for testing
    ]
    
    # For systems with multiple GPUs, add model split settings
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) > 1:
        cmd.extend([
            "--model_split_strategy", "model_type",
            "--transformer_device", "cuda:0,1",  # Use multiple GPUs for transformer
            "--diffusion_device", "cuda:0",      # Use first GPU for diffusion
            "--codec_device", "cuda:1"           # Use second GPU for codec
        ])
    
    # Run the command
    print("Running command: " + " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 