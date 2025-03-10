#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for using YuE with diffusion model enhancements

This script shows how to generate music using YuE with the three diffusion-based 
enhancement methods:
1. Hybrid Architecture: Transformer for structure, diffusion for quality
2. Post-Processing: Apply diffusion-based enhancement after generation
3. Conditional Generation: Use diffusion models conditioned on codec tokens

All three methods are opt-in via command line arguments.
"""

import os
import argparse
import subprocess

def main():
    """Run YuE with diffusion enhancements enabled"""
    parser = argparse.ArgumentParser(description="Generate music with YuE using diffusion enhancements")
    
    # Required arguments
    parser.add_argument("--genre_txt", type=str, required=True, 
                      help="Path to text file with genre descriptions")
    parser.add_argument("--lyrics_txt", type=str, default="", 
                      help="Path to text file with lyrics (optional)")
    parser.add_argument("--output_dir", type=str, default="./diffusion_output", 
                      help="Directory for output files")
    
    # Diffusion model parameters
    parser.add_argument("--diffusion_model_path", type=str, required=True,
                      help="Path to pre-trained diffusion model weights")
    parser.add_argument("--diffusion_steps", type=int, default=50,
                      help="Number of diffusion steps (higher = better quality but slower)")
    parser.add_argument("--diffusion_sampling", type=str, default="ddpm", 
                      choices=["ddpm", "ddim", "plms"],
                      help="Sampling method for diffusion")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                      help="Guidance scale for classifier-free guidance")
    
    # Enhancement strategy selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all_enhancements", action="store_true",
                     help="Enable all three diffusion enhancements")
    group.add_argument("--hybrid_only", action="store_true",
                     help="Only use hybrid architecture enhancement")
    group.add_argument("--postproc_only", action="store_true",
                     help="Only use post-processing enhancement")
    group.add_argument("--conditional_only", action="store_true",
                     help="Only use conditional diffusion enhancement")
    
    # Other YuE parameters
    parser.add_argument("--use_stereo", action="store_true",
                      help="Generate stereo audio")
    parser.add_argument("--use_audio_prompt", action="store_true",
                      help="Use audio prompt")
    parser.add_argument("--audio_prompt_path", type=str, default="",
                      help="Path to audio prompt file")
    
    args = parser.parse_args()
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build YuE command with appropriate diffusion flags
    command = [
        "python", "YuE/inference/main.py",
        "--genre_txt", args.genre_txt,
        "--output_dir", args.output_dir,
        "--use_diffusion"
    ]
    
    # Add diffusion model path
    command.extend(["--diffusion_model_path", args.diffusion_model_path])
    
    # Add diffusion parameters
    command.extend(["--diffusion_steps", str(args.diffusion_steps)])
    command.extend(["--diffusion_sampling_method", args.diffusion_sampling])
    command.extend(["--diffusion_guidance_scale", str(args.guidance_scale)])
    
    # Add lyrics if provided
    if args.lyrics_txt:
        command.extend(["--lyrics_txt", args.lyrics_txt])
    
    # Add stereo if enabled
    if args.use_stereo:
        command.append("--use_stereo")
    
    # Add audio prompt if enabled
    if args.use_audio_prompt:
        command.append("--use_audio_prompt")
        if args.audio_prompt_path:
            command.extend(["--audio_prompt_path", args.audio_prompt_path])
    
    # Add enhancement flags based on selection
    if args.all_enhancements:
        command.append("--use_hybrid_architecture")
        command.append("--use_diffusion_postprocessing")
        command.append("--use_conditional_diffusion")
        print("Using ALL diffusion enhancements: Hybrid Architecture + Post-Processing + Conditional")
    elif args.hybrid_only:
        command.append("--use_hybrid_architecture")
        print("Using Hybrid Architecture enhancement only")
    elif args.postproc_only:
        command.append("--use_diffusion_postprocessing")
        print("Using Post-Processing enhancement only")
    elif args.conditional_only:
        command.append("--use_conditional_diffusion")
        print("Using Conditional Generation enhancement only")
    
    # Print the command
    print("\nRunning YuE with the following command:")
    print(" ".join(command))
    print("\n")
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print("\nGeneration completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError running YuE: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 