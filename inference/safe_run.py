#!/usr/bin/env python3
"""
Safe Run Wrapper for YuE Music Generation

This script provides a safer way to run YuE music generation, especially on systems
with GPU compatibility issues or when running in WSL (Windows Subsystem for Linux).
It automatically sets conservative parameters to prioritize successful generation
over speed or quality.
"""

import os
import sys
import argparse
import platform
import subprocess
import signal

def is_wsl():
    """Check if running under Windows Subsystem for Linux"""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if "microsoft" in f.read().lower():
                return True
    return False

def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == "windows"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Safe wrapper for YuE music generation")
    
    # Required arguments
    parser.add_argument("--genre_txt", type=str, required=True, 
                        help="Path to genre text file", default="../prompt_egs/genre.txt")
    
    # Optional arguments (with safe defaults)
    parser.add_argument("--output_dir", type=str, default="../output",
                        help="Output directory")
    parser.add_argument("--lyrics_txt", type=str, default="../prompt_egs/lyrics.txt",
                        help="Path to lyrics text file (optional)")
    parser.add_argument("--title", type=str, default="",
                        help="Song title")
    parser.add_argument("--instruction", type=str, default="",
                        help="Additional instructions")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--run_n_segments", type=int, default=1,
                        help="Number of segments to generate")
    parser.add_argument("--stage2_batch_size", type=int, default=1,
                        help="Batch size for Stage 2 processing")
    
    # Safety flags - these can be disabled but are enabled by default
    parser.add_argument("--disable_safety", action="store_true",
                        help="Disable all safety measures (not recommended)")
    parser.add_argument("--allow_gpu", action="store_true",
                        help="Allow GPU usage if available (may be unstable in WSL)")
    
    # Pass-through arguments
    parser.add_argument("--extra_args", type=str, default="",
                        help="Additional arguments to pass to the main script")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Base command with essential arguments
    cmd = [
        sys.executable,
        "main.py",
        f"--genre_txt={args.genre_txt}",
        f"--output_dir={args.output_dir}",
    ]
    
    # Add optional arguments if provided
    if args.lyrics_txt:
        cmd.append(f"--lyrics_txt={args.lyrics_txt}")
    if args.title:
        cmd.append(f"--title={args.title}")
    if args.instruction:
        cmd.append(f"--instruction={args.instruction}")
    
    # Add generation parameters
    cmd.append(f"--max_new_tokens={args.max_new_tokens}")
    cmd.append(f"--run_n_segments={args.run_n_segments}")
    cmd.append(f"--stage2_batch_size={args.stage2_batch_size}")
    
    # Apply safety measures unless disabled
    if not args.disable_safety:
        # Always enable low memory mode for safety
        cmd.append("--low_memory_mode")
        
        # Disable bfloat16 for better compatibility
        cmd.append("--no_bfloat16")
        
        # Use CPU only in WSL unless specifically allowed
        if is_wsl() and not args.allow_gpu:
            cmd.append("--force_cpu")
            print("Detected WSL environment - forcing CPU-only mode for stability")
        
        # Set temperature and top_p for stable generation
        cmd.append("--temperature=0.7")
        cmd.append("--top_p=0.95")
        
        # Keep generation shorter for stability
        if args.max_new_tokens > 500:
            print(f"Warning: reducing max_new_tokens from {args.max_new_tokens} to 500 for stability")
            # Replace the existing max_new_tokens value
            for i, arg in enumerate(cmd):
                if arg.startswith("--max_new_tokens="):
                    cmd[i] = "--max_new_tokens=500"
                    break
    
    # Add any extra arguments
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    
    # Print the command
    print("\nRunning YuE with safe settings:")
    print(" ".join(cmd))
    print("\nThis may take some time, especially on CPU...\n")
    
    # Run the command
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping generation...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        print(f"Error running YuE: {e}")
        return 1
    
    return process.returncode

if __name__ == "__main__":
    sys.exit(main()) 