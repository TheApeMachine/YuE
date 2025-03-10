#!/usr/bin/env python
"""
Example script for testing the enhanced audio mixing features.
This demonstrates how to use the main entry point with appropriate flags for audio mixing tests.
"""

import os
import sys
import argparse
import subprocess

# Add parent directory to path to allow importing from inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run audio mixing tests"""
    parser = argparse.ArgumentParser(description="Test audio mixing features")
    parser.add_argument("--vocal_path", type=str, required=True,
                       help="Path to vocal audio file")
    parser.add_argument("--instrumental_path", type=str, required=True,
                       help="Path to instrumental audio file")
    parser.add_argument("--output_path", type=str, default="./mixed_output.wav",
                       help="Output path for mixed audio")
    parser.add_argument("--processing_level", type=str, 
                       choices=["minimal", "standard", "full"], default="full",
                       help="Level of audio processing to apply")
    
    args = parser.parse_args()
    
    # Prepare command-line arguments for running the main script with audio mixing test mode
    cmd = [
        "python", "../infer.py",
        "--test_audio_mixing",  # Special flag to trigger audio mixing test mode
        "--vocal_path", args.vocal_path,
        "--instrumental_path", args.instrumental_path,
        "--output_path", args.output_path,
        "--audio_processing_level", args.processing_level
    ]
    
    # Run the command
    print("Running audio mixing test with command: " + " ".join(cmd))
    subprocess.run(cmd)
    
    print(f"Mixed audio saved to: {args.output_path}")

if __name__ == "__main__":
    main() 