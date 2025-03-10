import os
import sys
import argparse
from main import main

def get_script_description():
    """Get a detailed script description for the help message"""
    desc = """
YuE Music Generation System

A powerful audio generation system that can create complete songs with vocals and instrumental tracks.
This is the main entry point that supports all functionality including:

- Standard music generation
- Compatibility with older GPUs (like Quadro M6000)
- Audio mixing testing
- Safe mode with conservative settings for resource-constrained systems

For detailed examples, check the examples/ directory.
"""
    return desc

if __name__ == "__main__":
    # Redirect to the modular implementation with status code
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
