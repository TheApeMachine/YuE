#!/usr/bin/env python3
"""
YuE System Diagnostic Tool

This script runs comprehensive diagnostics to detect potential issues with
your hardware, drivers, and libraries that might cause segmentation faults
or other errors during YuE inference.
"""

import sys
import argparse
from hardware import run_system_diagnostics
import torch
import os

def test_basic_gpu_operations(cuda_idx=0):
    """Run basic GPU operations to test stability"""
    print(f"\n--- TESTING BASIC GPU OPERATIONS (CUDA:{cuda_idx}) ---")
    
    try:
        # Make sure CUDA is available
        if not torch.cuda.is_available():
            print("CUDA is not available - cannot test GPU operations")
            return False
        
        # Make sure the requested device exists
        if cuda_idx >= torch.cuda.device_count():
            print(f"CUDA device {cuda_idx} not found (only {torch.cuda.device_count()} devices available)")
            return False
        
        # Set device
        device = torch.device(f"cuda:{cuda_idx}")
        print(f"Testing device: {device}")
        
        # Get memory info
        memory_info = torch.cuda.mem_get_info(cuda_idx)
        free_mem_gb = memory_info[0] / (1024**3)
        total_mem_gb = memory_info[1] / (1024**3)
        print(f"GPU memory: {free_mem_gb:.2f}GB free / {total_mem_gb:.2f}GB total")
        
        # Get device properties
        props = torch.cuda.get_device_properties(cuda_idx)
        print(f"Device: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        
        # Test increasing sizes to find limits
        test_sizes = [
            (1000, 1000),    # 1M elements
            (2000, 2000),    # 4M elements
            (4000, 4000),    # 16M elements
            (8000, 8000),    # 64M elements
        ]
        
        for size in test_sizes:
            try:
                print(f"Testing tensor operations with size {size[0]}x{size[1]}...")
                
                # Test memory allocation
                x = torch.zeros(size, device=device)
                y = torch.ones(size, device=device)
                
                # Test basic operations
                z = x + y
                
                # Test matrix multiplication
                a = torch.randn(size[0], 64, device=device)
                b = torch.randn(64, size[1], device=device)
                c = torch.matmul(a, b)
                
                # Clean up
                del x, y, z, a, b, c
                torch.cuda.empty_cache()
                
                print(f"✓ Size {size[0]}x{size[1]} successful")
            except Exception as e:
                print(f"✗ Size {size[0]}x{size[1]} failed: {e}")
                print("This suggests memory limitations or driver issues")
                return False
        
        print("\n✓ All basic GPU operations completed successfully!")
        return True
    
    except Exception as e:
        print(f"GPU test failed with error: {e}")
        return False

def test_bitsandbytes_operations():
    """Test if bitsandbytes quantization works"""
    print("\n--- TESTING BITSANDBYTES COMPATIBILITY ---")
    
    try:
        import bitsandbytes as bnb
        print(f"Bitsandbytes version: {bnb.__version__}")
        
        if not torch.cuda.is_available():
            print("CUDA is not available - cannot test bitsandbytes")
            return False
            
        # Simple linear layer test with 8-bit quantization
        print("Testing 8-bit linear layer...")
        linear_fp16 = torch.nn.Linear(1024, 1024).half().cuda()
        linear_8bit = bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False).cuda()
        
        # Copy weights for comparison
        linear_8bit.weight.data = linear_fp16.weight.data.clone()
        
        # Test forward pass
        test_input = torch.randn(32, 1024, device='cuda').half()
        out_fp16 = linear_fp16(test_input)
        out_8bit = linear_8bit(test_input)
        
        # Calculate difference
        diff = (out_fp16 - out_8bit).abs().mean().item()
        print(f"Mean absolute difference between FP16 and 8-bit: {diff:.6f}")
        
        # Test 4-bit
        print("Testing 4-bit linear layer...")
        try:
            linear_4bit = bnb.nn.Linear4bit(1024, 1024, bias=True).cuda()
            out_4bit = linear_4bit(test_input)
            print("✓ 4-bit operations successful")
        except Exception as e:
            print(f"✗ 4-bit operations failed: {e}")
            return False
            
        print("\n✓ Bitsandbytes tests completed successfully!")
        return True
        
    except ImportError:
        print("Bitsandbytes not installed - skipping quantization tests")
        return False
    except Exception as e:
        print(f"Bitsandbytes test failed with error: {e}")
        return False

def diagnose_nvidia_driver():
    """Check NVIDIA driver and provide recommendations"""
    print("\n--- NVIDIA DRIVER DIAGNOSTICS ---")
    
    try:
        import subprocess
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
        
        # Extract driver version
        driver_line = [line for line in nvidia_smi.split('\n') if 'Driver Version' in line][0]
        driver_version = driver_line.split('Driver Version:')[1].strip().split(' ')[0]
        print(f"NVIDIA driver version: {driver_version}")
        
        # Check CUDA version
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
        
        # Get GPU information
        gpu_info = [line for line in nvidia_smi.split('\n') if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'TITAN' in line]
        for gpu_line in gpu_info:
            print(f"Detected GPU: {gpu_line.strip()}")
        
        # If Quadro, check driver compatibility
        if any('Quadro' in line for line in gpu_info):
            print("\nDetected Quadro GPU - checking driver compatibility...")
            
            # Driver recommendations for Quadro cards
            driver_major = int(driver_version.split('.')[0])
            
            if driver_major < 470:
                print("⚠️ Driver version below 470.x may have issues with CUDA 11.x and PyTorch")
                print("   Recommendation: Update to driver version 470.x for Maxwell/Pascal GPUs")
            elif 470 <= driver_major < 500:
                print("✓ Driver version in 470.x range, which is optimal for older Quadro GPUs")
            else:
                print("⚠️ Driver version 500.x+ may have reduced support for older Quadro GPUs")
                print("   Recommendation: Consider downgrading to driver version 470.x")
        
        return True
    except Exception as e:
        print(f"NVIDIA driver diagnostics failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="YuE System Diagnostic Tool")
    parser.add_argument("--cuda_idx", type=int, default=0, help="CUDA device index to test")
    parser.add_argument("--skip_gpu_tests", action="store_true", help="Skip GPU-specific tests")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed information")
    args = parser.parse_args()
    
    print("=== YuE SYSTEM DIAGNOSTIC TOOL ===")
    print(f"Running complete diagnostics for your system...")
    
    # Run comprehensive system diagnostics
    run_system_diagnostics()
    
    # Run additional GPU tests if not skipped
    if not args.skip_gpu_tests:
        test_basic_gpu_operations(args.cuda_idx)
        test_bitsandbytes_operations()
    
    # Check NVIDIA driver
    diagnose_nvidia_driver()
    
    print("\n=== FINAL RECOMMENDATIONS ===")
    
    # Make recommendations based on findings
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.cuda_idx)
        cc = f"{props.major}.{props.minor}"
        
        if props.major < 7:
            print(f"GPU {props.name} has compute capability {cc} (pre-Volta architecture).")
            print("Recommended command line options:")
            print("  --disable_flash_attention              # Mandatory for older GPUs")
            print("  --quantization 4bit_nf4                # For better memory efficiency")
            print("  --low_memory_mode                      # Enable memory-saving optimizations")
            print("  --max_new_tokens 500                   # Limit generation length")
            print("  --chunk_size 400                       # Enable chunked processing")
            print("  --gradient_checkpointing               # Save memory during processing")
            
            if 'Quadro' in props.name:
                print("\nFor Maxwell-era Quadro GPUs (compute capability 5.x):")
                print("1. Try using CUDA 11.8 with NVIDIA driver version 470.x")
                print("2. If segmentation faults persist, consider disabling compilation:")
                print("   --disable_torch_compile")
                
        elif props.major == 7:
            print(f"GPU {props.name} has compute capability {cc} (Volta/Turing architecture).")
            print("Recommended command line options:")
            print("  --disable_flash_attention              # Recommended for stability")
            print("  --quantization 8bit                    # Good balance of quality/memory")
            print("  --low_memory_mode                      # Only if memory is limited")
            
        else:
            print(f"GPU {props.name} has compute capability {cc} (Ampere or newer).")
            print("Your GPU is fully compatible with all YuE features!")
    else:
        print("No CUDA-capable GPU detected. You can only run YuE in CPU mode.")
        print("Recommended command line options:")
        print("  --device cpu                               # Run on CPU")
        print("  --max_new_tokens 300                       # Keep generation short")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 