import torch
import math
import numpy as np
from YuE.inference.dsp.utils import apply_per_channel
from YuE.inference.dsp.filtering import apply_highpass, apply_lowpass

def apply_reverb(audio, mix=0.3, room_size=0.8, damping=0.5, sr=44100, pre_delay_ms=20):
    """
    Apply reverb to audio using an efficient feedback delay network.
    
    Args:
        audio: Audio tensor
        mix: Dry/wet mix (0.0 to 1.0)
        room_size: Room size (0.0 to 1.0)
        damping: Damping factor (0.0 to 1.0)
        sr: Sample rate
        pre_delay_ms: Pre-delay in milliseconds
        
    Returns:
        Reverberated audio
    """
    def _reverb_single(channel):
        # Calculate delay line lengths based on prime numbers for better diffusion
        # and room size parameter
        size_factor = 0.7 + room_size * 0.5  # Map 0-1 to 0.7-1.2 range
        delay_times = [0.0297, 0.0371, 0.0411, 0.0437]
        delay_times = [t * size_factor for t in delay_times]
        delay_lengths = [int(sr * t) for t in delay_times]
        max_delay = max(delay_lengths)
        
        # Pre-delay
        pre_delay_samples = int(sr * pre_delay_ms / 1000)
        
        # Feedback matrix (Hadamard)
        # This ensures the energy is conserved across the network
        feedback_matrix = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0]
        ]) * 0.5
        
        # Apply room size to feedback gain
        # Larger room = more feedback = longer decay
        feedback_gain = 0.7 * room_size
        
        # Initialize delay lines
        delay_lines = [torch.zeros(length, device=channel.device, dtype=channel.dtype) 
                      for length in delay_lengths]
        
        # Initialize output
        output = torch.zeros_like(channel)
        
        # Pre-delay buffer
        if pre_delay_samples > 0:
            pre_delay_buffer = torch.zeros(pre_delay_samples, 
                                         device=channel.device, 
                                         dtype=channel.dtype)
        
        # Process audio
        for i in range(len(channel)):
            # Get input sample (with pre-delay if enabled)
            if pre_delay_samples > 0:
                # Shift buffer and add new sample
                if i < len(channel) - pre_delay_samples:
                    input_sample = channel[i + pre_delay_samples]
                else:
                    input_sample = 0.0
            else:
                input_sample = channel[i]
            
            # Read from delay lines
            delay_outputs = torch.tensor([line[-1] for line in delay_lines], 
                                       device=channel.device,
                                       dtype=channel.dtype)
            
            # Apply feedback matrix
            feedback = torch.matmul(feedback_matrix, delay_outputs) * feedback_gain
            
            # Apply damping (low-pass filtering)
            damping_factor = 1.0 - damping * 0.5
            delay_outputs = delay_outputs * damping_factor
            
            # Write to delay lines (input + feedback)
            for j in range(len(delay_lines)):
                # Shift delay line and add new sample
                delay_lines[j] = torch.cat([
                    torch.tensor([input_sample + feedback[j]], 
                               device=channel.device, 
                               dtype=channel.dtype),
                    delay_lines[j][:-1]
                ])
            
            # Mix dry and wet
            wet_signal = torch.sum(delay_outputs) / 4.0
            output[i] = channel[i] * (1.0 - mix) + wet_signal * mix
        
        return output
    
    return apply_per_channel(audio, _reverb_single)

def apply_convolution_reverb(audio, impulse_response, mix=0.3, sr=44100):
    """
    Apply convolution reverb using an impulse response.
    
    Args:
        audio: Audio tensor
        impulse_response: Impulse response tensor
        mix: Dry/wet mix (0.0 to 1.0)
        sr: Sample rate
        
    Returns:
        Convolution reverb applied to audio
    """
    def _convolve_single(channel):
        # Convert to numpy for faster convolution
        device = channel.device
        dtype = channel.dtype
        
        # Extract IR and ensure it's the right format
        ir = impulse_response
        if ir.dim() > 1:
            if ir.shape[0] > 1:
                # Use first channel of stereo IR
                ir = ir[0]
            else:
                ir = ir.squeeze(0)
        
        # Efficient convolution using FFT
        channel_np = channel.cpu().numpy()
        ir_np = ir.cpu().numpy()
        
        # Perform convolution
        # Use numpy's built-in fft-based convolution for efficiency
        # (much faster than direct convolution)
        result_np = np.convolve(channel_np, ir_np, mode='full')[:len(channel_np)]
        
        # Convert back to tensor
        result = torch.tensor(result_np, device=device, dtype=dtype)
        
        # Normalize output level
        max_val = result.abs().max()
        if max_val > 1e-6:  # Avoid division by zero
            result = result / max_val
        
        # Apply dry/wet mix
        return channel * (1.0 - mix) + result * mix
    
    return apply_per_channel(audio, _convolve_single)

def add_space(audio, space_type='small_room', mix=0.3, sr=44100):
    """
    Add spatial characteristics to audio using preset reverb settings.
    
    Args:
        audio: Audio tensor
        space_type: Type of space ('small_room', 'medium_room', 'large_hall', 
                   'plate', 'chamber', 'ambient')
        mix: Dry/wet mix (0.0 to 1.0)
        sr: Sample rate
        
    Returns:
        Audio with added spatial characteristics
    """
    # Define preset reverb settings
    if space_type == 'small_room':
        room_size = 0.5
        damping = 0.7
        pre_delay_ms = 10
    elif space_type == 'medium_room':
        room_size = 0.7
        damping = 0.5
        pre_delay_ms = 20
    elif space_type == 'large_hall':
        room_size = 0.9
        damping = 0.3
        pre_delay_ms = 30
    elif space_type == 'plate':
        room_size = 0.6
        damping = 0.2
        pre_delay_ms = 5
    elif space_type == 'chamber':
        room_size = 0.75
        damping = 0.4
        pre_delay_ms = 15
    elif space_type == 'ambient':
        room_size = 0.8
        damping = 0.8
        pre_delay_ms = 40
    else:
        # Default to medium room
        room_size = 0.7
        damping = 0.5
        pre_delay_ms = 20
    
    # High-pass filter the reverb for cleaner sound (remove rumble/mud)
    # Common practice in professional mixing
    result = apply_reverb(audio, mix, room_size, damping, sr, pre_delay_ms)
    
    # Apply high-pass to the wet signal only
    dry = audio
    wet = (result - dry * (1.0 - mix)) / mix
    wet = apply_highpass(wet, 100, sr)  # Remove low-end rumble
    
    # Recombine
    return dry * (1.0 - mix) + wet * mix 