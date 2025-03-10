import math
import torch
import numpy as np

def enhance_stereo_width(audio, width=1.5, method='ms', preserve_bass=True, bass_freq=200):
    """
    Advanced stereo width enhancement with multiple methods and bass preservation.
    
    This enhanced implementation offers multiple stereo widening techniques and 
    preserves mono compatibility, especially in the bass frequencies where
    stereo widening can cause phase issues.
    
    Args:
        audio: Audio tensor, if mono will be converted to stereo
        width: Width factor (1.0=unchanged, >1.0=wider, <1.0=narrower)
        method: Widening method: 'ms' (mid/side), 'haas' (haas effect), 
                or 'complementary' (complementary comb filtering)
        preserve_bass: Whether to keep bass frequencies mono
        bass_freq: Frequency below which bass is preserved mono
        
    Returns:
        Stereo-enhanced audio
    """
    # Ensure stereo format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).repeat(2, 1)
    elif audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    
    # Get original channels
    left = audio[0]
    right = audio[1]
    
    # Split frequencies if bass preservation is enabled
    if preserve_bass and method != 'haas':
        from YuE.inference.dsp.filtering import apply_highpass, apply_bandpass
        
        # Extract bass as mono
        bass_mono = (left + right) / 2
        bass_mono = apply_bandpass(bass_mono, 20, bass_freq, sr=44100)
        
        # High-pass the original for processing
        highs_left = apply_highpass(left, bass_freq, sr=44100)
        highs_right = apply_highpass(right, bass_freq, sr=44100)
        
        # Create high-pass version for processing
        highs = torch.stack([highs_left, highs_right], dim=0)
    else:
        highs = audio
        bass_mono = None
    
    # Apply selected widening method
    if method == 'ms':
        # Mid/Side processing (traditional approach)
        mid = (highs[0] + highs[1]) / 2
        side = (highs[0] - highs[1]) / 2
        
        # Apply width factor to side channel
        side = side * width
        
        # Recombine
        new_left = mid + side
        new_right = mid - side
        
    elif method == 'haas':
        # Haas effect (delay-based widening)
        delay_samples = int(0.01 * 44100)  # 10ms default delay (adjust based on sample rate if needed)
        
        # Create delays in opposite directions
        delay_factor = min(1.0, width / 2)  # Scale delay amount by width
        actual_delay = int(delay_samples * delay_factor)
        
        if actual_delay > 0:
            # Delay right channel in left speaker, left channel in right speaker
            left_with_right = left.clone()
            right_with_left = right.clone()
            
            # Add delayed version of opposite channel with reduced amplitude
            if len(left) > actual_delay:
                left_with_right[actual_delay:] += right[:-actual_delay] * 0.6 * (width - 1.0)
                right_with_left[actual_delay:] += left[:-actual_delay] * 0.6 * (width - 1.0)
            
            # Mix with original
            mix_ratio = min(0.8, (width - 1.0) / 2)
            new_left = left * (1.0 - mix_ratio) + left_with_right * mix_ratio
            new_right = right * (1.0 - mix_ratio) + right_with_left * mix_ratio
        else:
            new_left = left
            new_right = right
            
    elif method == 'complementary':
        # Complementary comb filtering
        # This creates opposite comb patterns in L/R channels for widening
        comb_size = int(0.01 * 44100)  # 10ms comb (adjust if needed)
        comb_depth = min(0.5, (width - 1.0) / 2)
        
        new_left = left.clone()
        new_right = right.clone()
        
        # Create complementary comb pattern
        if comb_size > 0 and len(left) > comb_size:
            for i in range(1, 5):  # Use 5 reflection points for the comb
                if i * comb_size < len(left):
                    # Positive reflection in left, negative in right
                    reflection_gain = comb_depth * (0.7 ** (i-1))
                    new_left[i * comb_size:] += left[:-i * comb_size] * reflection_gain
                    new_right[i * comb_size:] -= right[:-i * comb_size] * reflection_gain
    else:
        # Fall back to default MS method
        mid = (highs[0] + highs[1]) / 2
        side = (highs[0] - highs[1]) / 2
        side = side * width
        new_left = mid + side
        new_right = mid - side
    
    # Recombine with mono bass if bass preservation was enabled
    if bass_mono is not None:
        new_left = new_left + bass_mono
        new_right = new_right + bass_mono
    
    # Ensure correct output levels (normalize if needed)
    if width > 1.5:
        # Prevent potential clipping from excessive width
        max_val = max(new_left.abs().max().item(), new_right.abs().max().item())
        if max_val > 0.98:
            gain = 0.98 / max_val
            new_left = new_left * gain
            new_right = new_right * gain
    
    return torch.stack([new_left, new_right], dim=0)

def pan_audio(audio, pan_position=0.0, pan_law='linear'):
    """
    Advanced constant-power panning with multiple pan laws.
    
    Args:
        audio: Audio tensor (will be converted to stereo if mono)
        pan_position: Pan from -1.0 (left) to 1.0 (right)
        pan_law: Panning law to use: 'linear', 'square_root' (constant power),
                or 'sinusoidal' (equal power)
    
    Returns:
        Panned audio tensor
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).repeat(2, 1)
    elif audio.dim() == 2 and audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    
    # Constrain pan position to valid range
    pan_ = torch.clamp(torch.tensor(pan_position), -1.0, 1.0)
    
    # Calculate gains based on pan law
    if pan_law == 'linear':
        # Linear panning (not constant power, but simple)
        left_gain = 1.0 - (pan_ + 1.0) / 2
        right_gain = (pan_ + 1.0) / 2
    
    elif pan_law == 'sinusoidal':
        # Sinusoidal law (equal power)
        angle = (pan_ + 1.0) * math.pi / 4
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)
    
    elif pan_law == 'square_root':
        # Square root law (constant power)
        # Normalized from -1...+1 to 0...1
        pan_norm = (pan_ + 1.0) / 2.0
        left_gain = math.sqrt(1.0 - pan_norm)
        right_gain = math.sqrt(pan_norm)
    
    else:
        # Default to equal power (sinusoidal) law
        angle = (pan_ + 1.0) * math.pi / 4
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)
    
    # Apply gains
    left = audio[0] * left_gain
    right = audio[1] * right_gain
    
    return torch.stack([left, right], dim=0)

def apply_haas_effect(audio, delay_ms=10.0, width=0.5, sr=44100):
    """
    Apply the Haas effect to create a natural sense of spaciousness.
    
    The Haas effect uses small delays (5-35ms) between channels to create
    a sense of spaciousness while maintaining mono compatibility.
    
    Args:
        audio: Audio tensor
        delay_ms: Delay time in milliseconds (5-35ms is typical)
        width: Strength of the effect (0.0=none to 1.0=maximum)
        sr: Sample rate
        
    Returns:
        Processed audio with enhanced spaciousness
    """
    # Ensure stereo
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).repeat(2, 1)
    elif audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    
    # Calculate delay samples
    delay_samples = int(delay_ms * sr / 1000)
    
    # Create delayed copies
    left = audio[0]
    right = audio[1]
    
    # Create output with original signal
    new_left = left.clone()
    new_right = right.clone()
    
    # Apply cross-delays at reduced amplitude
    if len(left) > delay_samples:
        # Add delayed right to left
        new_left[delay_samples:] += right[:-delay_samples] * width * 0.6
        
        # Add delayed left to right (different delay for more complexity)
        second_delay = int(delay_samples * 1.3)  # Slightly different delay
        if len(right) > second_delay:
            new_right[second_delay:] += left[:-second_delay] * width * 0.5
    
    return torch.stack([new_left, new_right], dim=0)