import torch
import math
import numpy as np
from YuE.inference.dsp.utils import apply_per_channel
from YuE.inference.dsp.filtering import apply_bandpass

def apply_saturation(audio, amount=0.5, saturation_type='tanh'):
    """
    Apply saturation effect to audio with multiple saturation types.
    
    Args:
        audio: Audio tensor
        amount: Saturation amount (0.0 to 1.0)
        saturation_type: Type of saturation curve:
            - 'tanh': Hyperbolic tangent (smooth, musical)
            - 'soft_clip': Soft clipping with variable knee
            - 'tube': Tube-style asymmetrical saturation
            - 'hard_clip': Hard clipping with variable threshold
        
    Returns:
        Saturated audio
    """
    def _saturate_single(channel):
        # Ensure amount is in valid range
        amt = max(0.0, min(1.0, amount))
        
        # Original signal
        if amt == 0.0:
            return channel
        
        # Apply drive to input (higher amount = more drive)
        drive = 1.0 + 4.0 * amt
        driven = channel * drive
        
        # Apply different saturation curves
        if saturation_type == 'tanh':
            # Hyperbolic tangent (smooth, musical)
            saturated = torch.tanh(driven)
        
        elif saturation_type == 'soft_clip':
            # Soft clipping with adjustable knee
            threshold = 1.0 - 0.4 * amt
            softness = 0.2 + 0.3 * amt
            
            # Create soft clip curve
            saturated = torch.zeros_like(driven)
            mask_below = driven.abs() <= threshold
            mask_above = ~mask_below
            
            # Pass through below threshold
            saturated[mask_below] = driven[mask_below]
            
            # Soft clip above threshold
            over = driven[mask_above]
            sign = torch.sign(over)
            over_normalized = (over.abs() - threshold) / softness
            clip_curve = threshold + softness * torch.tanh(over_normalized)
            saturated[mask_above] = sign * clip_curve
        
        elif saturation_type == 'tube':
            # Asymmetrical tube-style saturation (more distortion on negative half)
            pos_mask = driven > 0
            neg_mask = ~pos_mask
            
            saturated = torch.zeros_like(driven)
            
            # Positive half (cleaner)
            saturated[pos_mask] = torch.tanh(driven[pos_mask] * 0.8)
            
            # Negative half (more saturated)
            saturated[neg_mask] = torch.tanh(driven[neg_mask] * (1.0 + amt))
            
        elif saturation_type == 'hard_clip':
            # Hard clipping with variable threshold
            threshold = 1.0 - 0.5 * amt
            saturated = torch.clamp(driven, -threshold, threshold)
        
        else:
            # Default to tanh
            saturated = torch.tanh(driven)
        
        # Blend dry/wet based on amount
        mix_ratio = 0.5 + 0.5 * amt  # 0.5 to 1.0 mix range
        result = channel * (1.0 - mix_ratio) + (saturated / drive) * mix_ratio
        
        return result
    
    return apply_per_channel(audio, _saturate_single)

def multi_band_saturation(audio, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], 
                         amounts=[0.8, 0.4, 0.2, 0.1], types=None, sr=44100):
    """
    Apply different saturation amounts and types to different frequency bands.
    
    Args:
        audio: Audio tensor
        bands: List of (low_freq, high_freq) tuples defining bands
        amounts: Saturation amount for each band (0.0 to 1.0)
        types: Saturation type for each band (if None, all use 'tanh')
        sr: Sample rate
        
    Returns:
        Multi-band saturated audio
    """
    # Validate inputs
    if len(bands) != len(amounts):
        raise ValueError("Number of bands must match number of amounts")
    
    if types is None:
        types = ['tanh'] * len(bands)
    elif len(types) != len(bands):
        raise ValueError("If provided, number of types must match number of bands")
    
    # Split into bands
    band_signals = []
    
    for low_freq, high_freq in bands:
        band_signal = apply_bandpass(audio, low_freq, high_freq, sr)
        band_signals.append(band_signal)
    
    # Apply saturation to each band
    saturated_bands = []
    
    for i, band_signal in enumerate(band_signals):
        saturated = apply_saturation(band_signal, amounts[i], types[i])
        saturated_bands.append(saturated)
    
    # Sum bands back together
    result = torch.zeros_like(audio)
    for band in saturated_bands:
        result += band
    
    # Normalize to avoid clipping
    max_val = result.abs().max()
    if max_val > 1.0:
        result = result / max_val
    
    return result

def exciter(audio, amount=0.5, freq=3000, sr=44100):
    """
    Audio exciter effect that enhances high frequency harmonics
    by adding subtle saturation to the high frequency content.
    
    Args:
        audio: Audio tensor
        amount: Amount of excitement (0.0 to 1.0)
        freq: Frequency above which to apply the effect (Hz)
        sr: Sample rate
        
    Returns:
        Processed audio with enhanced high frequency content
    """
    # Extract high frequencies
    highs = apply_bandpass(audio, freq, sr/2, sr)
    
    # Apply saturation to highs only
    excited_highs = apply_saturation(highs, amount=amount*1.5, saturation_type='soft_clip')
    
    # Remove original highs and add excited highs
    lows = audio - highs
    result = lows + excited_highs
    
    # Blend with original based on amount
    return audio * (1.0 - amount*0.5) + result * (amount*0.5) 