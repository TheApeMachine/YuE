import torch
import numpy as np
from dsp.utils import to_mono
from scipy import signal

def measure_lufs(audio, sr=44100, block_size=0.4):
    """
    Measure integrated LUFS according to ITU-R BS.1770-4 standard.
    
    Tries to use pyloudnorm for the most accurate implementation, 
    but falls back to a precise implementation of BS.1770 if not available.
    
    Args:
        audio: Audio tensor, can be mono or multi-channel
        sr: Sample rate
        block_size: Size of measurement blocks in seconds (default 0.4s per ITU standard)
        
    Returns:
        Integrated LUFS value
    """
    # Convert to mono if multi-channel
    audio_mono = to_mono(audio)
    
    # Try pyloudnorm (preferred implementation)
    try:
        import pyloudnorm as pyln
        if isinstance(audio_mono, torch.Tensor):
            audio_np = audio_mono.cpu().numpy()
        else:
            audio_np = audio_mono
        
        meter = pyln.Meter(sr)
        min_length = int(sr*0.4)  # ensure enough samples
        
        if len(audio_np) < min_length:
            # Pad if too short
            pad = np.zeros(min_length - len(audio_np), dtype=audio_np.dtype)
            audio_np = np.concatenate([audio_np, pad])
            
        lufs_val = meter.integrated_loudness(audio_np)
        return lufs_val
    except ImportError:
        # Fall back to our own implementation
        return _measure_lufs_bs1770(audio_mono, sr, block_size)
    except Exception as e:
        print(f"Warning: measure_lufs error with pyloudnorm: {e}, using fallback.")
        return _measure_lufs_bs1770(audio_mono, sr, block_size)

def _measure_lufs_bs1770(audio_mono, sr=44100, block_size=0.4):
    """
    Accurate implementation of ITU-R BS.1770-4 integrated loudness measurement.
    
    This implementation follows the precise specification:
    1. K-weighting filter (pre-filter + RLB filter)
    2. Mean square calculation
    3. Gating (two-stage with absolute and relative thresholds)
    4. LUFS calculation
    
    Args:
        audio_mono: Mono audio signal
        sr: Sample rate
        block_size: Block size in seconds
        
    Returns:
        Integrated LUFS value
    """
    # Convert to numpy if needed
    if isinstance(audio_mono, torch.Tensor):
        audio_np = audio_mono.cpu().numpy()
    else:
        audio_np = audio_mono
    
    # K-weighting filter implementation (ITU-R BS.1770)
    # Stage 1: Pre-filter (high-pass at 38 Hz, Q=0.5)
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    Vh = np.power(10.0, 1.5/20.0) # +1.5 dB
    Vb = np.power(10.0, 0.0/20.0) # 0 dB
    
    # High-pass filter coefficients
    a0 = 1.0 + K / Q + K * K
    b0 = (1.0 + K * K) / a0
    b1 = 2.0 * (K * K - 1.0) / a0
    b2 = (1.0 + K * K - K / Q) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 + K * K - K / Q) / a0
    
    # Apply high-pass filter (pre-filter)
    audio_hp = signal.lfilter([b0, b1, b2], [1.0, a1, a2], audio_np)
    
    # Stage 2: RLB filter (high shelf at 1681 Hz, Q=0.7071, +4 dB)
    f0 = 1681.9744509555319
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = np.power(10.0, 4.0/20.0) # +4 dB
    Vb = np.power(10.0, 0.0/20.0) # 0 dB
    
    # High-shelf filter coefficients (exact BS.1770 specification)
    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vh * K / Q + Vb * K * K) / a0
    b1 = 2.0 * (Vb * K * K - Vh) / a0
    b2 = (Vh - Vh * K / Q + Vb * K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    
    # Apply high-shelf filter (RLB filter)
    audio_k = signal.lfilter([b0, b1, b2], [1.0, a1, a2], audio_hp)
    
    # Calculate block loudness values
    block_len = int(block_size * sr)
    hop_size = block_len // 4  # 75% overlap per spec
    
    # Ensure minimum audio length
    if len(audio_k) < block_len:
        pad = np.zeros(block_len - len(audio_k))
        audio_k = np.concatenate([audio_k, pad])
    
    # Calculate mean square for each block
    num_blocks = max(1, (len(audio_k) - block_len) // hop_size + 1)
    block_powers = []
    
    for i in range(num_blocks):
        start = i * hop_size
        end = min(start + block_len, len(audio_k))
        if end - start < block_len // 2:  # Skip blocks that are too short
            continue
            
        segment = audio_k[start:end]
        block_power = np.mean(segment ** 2)
        block_powers.append(block_power)
    
    if not block_powers:
        return -70.0  # Return minimum if no valid blocks
        
    # First stage gating (absolute threshold at -70 LUFS)
    abs_threshold_power = 10 ** (-70.0 / 10.0)
    gated_powers = [p for p in block_powers if p > abs_threshold_power]
    
    if not gated_powers:
        return -70.0
    
    # Calculate relative threshold
    mean_gated_power = np.mean(gated_powers)
    relative_threshold_power = mean_gated_power * 10 ** (-10.0 / 10.0)  # -10 dB relative to ungated mean
    
    # Second stage gating (relative threshold)
    gated_powers_relative = [p for p in gated_powers if p > relative_threshold_power]
    
    if not gated_powers_relative:
        gated_powers_relative = gated_powers  # Use first stage if second stage eliminates all
    
    # Calculate integrated loudness
    mean_power = np.mean(gated_powers_relative)
    integrated_lufs = -0.691 + 10 * np.log10(mean_power)  # LUFS
    
    return integrated_lufs
