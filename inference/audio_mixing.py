import os
import math
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np

# Basic Mixing Improvements
def mix_tracks(vocal, instrumental, vocal_gain=1.0, instrumental_gain=0.8):
    """
    Mix tracks with independent gain control
    
    Args:
        vocal: Vocal track tensor
        instrumental: Instrumental track tensor
        vocal_gain: Gain factor for vocals (1.0 = 0dB)
        instrumental_gain: Gain factor for instrumentals
        
    Returns:
        Mixed audio with proper gain staging
    """
    # Apply gains
    vocal_scaled = vocal * vocal_gain
    instrumental_scaled = instrumental * instrumental_gain
    
    # Mix with proper normalization to avoid clipping
    mix = (vocal_scaled + instrumental_scaled) / (vocal_gain + instrumental_gain)
    
    return mix

def apply_gain_staging(audio, target_peak=-6.0):
    """
    Apply gain staging to achieve target peak level
    
    Args:
        audio: Audio tensor
        target_peak: Target peak level in dB
        
    Returns:
        Gain-staged audio
    """
    # Convert dB to linear
    target_linear = 10 ** (target_peak / 20.0)
    
    # Measure current peak
    current_peak = audio.abs().max()
    
    # Calculate gain factor
    gain = target_linear / current_peak
    
    # Apply gain
    audio_staged = audio * gain
    
    return audio_staged

def measure_lufs(audio, sr=44100, block_size=0.4):
    """
    Accurate LUFS measurement using pyloudnorm implementation of ITU-R BS.1770 standard
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        block_size: Analysis block size in seconds (used for short-term LUFS)
        
    Returns:
        LUFS value (integrated loudness)
    """
    # Import pyloudnorm inside the function to avoid dependency issues if not installed
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("Warning: pyloudnorm not installed. Using fallback LUFS measurement.")
        return _measure_lufs_fallback(audio, sr, block_size)
    
    # Convert torch tensor to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Convert to mono if stereo
    if audio_np.ndim > 1 and audio_np.shape[0] > 1:
        audio_np = audio_np.mean(axis=0)
    elif audio_np.ndim > 1:
        # Handle case where first dimension is 1 (mono but with channel dimension)
        audio_np = audio_np.squeeze(0)
    
    # Ensure audio is the right shape for pyloudnorm (1D array)
    if audio_np.ndim != 1:
        raise ValueError(f"Audio should be 1D after conversion, got shape {audio_np.shape}")
    
    # Create a meter based on the sample rate
    meter = pyln.Meter(sr)
    
    # Measure integrated loudness (overall LUFS)
    try:
        # Meter requires a certain minimum number of samples for accurate measurement
        # Add silence padding if audio is too short
        min_length = int(sr * 0.4)  # 400ms minimum for accurate measurement
        if len(audio_np) < min_length:
            padding = np.zeros(min_length - len(audio_np))
            audio_np = np.concatenate([audio_np, padding])
        
        lufs = meter.integrated_loudness(audio_np)
        return lufs
    except Exception as e:
        print(f"Error measuring LUFS: {e}")
        return _measure_lufs_fallback(audio, sr, block_size)

def _measure_lufs_fallback(audio, sr=44100, block_size=0.4):
    """
    Simplified LUFS measurement (approximation) as a fallback
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        block_size: Analysis block size in seconds
        
    Returns:
        Approximate LUFS value
    """
    # Convert to mono if stereo
    if isinstance(audio, torch.Tensor) and audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    
    # Apply K-weighting filter (simplified)
    # Note: A proper implementation would use precise filters
    highpass = apply_highpass(audio, 60, sr)
    weighted = apply_high_shelf(highpass, 1500, 4.0, sr)
    
    # Segment into blocks
    block_samples = int(block_size * sr)
    num_blocks = max(1, audio.shape[-1] // block_samples)
    
    # Measure gated loudness (simplified)
    block_loudness = []
    for i in range(num_blocks):
        if i * block_samples + block_samples <= audio.shape[-1]:
            block = weighted[i * block_samples:(i + 1) * block_samples]
            energy = torch.mean(block ** 2)
            block_loudness.append(energy.item())
    
    # Apply gating (simplified)
    if block_loudness:
        mean_energy = sum(block_loudness) / len(block_loudness)
        gated_loudness = [l for l in block_loudness if l > mean_energy * 0.1]
        if gated_loudness:
            gated_mean = sum(gated_loudness) / len(gated_loudness)
            lufs = -0.691 + 10 * math.log10(gated_mean)
            return lufs
    
    # Fallback if no blocks or all blocks gated out
    return -30.0

def apply_highpass(audio, freq, sr):
    """
    Apply highpass filter using more efficient vectorized operations
    
    Args:
        audio: Audio tensor
        freq: Cutoff frequency
        sr: Sample rate
        
    Returns:
        Filtered audio
    """
    # Compute filter coefficients
    dt = 1.0 / sr
    RC = 1.0 / (2.0 * math.pi * freq)
    alpha = RC / (RC + dt)
    
    # Create output tensor
    y = torch.zeros_like(audio)
    
    # Apply filter (first sample remains zero)
    if len(audio) > 1:
        # Use vectorized operations for better performance
        y[1:] = alpha * (y[:-1] + audio[1:] - audio[:-1])
    
    return y

def apply_high_shelf(audio, freq, gain_db, sr):
    """
    Apply high shelf filter with proper coefficient initialization
    
    Args:
        audio: Audio tensor
        freq: Center frequency
        gain_db: Gain in dB
        sr: Sample rate
        
    Returns:
        Filtered audio
    """
    # Convert gain to linear
    gain = 10 ** (gain_db / 20.0)
    
    # Filter parameters
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / 2
    
    # Calculate filter coefficients
    b0 = (gain + 1) + (gain - 1) * math.cos(w0) + 2 * math.sqrt(gain) * alpha
    b1 = -2 * ((gain - 1) + (gain + 1) * math.cos(w0))
    b2 = (gain + 1) + (gain - 1) * math.cos(w0) - 2 * math.sqrt(gain) * alpha
    a0 = (gain + 1) - (gain - 1) * math.cos(w0) + 2 * math.sqrt(gain) * alpha
    a1 = 2 * ((gain - 1) - (gain + 1) * math.cos(w0))
    a2 = (gain + 1) - (gain - 1) * math.cos(w0) - 2 * math.sqrt(gain) * alpha
    
    # Normalize coefficients
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    
    # Apply filter
    y = torch.zeros_like(audio)
    
    # Initialize state variables
    x1, x2, y1, y2 = 0, 0, 0, 0
    
    # Process samples
    for i in range(len(audio)):
        x0 = audio[i].item() if hasattr(audio[i], 'item') else audio[i]
        y[i] = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        
        # Update state
        x2, x1 = x1, x0
        y2, y1 = y1, y[i].item() if hasattr(y[i], 'item') else y[i]
    
    return y

def normalize_before_mixing(vocal, instrumental, target_lufs=-16.0, sr=44100):
    """
    Normalize tracks to consistent loudness before mixing
    
    Args:
        vocal: Vocal track tensor
        instrumental: Instrumental track tensor
        target_lufs: Target loudness in LUFS
        sr: Sample rate
        
    Returns:
        Normalized vocal and instrumental tracks
    """
    # Calculate LUFS for each track
    vocal_lufs = measure_lufs(vocal, sr)
    instrumental_lufs = measure_lufs(instrumental, sr)
    
    # Calculate gain adjustments
    vocal_gain = 10 ** ((target_lufs - vocal_lufs) / 20.0)
    instrumental_gain = 10 ** ((target_lufs - instrumental_lufs) / 20.0)
    
    # Apply normalization
    vocal_normalized = vocal * vocal_gain
    instrumental_normalized = instrumental * instrumental_gain
    
    return vocal_normalized, instrumental_normalized

# Phase Alignment
def align_phases(reference, target, fft_size=2048, hop_size=512):
    """
    Align the phase of the target signal to match the reference signal
    using phase vocoder techniques.
    
    Args:
        reference: Reference audio signal (what we want to align to)
        target: Target audio signal (what we want to align)
        fft_size: FFT size for STFT
        hop_size: Hop size for STFT
        
    Returns:
        Phase-aligned version of the target signal
    """
    # Convert to mono for phase analysis if stereo
    if reference.dim() > 1 and reference.shape[0] > 1:
        ref_mono = reference.mean(dim=0)
    else:
        ref_mono = reference.squeeze(0) if reference.dim() > 1 else reference
        
    if target.dim() > 1 and target.shape[0] > 1:
        # Process each channel separately for stereo
        channels = []
        for ch in range(target.shape[0]):
            aligned_channel = align_single_channel(ref_mono, target[ch], fft_size, hop_size)
            channels.append(aligned_channel)
        return torch.stack(channels)
    else:
        target_mono = target.squeeze(0) if target.dim() > 1 else target
        return align_single_channel(ref_mono, target_mono, fft_size, hop_size).unsqueeze(0)

def align_single_channel(reference, target, fft_size, hop_size):
    """
    Align a single audio channel
    
    Args:
        reference: Reference audio (mono)
        target: Target audio (mono)
        fft_size: FFT size
        hop_size: Hop size
        
    Returns:
        Phase-aligned audio
    """
    # Ensure same length
    min_length = min(reference.shape[-1], target.shape[-1])
    reference = reference[..., :min_length]
    target = target[..., :min_length]
    
    # Make window
    window = torch.hann_window(fft_size)
    if torch.cuda.is_available():
        window = window.to(reference.device)
    
    # Compute STFTs
    ref_stft = torch.stft(reference, fft_size, hop_size, window=window, 
                          return_complex=True)
    target_stft = torch.stft(target, fft_size, hop_size, window=window, 
                             return_complex=True)
    
    # Extract magnitudes and phases
    ref_mag = torch.abs(ref_stft)
    ref_phase = torch.angle(ref_stft)
    target_mag = torch.abs(target_stft)
    
    # Create new STFT with target magnitude but reference phase
    aligned_stft = torch.polar(target_mag, ref_phase)
    
    # Convert back to time domain
    aligned_signal = torch.istft(aligned_stft, fft_size, hop_size, 
                                window=window, length=min_length)
    
    return aligned_signal

def find_time_offset(reference, target, max_offset_ms=100, sr=44100):
    """
    Find the optimal time offset between reference and target signals
    using cross-correlation.
    
    Args:
        reference: Reference audio signal
        target: Target audio signal
        max_offset_ms: Maximum offset to search in milliseconds
        sr: Sample rate
        
    Returns:
        Optimal sample offset (positive means target needs to be delayed)
    """
    max_offset_samples = int(sr * max_offset_ms / 1000)
    
    # Convert to mono for correlation
    if reference.dim() > 1 and reference.shape[0] > 1:
        ref_mono = reference.mean(dim=0)
    else:
        ref_mono = reference.squeeze(0) if reference.dim() > 1 else reference
        
    if target.dim() > 1 and target.shape[0] > 1:
        target_mono = target.mean(dim=0)
    else:
        target_mono = target.squeeze(0) if target.dim() > 1 else target
    
    # Ensure same length for correlation
    min_length = min(ref_mono.shape[-1], target_mono.shape[-1])
    ref_mono = ref_mono[..., :min_length]
    target_mono = target_mono[..., :min_length]
    
    # Compute cross-correlation
    correlation = torch.nn.functional.conv1d(
        ref_mono.unsqueeze(0).unsqueeze(0),
        target_mono.flip(0).unsqueeze(0).unsqueeze(0),
        padding=max_offset_samples
    )
    
    # Find peak correlation position
    _, peak_idx = torch.max(correlation, dim=2)
    offset = peak_idx.item() - max_offset_samples
    
    return offset

def apply_time_offset(audio, offset, mode='shift'):
    """
    Apply a time offset to audio.
    
    Args:
        audio: Audio tensor
        offset: Sample offset (positive = delay, negative = advance)
        mode: 'shift' or 'stretch' (stretch preserves length)
        
    Returns:
        Time-shifted audio
    """
    if offset == 0:
        return audio
    
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            shifted_channel = apply_time_offset(audio[ch], offset, mode)
            channels.append(shifted_channel)
        return torch.stack(channels)
    
    if mode == 'shift':
        # Simple shifting with zero-padding
        result = torch.zeros_like(audio)
        if offset > 0:
            # Delay
            result[offset:] = audio[:-offset]
        else:
            # Advance
            result[:offset] = audio[-offset:]
        return result
    else:
        # Placeholder for phase vocoder time stretching
        # For now, default to shift mode
        return apply_time_offset(audio, offset, 'shift')

def apply_bandpass(audio, low_freq, high_freq, sr=44100, order=4):
    """
    Apply bandpass filter to audio using proper IIR filters from scipy
    
    Args:
        audio: Audio tensor
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        sr: Sample rate
        order: Filter order (higher = steeper rolloff, but more CPU intensive)
        
    Returns:
        Filtered audio
    """
    try:
        from scipy import signal
    except ImportError:
        print("Warning: scipy not installed. Using fallback filter implementation.")
        return _apply_bandpass_fallback(audio, low_freq, high_freq, sr)
    
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            filtered_channel = apply_bandpass(audio[ch], low_freq, high_freq, sr, order)
            channels.append(filtered_channel)
        return torch.stack(channels)
    
    # Convert audio tensor to numpy array for scipy processing
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Prevent frequencies outside valid range
    nyquist = sr / 2
    low_freq = max(10, min(low_freq, nyquist - 10))  # Ensure at least 10Hz gap from DC and Nyquist
    high_freq = max(low_freq + 10, min(high_freq, nyquist - 10))
    
    # Design Butterworth bandpass filter
    sos = signal.butter(order, [low_freq, high_freq], btype='bandpass', 
                         fs=sr, output='sos')
    
    # Apply filter (zero-phase to preserve timing)
    filtered_audio = signal.sosfiltfilt(sos, audio_np)
    
    # Convert back to torch tensor
    filtered_tensor = torch.tensor(filtered_audio, dtype=audio.dtype, device=audio.device)
    
    return filtered_tensor

def _apply_bandpass_fallback(audio, low_freq, high_freq, sr=44100):
    """
    Simplified fallback bandpass filter implementation
    
    Args:
        audio: Audio tensor
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        sr: Sample rate
        
    Returns:
        Filtered audio
    """
    # Convert to frequencies to normalized frequency (0 to 1)
    nyquist = sr / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    # Create bandpass filter (simplified implementation)
    filter_length = 1024
    band_filter = torch.hamming_window(filter_length)
    
    # Create sinc filter (simplified)
    for i in range(filter_length):
        if i != filter_length // 2:  # Avoid division by zero
            # High-pass component
            high_component = torch.sin(torch.tensor(math.pi * low_normalized * (i - filter_length // 2))) / (math.pi * (i - filter_length // 2))
            # Low-pass component
            low_component = torch.sin(torch.tensor(math.pi * high_normalized * (i - filter_length // 2))) / (math.pi * (i - filter_length // 2))
            # Bandpass = low-pass - high-pass
            band_filter[i] *= (low_component - high_component)
    
    # Normalize filter
    band_filter /= band_filter.sum()
    
    # Apply filter using convolution
    audio_padded = torch.nn.functional.pad(audio, (filter_length//2, filter_length//2))
    filtered = torch.nn.functional.conv1d(
        audio_padded.view(1, 1, -1),
        band_filter.view(1, 1, -1),
        padding=0
    ).view(-1)
    
    return filtered

def multiband_phase_alignment(reference, target, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], sr=44100):
    """
    Align phases separately for different frequency bands.
    
    Args:
        reference: Reference audio signal
        target: Target audio signal
        bands: List of (low_freq, high_freq) tuples defining bands
        sr: Sample rate
        
    Returns:
        Phase-aligned version of target signal
    """
    # Handle multi-channel audio
    if reference.dim() > 1 and reference.shape[0] > 1:
        channels = []
        for ch in range(reference.shape[0]):
            if target.dim() > 1 and target.shape[0] > 1:
                aligned_channel = multiband_phase_alignment(reference[ch], target[ch], bands, sr)
            else:
                aligned_channel = multiband_phase_alignment(reference[ch], target, bands, sr)
            channels.append(aligned_channel)
        return torch.stack(channels)
    
    if target.dim() > 1 and target.shape[0] > 1:
        channels = []
        for ch in range(target.shape[0]):
            aligned_channel = multiband_phase_alignment(reference, target[ch], bands, sr)
            channels.append(aligned_channel)
        return torch.stack(channels)
    
    # Create filters for each band
    filtered_refs = []
    filtered_targets = []
    
    for low_freq, high_freq in bands:
        # Apply bandpass filters to isolate frequency bands
        ref_band = apply_bandpass(reference, low_freq, high_freq, sr)
        target_band = apply_bandpass(target, low_freq, high_freq, sr)
        
        filtered_refs.append(ref_band)
        filtered_targets.append(target_band)
    
    # Align each band separately
    aligned_bands = []
    for ref_band, target_band in zip(filtered_refs, filtered_targets):
        aligned_band = align_phases(ref_band, target_band)
        aligned_bands.append(aligned_band)
    
    # Recombine aligned bands
    result = sum(aligned_bands)
    
    return result

# Dynamic Processing
def apply_compression(audio, threshold=-20.0, ratio=2.0, attack=0.005, release=0.05, sr=44100):
    """
    Apply basic compression to audio using vectorized operations for better performance
    
    Args:
        audio: Audio tensor
        threshold: Threshold in dB
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate
        
    Returns:
        Compressed audio
    """
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            compressed_channel = apply_compression(audio[ch], threshold, ratio, attack, release, sr)
            channels.append(compressed_channel)
        return torch.stack(channels)
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate attack and release coefficients
    attack_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * attack))
    release_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * release))
    
    # Create vectorized version for better performance
    # First compute the signal envelope
    audio_abs = audio.abs()
    
    # Pre-allocate envelope and gain arrays
    envelope = torch.zeros_like(audio_abs)
    gain = torch.ones_like(audio_abs)
    
    # Initial envelope value
    envelope[0] = audio_abs[0]
    
    # Use optimized loops for envelope calculation
    # This part is difficult to fully vectorize due to its recursive nature
    for i in range(1, len(audio_abs)):
        if audio_abs[i] > envelope[i-1]:
            # Attack phase - signal is rising
            envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * audio_abs[i]
        else:
            # Release phase - signal is falling
            envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * audio_abs[i]
    
    # Calculate gain reduction vectorized
    above_threshold = envelope > threshold_linear
    gain[above_threshold] = (threshold_linear + (envelope[above_threshold] - threshold_linear) / ratio) / (envelope[above_threshold] + 1e-10)
    
    # Apply gain to audio signal
    output = audio * gain
    
    return output

def multi_band_compression(audio, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], 
                          thresholds=[-24, -18, -18, -16], ratios=[2.5, 2.0, 1.8, 1.5], sr=44100):
    """
    Apply multi-band compression
    
    Args:
        audio: Audio tensor
        bands: List of (low_freq, high_freq) tuples defining bands
        thresholds: Threshold for each band in dB
        ratios: Compression ratio for each band
        sr: Sample rate
        
    Returns:
        Multi-band compressed audio
    """
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            compressed_channel = multi_band_compression(audio[ch], bands, thresholds, ratios, sr)
            channels.append(compressed_channel)
        return torch.stack(channels)
    
    # Split into bands
    band_signals = []
    for low_freq, high_freq in bands:
        band_signal = apply_bandpass(audio, low_freq, high_freq, sr)
        band_signals.append(band_signal)
    
    # Compress each band
    compressed_bands = []
    for i, band_signal in enumerate(band_signals):
        compressed = apply_compression(band_signal, thresholds[i], ratios[i])
        compressed_bands.append(compressed)
    
    # Sum bands back together
    result = sum(compressed_bands)
    
    return result

def sidechain_compression(audio, sidechain_signal, threshold=-20.0, ratio=2.0, 
                          attack=0.005, release=0.05, sr=44100):
    """
    Apply sidechain compression to audio using vectorized operations for better performance
    
    Args:
        audio: Audio tensor to compress
        sidechain_signal: Control signal (typically vocals)
        threshold: Threshold in dB
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate
        
    Returns:
        Sidechained audio
    """
    # Handle multi-channel audio with single-channel sidechain
    if audio.dim() > 1:
        # If sidechain is mono but audio is stereo, use the same sidechain for both channels
        if sidechain_signal.dim() <= 1 or sidechain_signal.shape[0] == 1:
            sidechain_mono = sidechain_signal.squeeze(0) if sidechain_signal.dim() > 1 else sidechain_signal
            channels = []
            for ch in range(audio.shape[0]):
                processed_channel = sidechain_compression(audio[ch], sidechain_mono, threshold, ratio, attack, release, sr)
                channels.append(processed_channel)
            return torch.stack(channels)
        # If both are stereo, process channels separately
        else:
            channels = []
            for ch in range(audio.shape[0]):
                processed_channel = sidechain_compression(audio[ch], sidechain_signal[ch], threshold, ratio, attack, release, sr)
                channels.append(processed_channel)
            return torch.stack(channels)
            
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate attack and release coefficients
    attack_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * attack))
    release_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * release))
    
    # Make sure sidechain signal is the same length as audio
    # If sidechain is shorter, pad with zeros
    if len(sidechain_signal) < len(audio):
        padding = torch.zeros(len(audio) - len(sidechain_signal), device=sidechain_signal.device)
        sidechain_padded = torch.cat([sidechain_signal, padding])
    # If sidechain is longer, truncate
    elif len(sidechain_signal) > len(audio):
        sidechain_padded = sidechain_signal[:len(audio)]
    else:
        sidechain_padded = sidechain_signal
        
    # Compute the sidechain envelope
    sidechain_abs = sidechain_padded.abs()
    
    # Pre-allocate envelope and gain arrays
    envelope = torch.zeros_like(sidechain_abs)
    gain = torch.ones_like(sidechain_abs)
    
    # Initial envelope value
    envelope[0] = sidechain_abs[0]
    
    # Use optimized loops for envelope calculation
    # This part is difficult to fully vectorize due to its recursive nature
    for i in range(1, len(sidechain_abs)):
        if sidechain_abs[i] > envelope[i-1]:
            # Attack phase - signal is rising
            envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * sidechain_abs[i]
        else:
            # Release phase - signal is falling
            envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * sidechain_abs[i]
    
    # Calculate gain reduction vectorized
    above_threshold = envelope > threshold_linear
    gain[above_threshold] = (threshold_linear + (envelope[above_threshold] - threshold_linear) / ratio) / (envelope[above_threshold] + 1e-10)
    
    # Apply gain to audio signal (not the sidechain)
    output = audio * gain
    
    return output

# Stereo Processing
def enhance_stereo_width(audio, width=1.5):
    """
    Enhance stereo width
    
    Args:
        audio: Stereo audio tensor (2 channels)
        width: Width factor (1.0 = normal, > 1.0 = wider)
        
    Returns:
        Width-enhanced stereo audio
    """
    if audio.dim() <= 1 or audio.shape[0] < 2:
        # Convert mono to stereo first
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.repeat(2, 1)
    
    # Extract mid and side
    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2
    
    # Apply width adjustment to side
    side_enhanced = side * width
    
    # Recombine to stereo
    left = mid + side_enhanced
    right = mid - side_enhanced
    
    return torch.stack([left, right])

def pan_audio(audio, pan_position=0.0):
    """
    Pan audio in the stereo field
    
    Args:
        audio: Audio tensor (1 or 2 channels)
        pan_position: -1.0 (full left) to 1.0 (full right), 0.0 is center
        
    Returns:
        Panned audio (2 channels)
    """
    # Ensure stereo output
    if audio.dim() == 1 or audio.shape[0] == 1:
        # If mono, duplicate to stereo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.repeat(2, 1)
    
    # Calculate pan gains using constant power panning
    pan = torch.clamp(torch.tensor(pan_position), -1.0, 1.0)
    angle = (pan + 1.0) * math.pi / 4.0  # 0 to pi/2
    
    left_gain = math.cos(angle)
    right_gain = math.sin(angle)
    
    # Apply gains
    left = audio[0] * left_gain
    right = audio[1] * right_gain
    
    return torch.stack([left, right])

def apply_soft_clipper(audio, threshold=0.8, softness=0.1):
    """
    Apply a soft clipper to prevent harsh clipping
    
    Args:
        audio: Audio tensor
        threshold: Threshold level (0.0 to 1.0) where soft clipping begins
        softness: Softness of the transition (higher = smoother transition)
        
    Returns:
        Soft clipped audio
    """
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            clipped_channel = apply_soft_clipper(audio[ch], threshold, softness)
            channels.append(clipped_channel)
        return torch.stack(channels)
    
    # Create output tensor
    output = torch.zeros_like(audio)
    
    # Vectorized implementation of a hyperbolic tangent soft clipper
    # Apply soft clipping only to samples above threshold
    below_threshold = torch.abs(audio) <= threshold
    above_threshold = ~below_threshold
    
    # For samples below threshold, keep them unchanged
    output[below_threshold] = audio[below_threshold]
    
    # For samples above threshold, apply soft clipping
    # Formula: threshold + (1 - threshold) * tanh((x - threshold) / softness)
    x_above = audio[above_threshold]
    sign_x = torch.sign(x_above)
    abs_x = torch.abs(x_above)
    
    # Apply the soft clipping curve
    soft_clipped = threshold + (1 - threshold) * torch.tanh((abs_x - threshold) / softness)
    output[above_threshold] = sign_x * soft_clipped
    
    return output

# Integrated mixing function that applies multiple enhancements
def enhanced_audio_mix(vocal, instrumental, mix_params=None, sr=44100):
    """
    Apply comprehensive mixing enhancements to produce a professional-quality mix
    
    Args:
        vocal: Vocal track tensor
        instrumental: Instrumental track tensor
        mix_params: Dictionary of mixing parameters
        sr: Sample rate
        
    Returns:
        Enhanced audio mix
    """
    # Default parameters
    default_mix_params = {
        'vocal_gain': 1.0,
        'instrumental_gain': 0.8,
        'target_lufs': -16.0,
        'vocal_compression': {
            'threshold': -20.0,
            'ratio': 2.0,
            'attack': 0.005,
            'release': 0.05
        },
        'sidechain': {
            'enabled': True,
            'threshold': -24.0,
            'ratio': 2.5
        },
        'stereo_width': 1.2,
        'pan_position': 0.0,
        'phase_align': True,
        'soft_clip': {
            'enabled': True,
            'threshold': 0.8,
            'softness': 0.1
        }
    }
    
    # Use provided mix params or defaults
    if mix_params is None:
        mix_params = default_mix_params
    
    # 1. Normalize loudness
    target_lufs = mix_params.get('target_lufs', default_mix_params['target_lufs'])
    vocal_norm, inst_norm = normalize_before_mixing(vocal, instrumental, target_lufs, sr)
    
    # 2. Apply compression to vocal
    vocal_comp_params = mix_params.get('vocal_compression', default_mix_params['vocal_compression'])
    vocal_comp = apply_compression(
        vocal_norm, 
        vocal_comp_params.get('threshold', default_mix_params['vocal_compression']['threshold']), 
        vocal_comp_params.get('ratio', default_mix_params['vocal_compression']['ratio']),
        vocal_comp_params.get('attack', default_mix_params['vocal_compression']['attack']),
        vocal_comp_params.get('release', default_mix_params['vocal_compression']['release']),
        sr
    )
    
    # 3. Apply stereo width to instrumental
    stereo_width = mix_params.get('stereo_width', default_mix_params['stereo_width'])
    inst_width = enhance_stereo_width(inst_norm, stereo_width)
    
    # 4. Align phases if enabled
    phase_align = mix_params.get('phase_align', default_mix_params['phase_align'])
    if phase_align:
        inst_aligned = align_phases(vocal_comp, inst_width)
    else:
        inst_aligned = inst_width
    
    # 5. Apply sidechain compression to instrumental if enabled
    sidechain_params = mix_params.get('sidechain', default_mix_params['sidechain'])
    if sidechain_params.get('enabled', default_mix_params['sidechain']['enabled']):
        inst_sc = sidechain_compression(
            inst_aligned,
            vocal_comp,
            sidechain_params.get('threshold', default_mix_params['sidechain']['threshold']),
            sidechain_params.get('ratio', default_mix_params['sidechain']['ratio']),
            0.01,  # Slightly slower attack for sidechain
            0.1,   # Slightly longer release for sidechain
            sr
        )
    else:
        inst_sc = inst_aligned
    
    # 6. Pan vocals if needed
    pan_position = mix_params.get('pan_position', default_mix_params['pan_position'])
    if pan_position != 0.0:
        vocal_panned = pan_audio(vocal_comp, pan_position)
    else:
        vocal_panned = vocal_comp
    
    # 7. Mix with proper gain staging
    vocal_gain = mix_params.get('vocal_gain', default_mix_params['vocal_gain'])
    instrumental_gain = mix_params.get('instrumental_gain', default_mix_params['instrumental_gain'])
    final_mix = mix_tracks(
        vocal_panned, 
        inst_sc, 
        vocal_gain, 
        instrumental_gain
    )
    
    # 8. Apply gain staging
    final_mix = apply_gain_staging(final_mix, -0.3)  # Leave headroom
    
    # 9. Apply soft clipping to prevent harsh digital clipping
    soft_clip_params = mix_params.get('soft_clip', default_mix_params['soft_clip'])
    if soft_clip_params.get('enabled', default_mix_params['soft_clip']['enabled']):
        final_mix = apply_soft_clipper(
            final_mix,
            soft_clip_params.get('threshold', default_mix_params['soft_clip']['threshold']),
            soft_clip_params.get('softness', default_mix_params['soft_clip']['softness'])
        )
    
    return final_mix

# Utility functions for file-based processing
def process_files_with_enhancements(vocal_path, instrumental_path, output_path, mix_params=None):
    """
    Process audio files with mixing enhancements
    
    Args:
        vocal_path: Path to vocal file
        instrumental_path: Path to instrumental file
        output_path: Path for output file
        mix_params: Dictionary of mixing parameters
        
    Returns:
        Path to processed file
    """
    # Load audio files
    vocal, sr_v = torchaudio.load(vocal_path)
    instrumental, sr_i = torchaudio.load(instrumental_path)
    
    # Ensure consistent sample rate
    if sr_v != sr_i:
        resampler = Resample(orig_freq=sr_i, new_freq=sr_v)
        instrumental = resampler(instrumental)
        sr = sr_v
    else:
        sr = sr_v
    
    # Apply enhanced mixing
    mixed = enhanced_audio_mix(vocal, instrumental, mix_params, sr)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the result
    torchaudio.save(output_path, mixed, sr)
    
    return output_path 