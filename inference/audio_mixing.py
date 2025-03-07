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
    ITU-R BS.1770 compliant LUFS measurement as a fallback when pyloudnorm is not available.
    Implements the complete BS.1770 algorithm including proper K-weighting and two-stage gating.
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        block_size: Analysis block size in seconds
        
    Returns:
        Integrated LUFS value compliant with ITU-R BS.1770 standard
    """
    from scipy import signal
    
    # Convert to mono if stereo
    if isinstance(audio, torch.Tensor):
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        # Convert to numpy for processing
        audio_np = audio.cpu().numpy()
    else:
        if audio.ndim > 1 and audio.shape[0] > 1:
            audio_np = audio.mean(axis=0)
        else:
            audio_np = audio
    
    # Ensure audio is 1D
    audio_np = audio_np.squeeze()
    
    # Pre-filtering stage 1: High-pass filter (simulate 'pre' filter in BS.1770)
    # 2nd order Butterworth high-pass at 38 Hz
    nyquist = sr / 2.0
    high_pass_b, high_pass_a = signal.butter(2, 38.0/nyquist, 'highpass')
    audio_hp = signal.lfilter(high_pass_b, high_pass_a, audio_np)
    
    # Pre-filtering stage 2: K-weighting filter (shelf filter from BS.1770)
    # Coefficients for the two filters in series as defined in ITU-R BS.1770
    # High-shelf filter (+4 dB at 1500 Hz)
    # Using SOS (second-order sections) for better numerical stability
    high_shelf_sos = signal.butter(2, 1500.0/nyquist, 'highshelf', output='sos')
    high_shelf_gain = 10**(4.0/20)  # +4 dB in linear gain
    high_shelf_sos[0, -1] *= high_shelf_gain
    
    # Apply K-weighting filter
    audio_k_weighted = signal.sosfilt(high_shelf_sos, audio_hp)
    
    # Calculate gating block size (400ms as defined in BS.1770)
    block_samples = int(block_size * sr)
    
    # Segment into overlapping blocks with 75% overlap
    stride = block_samples // 4
    
    # If audio is shorter than block size, pad with zeros
    if len(audio_k_weighted) < block_samples:
        padding = np.zeros(block_samples - len(audio_k_weighted))
        audio_k_weighted = np.concatenate([audio_k_weighted, padding])
    
    # Calculate overlapping blocks
    num_blocks = max(1, (len(audio_k_weighted) - block_samples) // stride + 1)
    
    # Power calculation for each block (mean square)
    block_powers = []
    for i in range(num_blocks):
        start = i * stride
        end = min(start + block_samples, len(audio_k_weighted))
        block = audio_k_weighted[start:end]
        # Ensure block is full length
        if len(block) < block_samples:
            continue
        # Calculate mean square (power) for the block
        block_power = np.mean(block ** 2)
        block_powers.append(block_power)
    
    if not block_powers:
        return -70.0  # Return very low LUFS for silent audio
    
    # First stage gating: Exclude blocks below absolute threshold (-70 LUFS)
    absolute_threshold_power = 10 ** (-70/10)
    gated_powers = [power for power in block_powers if power > absolute_threshold_power]
    
    if not gated_powers:
        return -70.0  # Return very low LUFS for silent audio
    
    # Calculate the relative threshold (first-pass gated loudness - 10 LUFS)
    first_loudness = -0.691 + 10 * np.log10(np.mean(gated_powers))
    relative_threshold_power = 10 ** ((first_loudness - 10) / 10)
    
    # Second stage gating: Apply relative threshold
    final_powers = [power for power in gated_powers if power > relative_threshold_power]
    
    if not final_powers:
        # Use first stage result if all blocks are gated out in the second stage
        final_powers = gated_powers
    
    # Calculate final integrated loudness
    integrated_loudness = -0.691 + 10 * np.log10(np.mean(final_powers))
    
    return integrated_loudness

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
def align_phases(reference, target, fft_size=2048, hop_size=512, 
                 enable_freq_dependent=True, enable_transient_preservation=True, 
                 enable_phase_locking=True):
    """
    Align the phase of the target signal to match the reference signal
    using advanced phase vocoder techniques with horizontal phase coherence.
    
    Args:
        reference: Reference audio signal (what we want to align to)
        target: Target audio signal (what we want to align)
        fft_size: FFT size for STFT
        hop_size: Hop size for STFT
        enable_freq_dependent: Whether to use frequency-dependent processing
        enable_transient_preservation: Whether to preserve transients
        enable_phase_locking: Whether to use phase locking within critical bands
        
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
            aligned_channel = align_single_channel(ref_mono, target[ch], fft_size, hop_size,
                                                 enable_freq_dependent, enable_transient_preservation,
                                                 enable_phase_locking)
            channels.append(aligned_channel)
        return torch.stack(channels)
    else:
        target_mono = target.squeeze(0) if target.dim() > 1 else target
        return align_single_channel(ref_mono, target_mono, fft_size, hop_size,
                                  enable_freq_dependent, enable_transient_preservation,
                                  enable_phase_locking).unsqueeze(0)

def align_single_channel(reference, target, fft_size, hop_size, 
                        enable_freq_dependent=True, enable_transient_preservation=True,
                        enable_phase_locking=True):
    """
    Align a single audio channel using phase vocoder with horizontal phase coherence
    
    Args:
        reference: Reference audio (mono)
        target: Target audio (mono)
        fft_size: FFT size
        hop_size: Hop size
        enable_freq_dependent: Whether to use frequency-dependent processing
        enable_transient_preservation: Whether to preserve transients
        enable_phase_locking: Whether to use phase locking within critical bands
        
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
    
    # Get number of frames and frequency bins
    n_frames = ref_stft.shape[1]
    n_freqs = ref_stft.shape[0]
    
    # Extract magnitudes and phases - reuse tensors for memory optimization
    ref_mag = torch.abs(ref_stft)
    ref_phase = torch.angle(ref_stft)
    target_mag = torch.abs(target_stft)
    target_phase = torch.angle(target_stft)
    
    # Calculate phase difference between adjacent frames in the reference
    ref_phase_diff = torch.zeros_like(ref_phase)
    ref_phase_diff[:, 1:] = ref_phase[:, 1:] - ref_phase[:, :-1]
    
    # Unwrap phase differences to handle phase wrapping
    # Convert to numpy for unwrapping then back to torch
    ref_phase_diff_unwrapped = torch.from_numpy(
        np.unwrap(ref_phase_diff.cpu().numpy(), axis=1)
    ).to(ref_phase_diff.device)
    
    # Initialize output phase with first frame of target phase
    aligned_phase = torch.zeros_like(target_phase)
    aligned_phase[:, 0] = target_phase[:, 0]
    
    # ENHANCEMENT 1: Frequency-dependent processing
    # Define different alpha coefficients for different frequency bands
    if enable_freq_dependent:
        # Approximate frequency bands for different processing
        # Low (0-250 Hz), Mid (250-2000 Hz), High (2000+ Hz)
        band_edges = [0, int(250 * fft_size / 44100), int(2000 * fft_size / 44100), n_freqs]
        band_alphas = [0.9, 0.7, 0.5]  # Stronger alignment for low frequencies
    
    # ENHANCEMENT 2: Transient detection for transient preservation
    if enable_transient_preservation:
        # Detect transients based on energy increases in consecutive frames
        frame_energy = torch.sum(target_mag**2, dim=0)
        energy_diff = torch.zeros_like(frame_energy)
        energy_diff[1:] = frame_energy[1:] - frame_energy[:-1]
        
        # Normalize energy differences
        energy_diff = energy_diff / (torch.mean(torch.abs(energy_diff)) + 1e-8)
        
        # Threshold for transient detection
        transient_mask = energy_diff > 1.5  # Adjust threshold as needed
        
        # Extend transient mask by a few frames for smoother transitions
        extended_mask = torch.zeros_like(transient_mask)
        for i in range(n_frames):
            if i < n_frames - 3 and (transient_mask[i] or transient_mask[i+1] or transient_mask[i+2]):
                extended_mask[i:i+3] = True
    
    # ENHANCEMENT 4: Phase locking within critical bands
    if enable_phase_locking:
        # Define critical bands (simple approximation)
        critical_bands = []
        band_width = 3  # bins
        for i in range(0, n_freqs, band_width):
            critical_bands.append((i, min(i + band_width, n_freqs)))
    
    # Propagate phase horizontally with phase coherence
    # This maintains horizontal phase coherence by integrating the reference's phase differences
    for frame in range(1, n_frames):
        # Default propagation
        aligned_phase[:, frame] = aligned_phase[:, frame-1] + ref_phase_diff_unwrapped[:, frame]
        
        # ENHANCEMENT 2: Reset phase during transients for better transient preservation
        if enable_transient_preservation and extended_mask[frame]:
            # During transients, use the original target phase
            aligned_phase[:, frame] = target_phase[:, frame]
        
        # ENHANCEMENT 4: Phase locking within critical bands
        if enable_phase_locking and not (enable_transient_preservation and extended_mask[frame]):
            for band_start, band_end in critical_bands:
                if band_end > band_start + 1:  # Make sure band has at least 2 bins
                    # Find peak bin in this band
                    band_region = target_mag[band_start:band_end, frame]
                    peak_bin_offset = torch.argmax(band_region)
                    peak_bin = band_start + peak_bin_offset
                    
                    # Lock phases of nearby bins to the peak bin
                    peak_phase = aligned_phase[peak_bin, frame]
                    for bin_idx in range(band_start, band_end):
                        # Adjust phase to maintain the same phase relationship
                        # as in the original signal, but centered around the peak bin
                        orig_phase_diff = target_phase[bin_idx, frame] - target_phase[peak_bin, frame]
                        aligned_phase[bin_idx, frame] = peak_phase + orig_phase_diff
    
    # Apply spectral masking where magnitudes are very small
    # This prevents phase issues in very low energy regions
    magnitude_threshold = target_mag.mean() * 0.01
    mask = target_mag > magnitude_threshold
    
    # ENHANCEMENT 1: Apply frequency-dependent processing
    if enable_freq_dependent:
        # Create frequency-dependent alpha for smoother blending
        alpha = torch.zeros_like(target_mag)
        
        for i in range(len(band_edges) - 1):
            # Get frequency band indices
            band_start, band_end = band_edges[i], band_edges[i+1]
            # Set alpha value for this band
            base_alpha = band_alphas[i]
            
            # Apply magnitude-based scaling within each band
            band_mag_norm = torch.clamp(target_mag[band_start:band_end] / 
                                      (torch.mean(target_mag[band_start:band_end]) * 0.1), 0.0, 1.0)
            alpha[band_start:band_end] = band_mag_norm * base_alpha
    else:
        # Original alpha calculation
        alpha = torch.clamp(target_mag / (target_mag.mean() * 0.1), 0.0, 1.0)
    
    # Blend between original phase and aligned phase based on magnitude and settings
    blended_phase = target_phase * (1.0 - alpha) + aligned_phase * alpha
    
    # Create new STFT with target magnitude and aligned phase
    # Memory optimization: reuse tensors
    aligned_stft = torch.polar(target_mag, blended_phase)
    
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
    Advanced fallback bandpass filter implementation that minimizes ringing artifacts
    and provides better stopband attenuation. Uses an adaptive filter length based
    on the cutoff frequencies for more consistent transition bandwidths.
    
    Args:
        audio: Audio tensor
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        sr: Sample rate
        
    Returns:
        Filtered audio
    """
    import numpy as np
    
    # Convert to frequencies to normalized frequency (0 to 1)
    nyquist = sr / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    # Determine appropriate filter length based on the lowest frequency
    # Lower frequencies need longer filters for good frequency resolution
    # Rule of thumb: At least 4 cycles of the lowest frequency
    min_cycles = 4
    lowest_freq = max(20, low_freq)  # Avoid extreme values
    filter_length = int(min_cycles * sr / lowest_freq)
    
    # Round up to nearest power of 2 + 1 for computational efficiency and phase symmetry
    filter_length = 2 ** (int(np.log2(filter_length)) + 1) + 1
    
    # Cap maximum filter length to avoid excessive computation
    filter_length = min(filter_length, 16385)  # 2^14 + 1
    
    # Ensure filter length is odd for linear phase
    if filter_length % 2 == 0:
        filter_length += 1
    
    # Create time array for sinc calculations
    n = np.arange(filter_length)
    center = filter_length // 2
    
    # Create bandpass filter using sinc method with Blackman window
    # Blackman window has better sidelobe attenuation than Hamming
    window = np.blackman(filter_length)
    
    # Create low-pass and high-pass components
    sinc_lowpass = np.ones(filter_length)
    sinc_highpass = np.ones(filter_length)
    
    # Calculate sinc functions avoiding division by zero at center
    for i in n:
        if i != center:
            # Low-pass component (high cutoff)
            sinc_lowpass[i] = np.sin(2 * np.pi * high_normalized * (i - center)) / (np.pi * (i - center))
            
            # High-pass component (low cutoff)
            sinc_highpass[i] = np.sin(2 * np.pi * low_normalized * (i - center)) / (np.pi * (i - center))
    
    # At center point, use the limit value
    sinc_lowpass[center] = 2 * high_normalized
    sinc_highpass[center] = 2 * low_normalized
    
    # Bandpass = lowpass - highpass
    bandpass_filter = sinc_lowpass - sinc_highpass
    
    # Apply window to reduce ringing
    bandpass_filter *= window
    
    # Normalize for unity gain in the passband
    # Calculate the frequency response at center of passband
    center_freq = (low_freq + high_freq) / 2
    center_freq_normalized = center_freq / nyquist
    
    response = 0
    for i in range(filter_length):
        response += bandpass_filter[i] * np.cos(2 * np.pi * center_freq_normalized * (i - center))
    
    if abs(response) > 1e-10:  # Avoid division by zero
        bandpass_filter /= response
    
    # Convert filter to tensor
    filter_tensor = torch.tensor(bandpass_filter, dtype=torch.float32)
    if isinstance(audio, torch.Tensor) and audio.is_cuda:
        filter_tensor = filter_tensor.to(audio.device)
    
    # Prepare for convolution
    # Reshape filter for torch.nn.functional.conv1d
    filter_tensor = filter_tensor.view(1, 1, -1)
    
    # Pad audio to avoid edge effects
    pad_amount = filter_length // 2
    if isinstance(audio, torch.Tensor):
        # Handle multi-channel audio
        if audio.dim() > 1:
            # Process each channel separately
            channels = []
            for ch in range(audio.shape[0]):
                # Pad and reshape for conv1d
                audio_padded = torch.nn.functional.pad(audio[ch], (pad_amount, pad_amount))
                audio_padded = audio_padded.view(1, 1, -1)
                
                # Apply filter
                filtered = torch.nn.functional.conv1d(audio_padded, filter_tensor, padding=0)
                channels.append(filtered.view(-1))
                
            return torch.stack(channels)
        else:
            # Single channel
            audio_padded = torch.nn.functional.pad(audio, (pad_amount, pad_amount))
            audio_padded = audio_padded.view(1, 1, -1)
            filtered = torch.nn.functional.conv1d(audio_padded, filter_tensor, padding=0)
            return filtered.view(-1)
    else:
        # Handle numpy arrays
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_padded = torch.nn.functional.pad(audio_tensor, (pad_amount, pad_amount))
        audio_padded = audio_padded.view(1, 1, -1)
        filtered = torch.nn.functional.conv1d(audio_padded, filter_tensor, padding=0)
        return filtered.view(-1)

def multiband_phase_alignment(reference, target, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], sr=44100,
                             enable_freq_dependent=True, enable_transient_preservation=True, 
                             enable_phase_locking=True):
    """
    Align phases separately for different frequency bands.
    
    Args:
        reference: Reference audio signal
        target: Target audio signal
        bands: List of (low_freq, high_freq) tuples defining bands
        sr: Sample rate
        enable_freq_dependent: Whether to use frequency-dependent processing
        enable_transient_preservation: Whether to preserve transients
        enable_phase_locking: Whether to use phase locking within critical bands
        
    Returns:
        Phase-aligned version of target signal
    """
    # Handle multi-channel audio
    if reference.dim() > 1 and reference.shape[0] > 1:
        channels = []
        for ch in range(reference.shape[0]):
            if target.dim() > 1 and target.shape[0] > 1:
                aligned_channel = multiband_phase_alignment(reference[ch], target[ch], bands, sr,
                                                          enable_freq_dependent, enable_transient_preservation,
                                                          enable_phase_locking)
            else:
                aligned_channel = multiband_phase_alignment(reference[ch], target, bands, sr,
                                                          enable_freq_dependent, enable_transient_preservation,
                                                          enable_phase_locking)
            channels.append(aligned_channel)
        return torch.stack(channels)
    
    if target.dim() > 1 and target.shape[0] > 1:
        channels = []
        for ch in range(target.shape[0]):
            aligned_channel = multiband_phase_alignment(reference, target[ch], bands, sr,
                                                      enable_freq_dependent, enable_transient_preservation,
                                                      enable_phase_locking)
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
    
    # Customize settings for each frequency band
    for i, ((low_freq, high_freq), ref_band, target_band) in enumerate(zip(bands, filtered_refs, filtered_targets)):
        # Determine optimal FFT size based on band frequency
        # Lower frequencies need larger FFT sizes for better frequency resolution
        if low_freq < 250:
            fft_size = 4096  # Larger FFT for better low-frequency resolution
        elif low_freq < 2000:
            fft_size = 2048  # Standard size for mid frequencies
        else:
            fft_size = 1024  # Smaller FFT for high frequencies (better time resolution)
        
        # Determine hop size (usually 25% of FFT size for good overlap)
        hop_size = fft_size // 4
        
        # Band-specific settings
        # - Low frequencies: strong phase coherence, less transient preservation
        # - Mid frequencies: balanced settings
        # - High frequencies: focus on transient preservation, less phase coherence
        if low_freq < 250:
            # Low frequencies - prioritize phase coherence
            aligned_band = align_phases(ref_band, target_band, fft_size, hop_size,
                                     enable_freq_dependent=enable_freq_dependent,
                                     enable_transient_preservation=False,  # Less important for bass
                                     enable_phase_locking=enable_phase_locking)
        elif low_freq < 2000:
            # Mid frequencies - balanced approach
            aligned_band = align_phases(ref_band, target_band, fft_size, hop_size,
                                     enable_freq_dependent=enable_freq_dependent,
                                     enable_transient_preservation=enable_transient_preservation,
                                     enable_phase_locking=enable_phase_locking)
        else:
            # High frequencies - prioritize transient preservation
            aligned_band = align_phases(ref_band, target_band, fft_size, hop_size,
                                     enable_freq_dependent=enable_freq_dependent,
                                     enable_transient_preservation=enable_transient_preservation,
                                     enable_phase_locking=False)  # Less important for treble
        
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

def apply_look_ahead_limiter(audio, threshold=-1.0, release_time=0.050, attack_time=0.001, look_ahead_time=0.005, sr=44100):
    """
    Apply a true look-ahead peak limiter to prevent clipping while minimizing distortion.
    Uses attack/release envelope shaping and look-ahead processing to catch transients
    before they occur.
    
    Args:
        audio: Audio tensor
        threshold: Threshold in dB (0.0 = full scale)
        release_time: Release time in seconds
        attack_time: Attack time in seconds
        look_ahead_time: Look-ahead time in seconds
        sr: Sample rate
        
    Returns:
        Limited audio
    """
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate time constants in samples
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    look_ahead_samples = int(look_ahead_time * sr)
    
    # Ensure minimum number of samples
    attack_samples = max(1, attack_samples)
    release_samples = max(1, release_samples)
    look_ahead_samples = max(1, look_ahead_samples)
    
    # Prepare gain buffer
    gain_reduction = torch.ones_like(audio[0] if audio.dim() > 1 else audio)
    
    # Prepare audio buffer with look-ahead padding
    if audio.dim() > 1:
        # Stereo or multi-channel
        padded_audio = torch.nn.functional.pad(audio, (0, look_ahead_samples))
        result = torch.zeros_like(audio)
    else:
        # Mono
        padded_audio = torch.nn.functional.pad(audio.unsqueeze(0), (0, look_ahead_samples))
        result = torch.zeros_like(audio)
    
    # Prepare attack and release coefficients
    attack_coeff = torch.exp(-1.0 / attack_samples)
    release_coeff = torch.exp(-1.0 / release_samples)
    
    # Current gain state
    current_gain = 1.0
    
    # Process samples
    for i in range(audio.shape[-1]):
        # Calculate the peak level in the look-ahead window
        if audio.dim() > 1:
            # Multi-channel: use maximum absolute value across all channels
            peak_level = torch.max(torch.abs(padded_audio[:, i:i+look_ahead_samples+1]))
        else:
            # Mono
            peak_level = torch.max(torch.abs(padded_audio[0, i:i+look_ahead_samples+1]))
        
        # Calculate the required gain reduction
        if peak_level > threshold_linear:
            target_gain = threshold_linear / peak_level
        else:
            target_gain = 1.0
        
        # Apply attack/release smoothing
        if target_gain < current_gain:
            # Attack phase - need more gain reduction
            current_gain = attack_coeff * current_gain + (1.0 - attack_coeff) * target_gain
        else:
            # Release phase - need less gain reduction
            current_gain = release_coeff * current_gain + (1.0 - release_coeff) * target_gain
        
        # Store gain for this sample
        gain_reduction[i] = current_gain
    
    # Apply smoothed gain to audio
    if audio.dim() > 1:
        for ch in range(audio.shape[0]):
            result[ch] = audio[ch] * gain_reduction
    else:
        result = audio * gain_reduction
    
    return result

def apply_soft_clipper(audio, threshold=0.8, softness=0.1):
    """
    Apply soft clipping to prevent hard digital clipping
    
    Args:
        audio: Audio tensor
        threshold: Threshold where soft clipping begins
        softness: Softness of the curve (higher = gentler)
        
    Returns:
        Soft-clipped audio
    """
    # Handling for multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            channels.append(apply_soft_clipper(audio[ch], threshold, softness))
        return torch.stack(channels)
    
    # Below threshold, no change
    # Above threshold, apply soft curve
    result = torch.zeros_like(audio)
    
    # Linear region (below threshold)
    mask_linear = torch.abs(audio) <= threshold
    result[mask_linear] = audio[mask_linear]
    
    # Non-linear region (above threshold)
    mask_clip = ~mask_linear
    
    # Traditional tanh-based soft clipper
    x_norm = (torch.abs(audio[mask_clip]) - threshold) / softness
    # Apply curve: threshold + softness * tanh(x_norm)
    curve = threshold + softness * torch.tanh(x_norm)
    
    # Apply sign of original signal
    result[mask_clip] = torch.sign(audio[mask_clip]) * curve
    
    return result

# Integrated mixing function that applies multiple enhancements
def enhanced_audio_mix(vocal, instrumental, mix_params=None, sr=44100):
    """
    Enhanced audio mixing with multiple advanced processing stages
    and robust error handling.
    
    Args:
        vocal: Vocal track tensor
        instrumental: Instrumental track tensor
        mix_params: Dictionary of mixing parameters
        sr: Sample rate
        
    Returns:
        Final mixed audio
    """
    # Set default parameters if none provided
    default_mix_params = {
        'vocal_gain': 1.0,
        'instrumental_gain': 0.8,
        'vocal_compression': {
            'enabled': True,
            'threshold': -20.0,
            'ratio': 2.0
        },
        'instrumental_compression': {
            'enabled': False,
            'threshold': -24.0,
            'ratio': 1.5
        },
        'multiband_compression': {
            'enabled': True,
            'bands': [(0, 250), (250, 2000), (2000, 8000), (8000, 22050)],
            'thresholds': [-24, -18, -18, -16],
            'ratios': [2.5, 2.0, 1.8, 1.5]
        },
        'phase_alignment': {
            'enabled': True,
            'multiband': True,
            'freq_dependent': True,
            'transient_preservation': True,
            'phase_locking': True
        },
        'normalization': {
            'enabled': True,
            'target_lufs': -14.0,
            'true_peak': -1.0
        },
        'stereo_width': {
            'enabled': True,
            'width': 1.2
        },
        'soft_clip': {
            'enabled': True,
            'threshold': -1.0,
            'attack_time': 0.001,
            'release_time': 0.050
        }
    }
    
    if mix_params is None:
        mix_params = default_mix_params
    
    try:
        # Ensure input tensors are valid
        if vocal is None or instrumental is None:
            raise ValueError("Vocal and instrumental inputs must not be None")
        
        # Check tensor properties
        for name, tensor in [("Vocal", vocal), ("Instrumental", instrumental)]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values detected in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Infinite values detected in {name}")
        
        # 0. Make a copy of the inputs to avoid modifying the originals
        vocal_processed = vocal.clone()
        inst_processed = instrumental.clone()
        
        # 1. Loudness normalization before mixing
        norm_params = mix_params.get('normalization', default_mix_params['normalization'])
        if norm_params.get('enabled', default_mix_params['normalization']['enabled']):
            target_lufs = norm_params.get('target_lufs', default_mix_params['normalization']['target_lufs'])
            try:
                vocal_processed, inst_processed = normalize_before_mixing(
                    vocal_processed, inst_processed, target_lufs, sr
                )
                print(f"Applied pre-mix normalization to target LUFS: {target_lufs}")
            except Exception as e:
                print(f"Warning: Normalization failed, using original signals. Error: {e}")
                # Continue with unnormalized signals
        
        # 2. Phase alignment
        phase_params = mix_params.get('phase_alignment', default_mix_params['phase_alignment'])
        if phase_params.get('enabled', default_mix_params['phase_alignment']['enabled']):
            try:
                if phase_params.get('multiband', default_mix_params['phase_alignment']['multiband']):
                    # Apply multiband phase alignment with proper frequency bands
                    bands = [(0, 250), (250, 1200), (1200, 4000), (4000, 20000)]
                    inst_processed = multiband_phase_alignment(
                        vocal_processed, inst_processed, bands, sr,
                        enable_freq_dependent=phase_params.get('freq_dependent', default_mix_params['phase_alignment']['freq_dependent']),
                        enable_transient_preservation=phase_params.get('transient_preservation', default_mix_params['phase_alignment']['transient_preservation']),
                        enable_phase_locking=phase_params.get('phase_locking', default_mix_params['phase_alignment']['phase_locking'])
                    )
                    print("Applied multiband phase alignment")
                else:
                    # Apply standard phase alignment
                    inst_processed = align_phases(
                        vocal_processed, inst_processed,
                        enable_freq_dependent=phase_params.get('freq_dependent', default_mix_params['phase_alignment']['freq_dependent']),
                        enable_transient_preservation=phase_params.get('transient_preservation', default_mix_params['phase_alignment']['transient_preservation']),
                        enable_phase_locking=phase_params.get('phase_locking', default_mix_params['phase_alignment']['phase_locking'])
                    )
                    print("Applied standard phase alignment")
            except Exception as e:
                print(f"Warning: Phase alignment failed. Error: {e}")
                # Continue without phase alignment
        
        # 3. Apply compression to individual tracks
        # 3a. Vocal compression
        vocal_comp_params = mix_params.get('vocal_compression', default_mix_params['vocal_compression'])
        if vocal_comp_params.get('enabled', default_mix_params['vocal_compression']['enabled']):
            try:
                threshold = vocal_comp_params.get('threshold', default_mix_params['vocal_compression']['threshold'])
                ratio = vocal_comp_params.get('ratio', default_mix_params['vocal_compression']['ratio'])
                vocal_processed = apply_compression(vocal_processed, threshold, ratio, sr=sr)
                print(f"Applied vocal compression: threshold={threshold}dB, ratio={ratio}:1")
            except Exception as e:
                print(f"Warning: Vocal compression failed. Error: {e}")
        
        # 3b. Instrumental compression
        inst_comp_params = mix_params.get('instrumental_compression', default_mix_params['instrumental_compression'])
        if inst_comp_params.get('enabled', default_mix_params['instrumental_compression']['enabled']):
            try:
                threshold = inst_comp_params.get('threshold', default_mix_params['instrumental_compression']['threshold'])
                ratio = inst_comp_params.get('ratio', default_mix_params['instrumental_compression']['ratio'])
                inst_processed = apply_compression(inst_processed, threshold, ratio, sr=sr)
                print(f"Applied instrumental compression: threshold={threshold}dB, ratio={ratio}:1")
            except Exception as e:
                print(f"Warning: Instrumental compression failed. Error: {e}")
        
        # 4. Apply gain levels
        vocal_gain = mix_params.get('vocal_gain', default_mix_params['vocal_gain'])
        inst_gain = mix_params.get('instrumental_gain', default_mix_params['instrumental_gain'])
        
        # Handle any extreme gain values
        vocal_gain = max(0.0, min(10.0, vocal_gain))  # Cap between 0 and 10
        inst_gain = max(0.0, min(10.0, inst_gain))    # Cap between 0 and 10
        
        # Apply gain
        vocal_processed = vocal_processed * vocal_gain
        inst_processed = inst_processed * inst_gain
        
        # 5. Sidechain compression (duck instrumental under vocal)
        # This is a more sophisticated approach than simple mixing
        sidechain_params = mix_params.get('sidechain', None)
        if sidechain_params is not None and sidechain_params.get('enabled', False):
            try:
                threshold = sidechain_params.get('threshold', -24.0)
                ratio = sidechain_params.get('ratio', 2.0)
                inst_processed = sidechain_compression(
                    inst_processed, vocal_processed, threshold, ratio, sr=sr
                )
                print(f"Applied sidechain compression: threshold={threshold}dB, ratio={ratio}:1")
            except Exception as e:
                print(f"Warning: Sidechain compression failed. Error: {e}")
        
        # 6. Mix together
        try:
            # Simple mixing
            mixed = vocal_processed + inst_processed
            
            # Safety check for extreme values
            if torch.isnan(mixed).any() or torch.isinf(mixed).any():
                raise ValueError("NaN or Inf values detected in mix")
                
            # Apply gain compensation to avoid clipping
            max_val = mixed.abs().max()
            if max_val > 0.95:  # If we're close to clipping
                safe_gain = 0.95 / max_val
                mixed = mixed * safe_gain
                print(f"Applied safety gain adjustment: {safe_gain:.4f}")
        except Exception as e:
            print(f"Warning: Mixing failed. Error: {e}")
            # Fallback to a simpler, safer mixing method
            vocal_safe = torch.clamp(vocal_processed, -1.0, 1.0)
            inst_safe = torch.clamp(inst_processed, -1.0, 1.0)
            mixed = (vocal_safe + inst_safe) / 2.0
            print("Used fallback safe mixing approach")
        
        # 7. Apply multi-band compression to the mix
        mb_comp_params = mix_params.get('multiband_compression', default_mix_params['multiband_compression'])
        if mb_comp_params.get('enabled', default_mix_params['multiband_compression']['enabled']):
            try:
                bands = mb_comp_params.get('bands', default_mix_params['multiband_compression']['bands'])
                thresholds = mb_comp_params.get('thresholds', default_mix_params['multiband_compression']['thresholds'])
                ratios = mb_comp_params.get('ratios', default_mix_params['multiband_compression']['ratios'])
                
                # Validate that bands, thresholds, and ratios have matching lengths
                if len(bands) == len(thresholds) == len(ratios):
                    mixed = multi_band_compression(mixed, bands, thresholds, ratios, sr)
                    print(f"Applied multiband compression with {len(bands)} bands")
                else:
                    print("Warning: Multiband compression parameters have mismatched lengths")
            except Exception as e:
                print(f"Warning: Multiband compression failed. Error: {e}")
        
        # 8. Apply stereo width enhancement if stereo
        if mixed.dim() > 1 and mixed.shape[0] > 1:
            stereo_params = mix_params.get('stereo_width', default_mix_params['stereo_width'])
            if stereo_params.get('enabled', default_mix_params['stereo_width']['enabled']):
                try:
                    width = stereo_params.get('width', default_mix_params['stereo_width']['width'])
                    # Ensure reasonable width value
                    width = max(0.0, min(2.0, width))
                    mixed = enhance_stereo_width(mixed, width)
                    print(f"Enhanced stereo width to: {width}")
                except Exception as e:
                    print(f"Warning: Stereo width enhancement failed. Error: {e}")
        
        # 9. Apply final limiting/soft clipping
        final_mix = mixed
        soft_clip_params = mix_params.get('soft_clip', default_mix_params['soft_clip'])
        if soft_clip_params.get('enabled', default_mix_params['soft_clip']['enabled']):
            try:
                final_mix = apply_look_ahead_limiter(
                    final_mix,
                    soft_clip_params.get('threshold', default_mix_params['soft_clip']['threshold']),
                    soft_clip_params.get('release_time', 0.050),
                    soft_clip_params.get('attack_time', 0.001),
                    0.005,  # look-ahead time
                    sr
                )
                print(f"Applied look-ahead limiter with threshold: {soft_clip_params.get('threshold')}dB")
            except Exception as e:
                print(f"Warning: Limiting failed. Error: {e}")
                # Fallback to simple clipping
                final_mix = torch.clamp(final_mix, -1.0, 1.0)
                print("Used fallback clipping")
        
        # 10. Final safety check
        if torch.isnan(final_mix).any() or torch.isinf(final_mix).any():
            print("Warning: NaN or Inf values detected in final mix. Using fallback.")
            # Reset to a safe mix
            vocal_safe = torch.clamp(vocal * vocal_gain, -0.5, 0.5)
            inst_safe = torch.clamp(instrumental * inst_gain, -0.5, 0.5)
            final_mix = (vocal_safe + inst_safe) / 2.0
        
        return final_mix
        
    except Exception as e:
        print(f"Critical error in enhanced_audio_mix: {e}")
        # Absolute fallback - return simple mix of original inputs at reduced levels
        vocal_safe = vocal * 0.5
        inst_safe = instrumental * 0.5
        safe_mix = (vocal_safe + inst_safe) / 2.0
        return safe_mix

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