import math
import torch
from scipy import signal

from YuE.inference.dsp.utils import apply_per_channel, to_mono

def apply_highpass(audio, freq, sr):
    """
    Advanced Butterworth highpass filter with precise digital biquad implementation.
    
    This is significantly more accurate than the original 1-pole implementation,
    providing steeper rolloff and better phase response.
    
    Args:
        audio: Audio tensor
        freq: Cutoff frequency in Hz
        sr: Sample rate in Hz
        
    Returns:
        Filtered audio with same shape as input
    """
    try:
        # Try using SciPy's zero-phase filtfilt for best quality
        return apply_per_channel(audio, _highpass_scipy, freq, sr)
    except Exception:
        # Fall back to direct biquad implementation
        return apply_per_channel(audio, _highpass_biquad, freq, sr)

def _highpass_scipy(channel, freq, sr):
    """SciPy-based zero-phase highpass filter implementation."""
    nyquist = sr / 2
    # Use a 4th order Butterworth filter for better quality
    sos = signal.butter(4, freq / nyquist, btype='highpass', output='sos')
    
    # Convert to numpy for processing
    device = channel.device
    dtype = channel.dtype
    c_np = channel.cpu().numpy()
    
    # Use filtfilt for zero-phase filtering (no delay)
    filtered = signal.sosfiltfilt(sos, c_np)
    
    return torch.tensor(filtered, dtype=dtype, device=device)

def _highpass_biquad(channel, freq, sr):
    """
    Advanced biquad highpass filter implementation.
    
    This implementation uses the transposed direct form II structure
    which has better numerical properties than the original 1-pole filter.
    """
    # Convert frequency and sampling rate to angular frequency
    omega = 2 * math.pi * freq / sr
    alpha = math.sin(omega) / (2 * 0.7071)  # Q=0.7071 for Butterworth response
    
    # Calculate biquad coefficients for highpass
    b0 = (1 + math.cos(omega)) / 2
    b1 = -(1 + math.cos(omega))
    b2 = (1 + math.cos(omega)) / 2
    a0 = 1 + alpha
    a1 = -2 * math.cos(omega)
    a2 = 1 - alpha
    
    # Normalize coefficients
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    
    # Transposed Direct Form II implementation (better numerical stability)
    out = torch.zeros_like(channel)
    z1 = 0.0
    z2 = 0.0
    
    for i in range(len(channel)):
        # Get input sample
        x = channel[i].item() if hasattr(channel[i], 'item') else float(channel[i])
        
        # Calculate output
        y = b0 * x + z1
        
        # Update state variables
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        
        out[i] = y
        
    return out

def apply_high_shelf(audio, freq, gain_db, sr):
    """
    High-shelf filter using a Biquad approach from your original code.
    """
    def _hs_single(channel):
        gain_lin = 10**(gain_db/20.0)
        w0 = 2*math.pi*freq/sr
        alpha = math.sin(w0)/2
        
        b0 = (gain_lin+1) + (gain_lin-1)*math.cos(w0) + 2*math.sqrt(gain_lin)*alpha
        b1 = -2*((gain_lin-1) + (gain_lin+1)*math.cos(w0))
        b2 = (gain_lin+1) + (gain_lin-1)*math.cos(w0) - 2*math.sqrt(gain_lin)*alpha
        a0 = (gain_lin+1) - (gain_lin-1)*math.cos(w0) + 2*math.sqrt(gain_lin)*alpha
        a1 = 2*((gain_lin-1) - (gain_lin+1)*math.cos(w0))
        a2 = (gain_lin+1) - (gain_lin-1)*math.cos(w0) - 2*math.sqrt(gain_lin)*alpha
        
        b0/=a0; b1/=a0; b2/=a0
        a1/=a0; a2/=a0
        
        out = torch.zeros_like(channel)
        x1=x2=0.0
        y1=y2=0.0
        # Process
        for i in range(len(channel)):
            x0 = channel[i].item() if hasattr(channel[i], 'item') else channel[i]
            y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
            out[i] = y0
            x2,x1 = x1,x0
            y2,y1 = y1,y0
        return out
    
    return apply_per_channel(audio, _hs_single)

def apply_bandpass(audio, low_freq, high_freq, sr=44100, order=4):
    """
    Bandpass filter with advanced fallback or zero-phase approach if scipy is available.
    Preserves your advanced fallback code.
    """
    try:
        return apply_per_channel(audio, _bandpass_scipy, low_freq, high_freq, sr, order)
    except ImportError:
        return apply_per_channel(audio, _apply_bandpass_fallback, low_freq, high_freq, sr)

def _bandpass_scipy(channel, low_freq, high_freq, sr, order):
    device = channel.device
    dtype = channel.dtype
    
    nyquist = sr/2
    lo = max(10, min(low_freq, nyquist-10))
    hi = max(lo+10, min(high_freq, nyquist-10))
    
    sos = signal.butter(order, [lo, hi], btype='bandpass', fs=sr, output='sos')
    
    c_np = channel.cpu().numpy()
    filtered = signal.sosfiltfilt(sos, c_np)
    return torch.tensor(filtered, dtype=dtype, device=device)

def _apply_bandpass_fallback(channel, low_freq, high_freq, sr):
    """
    The advanced fallback code from your original snippet 
    (the big windowed-sinc + blackman window, etc.).
    """
    # We'll keep it basically the same as your original for completeness.
    import numpy as np
    import math
    from torch.nn.functional import conv1d, pad as torch_pad
    
    # Convert channel to numpy if needed
    if not isinstance(channel, torch.Tensor):
        channel = torch.tensor(channel, dtype=torch.float32)
    device = channel.device
    dtype = channel.dtype
    channel_np = channel.cpu().numpy()
    
    nyquist = sr/2
    low_norm = low_freq/nyquist
    high_norm = high_freq/nyquist
    
    # Determine filter length
    # same logic as original
    lowest_freq = max(20, low_freq)
    min_cycles = 4
    filter_len = int(min_cycles*sr/lowest_freq)
    # round up to nearest power of 2 + 1
    filter_len = 2**(int(math.log2(filter_len))+1) + 1
    filter_len = min(filter_len, 16385)
    if filter_len%2 == 0:
        filter_len += 1
    
    n = np.arange(filter_len)
    center = filter_len//2
    window = np.blackman(filter_len)
    
    sinc_lowpass = np.ones(filter_len)
    sinc_highpass = np.ones(filter_len)
    
    for i in n:
        if i!=center:
            # Lowpass => sin(2*pi*high_norm*(i-center))/(pi*(i-center))
            sinc_lowpass[i] = (math.sin(2*math.pi*high_norm*(i-center))/
                               (math.pi*(i-center)))
            # Highpass => sin(2*pi*low_norm*(i-center))/(pi*(i-center))
            sinc_highpass[i] = (math.sin(2*math.pi*low_norm*(i-center))/
                                (math.pi*(i-center)))
    # center
    sinc_lowpass[center] = 2*high_norm
    sinc_highpass[center] = 2*low_norm
    
    bandpass_filter = sinc_lowpass - sinc_highpass
    bandpass_filter *= window
    
    # We do the frequency domain normalization by measuring the response at center freq:
    center_freq = (low_freq+high_freq)/2
    center_freq_norm = center_freq/nyquist
    resp = 0.0
    for i in range(filter_len):
        resp += bandpass_filter[i]*math.cos(2*math.pi*center_freq_norm*(i-center))
    if abs(resp)>1e-10:
        bandpass_filter /= resp
    
    # Convert to torch
    filt_torch = torch.tensor(bandpass_filter, dtype=torch.float32)
    if channel.is_cuda:
        filt_torch = filt_torch.to(device)
    
    # shape => [1, 1, filter_len]
    filt_torch = filt_torch.view(1,1,-1)
    
    # Convolution
    pad_amount = filter_len//2
    ch_padded = torch_pad(channel.view(1,1,-1), (pad_amount,pad_amount))
    filtered_t = conv1d(ch_padded, filt_torch).view(-1)
    return filtered_t.to(dtype=dtype)

def apply_parametric_eq(audio, center_freq, q, gain_db, sr=44100):
    """
    Apply parametric EQ to audio with a precise biquad filter implementation.
    
    Args:
        audio: Audio tensor (mono or stereo)
        center_freq: Center frequency in Hz
        q: Q factor (bandwidth)
        gain_db: Gain in dB
        sr: Sample rate
        
    Returns:
        EQ'd audio
    """
    def _eq_single(channel):
        # Convert gain to linear
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Compute filter coefficients
        w0 = 2 * math.pi * center_freq / sr
        alpha = math.sin(w0) / (2 * q)
        
        # Peaking EQ filter coefficients
        b0 = 1 + alpha * gain_linear
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * gain_linear
        a0 = 1 + alpha / gain_linear
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / gain_linear
        
        # Normalize
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply filter using direct form II
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        result = torch.zeros_like(channel)
        
        for i in range(len(channel)):
            # Direct form II implementation
            x0 = channel[i].item() if hasattr(channel[i], 'item') else float(channel[i])
            w = x0 - a1 * x1 - a2 * x2
            y0 = b0 * w + b1 * x1 + b2 * x2
            
            # Update state
            x2 = x1
            x1 = w
            y2 = y1
            y1 = y0
            
            result[i] = y0
        
        return result
    
    return apply_per_channel(audio, _eq_single)

def enhance_vocals(vocals, level=1.0, sr=44100):
    """
    Apply specialized EQ to enhance vocals with carefully chosen frequency bands
    to reduce mud, boost presence, and add air.
    
    Args:
        vocals: Vocal track tensor
        level: Enhancement level (0.0 to 2.0)
        sr: Sample rate
        
    Returns:
        Enhanced vocals
    """
    # Define EQ bands and gains
    bands = [
        (100, 250, -0.5),   # Reduce low-end mud
        (250, 800, 0.0),    # Keep low-mids neutral
        (800, 1200, 0.5),   # Slight boost for vocal presence
        (1200, 3500, 1.0),  # Main vocal presence boost
        (3500, 8000, 0.8),  # Air and clarity
        (8000, 16000, 0.5)  # Top-end air
    ]
    
    # Apply multi-band EQ approach
    enhanced = torch.zeros_like(vocals)
    
    for low_freq, high_freq, gain in bands:
        # Apply bandpass filter
        band = apply_bandpass(vocals, low_freq, high_freq, sr)
        
        # Apply gain based on enhancement level
        enhanced += band * (1.0 + gain * level)
    
    # Final normalization to avoid clipping
    max_val = enhanced.abs().max().item()
    if max_val > 1.0:
        enhanced = enhanced / max_val
    
    return enhanced

def carve_space_for_vocals(instrumental, vocals, level=1.0, sr=44100):
    """
    Dynamically carve frequency space for vocals in the instrumental
    based on the vocal spectrum.
    
    Args:
        instrumental: Instrumental track tensor
        vocals: Vocal track to analyze
        level: Amount of carving (0.0 to 1.0)
        sr: Sample rate
        
    Returns:
        Processed instrumental with space for vocals
    """
    # Convert to mono for analysis if needed
    vocals_mono = to_mono(vocals)
    
    # Analyze vocal spectrum using STFT
    n_fft = 2048
    window = torch.hann_window(n_fft, device=vocals_mono.device if vocals_mono.is_cuda else 'cpu')
    stft = torch.stft(vocals_mono, n_fft, hop_length=512, window=window, return_complex=True)
    
    # Get magnitude spectrum
    mag_spec = torch.abs(stft)
    
    # Average over time to get overall spectral shape
    avg_spectrum = torch.mean(mag_spec, dim=1)
    
    # Find dominant frequency regions (top 3)
    _, peak_indices = torch.topk(avg_spectrum, k=3)
    
    # Convert bin indices to frequencies
    bin_to_freq = sr / n_fft
    peak_freqs = peak_indices.cpu().numpy() * bin_to_freq
    
    # Apply notches at the peak frequencies
    result = instrumental.clone()
    
    for freq in peak_freqs:
        # Apply a gentle notch at each peak frequency
        q = 1.5  # Q factor for notch width
        gain = -6.0 * level  # Reduction in dB based on carving level
        
        result = apply_parametric_eq(result, freq, q, gain, sr)
    
    return result

def spectral_balance(audio, target_spectrum=None, strength=1.0, fft_size=2048, hop_size=512, sr=44100):
    """
    Adjust the spectrum of audio to match a target spectrum or a balanced frequency curve.
    
    Args:
        audio: Audio tensor
        target_spectrum: Target spectral shape (if None, a balanced curve is used)
        strength: Strength of the spectral matching (0.0 to 1.0)
        fft_size: FFT size
        hop_size: Hop size
        sr: Sample rate
        
    Returns:
        Spectrally balanced audio
    """
    def _spectral_balance_single(channel):
        # Compute STFT
        window = torch.hann_window(fft_size, device=channel.device if channel.is_cuda else 'cpu')
        stft = torch.stft(channel, fft_size, hop_size, window=window, return_complex=True)
        
        # Extract magnitude and phase
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Compute current spectral shape (average across time)
        current_spectrum = torch.mean(mag, dim=1)
        
        # Create default target spectrum if not provided (balanced curve)
        if target_spectrum is None:
            # Create a balanced spectrum curve based on typical "ideal" frequency response
            # with slight bass boost, flat mids, and gentle high-end roll-off
            freq_bins = torch.linspace(0, sr/2, fft_size//2 + 1, device=current_spectrum.device)
            target = torch.ones_like(current_spectrum)
            
            # Apply frequency-dependent adjustments
            # Bass boost below 200Hz
            bass_mask = freq_bins < 200
            target[bass_mask] = 1.2
            
            # Flat mids (200Hz to 4kHz)
            mid_mask = (freq_bins >= 200) & (freq_bins < 4000)
            target[mid_mask] = 1.0
            
            # Gentle high-frequency roll-off above 4kHz
            high_mask = freq_bins >= 4000
            roll_off = 1.0 - 0.3 * torch.log10(torch.clamp(freq_bins[high_mask] / 4000, min=1.0))
            target[high_mask] = roll_off
        else:
            # Ensure target spectrum has the right shape
            if len(target_spectrum) != len(current_spectrum):
                # Resample target to match
                target = torch.nn.functional.interpolate(
                    target_spectrum.unsqueeze(0).unsqueeze(0), 
                    size=len(current_spectrum), 
                    mode='linear'
                ).squeeze()
            else:
                target = target_spectrum
        
        # Compute scaling factors for each frequency bin
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        scaling = target / (current_spectrum + epsilon)
        
        # Apply strength control (blend between original and fully processed)
        scaling = 1.0 + strength * (scaling - 1.0)
        
        # Expand to match STFT time dimension
        scaling = scaling.unsqueeze(1).expand_as(mag)
        
        # Apply scaling to magnitude
        scaled_mag = mag * scaling
        
        # Reconstruct STFT with original phase
        scaled_stft = torch.polar(scaled_mag, phase)
        
        # Convert back to time domain
        result = torch.istft(scaled_stft, fft_size, hop_size, window=window)
        
        # Ensure output length matches input
        if len(result) > len(channel):
            result = result[:len(channel)]
        elif len(result) < len(channel):
            # Pad with zeros if needed
            padding = torch.zeros(len(channel) - len(result), device=result.device)
            result = torch.cat([result, padding])
        
        return result
    
    return apply_per_channel(audio, _spectral_balance_single)
