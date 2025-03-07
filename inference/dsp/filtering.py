import math
import torch
from scipy import signal

from dsp.utils import apply_per_channel

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
