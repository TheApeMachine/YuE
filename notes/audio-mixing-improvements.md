# Comprehensive Audio Mixing Improvements for YuE Music Generation

This document outlines a complete set of mixing improvements to enhance the audio quality of the YuE music generation system. These techniques address common issues in AI-generated music and bring the output closer to professional studio quality.

## Table of Contents

1. [Basic Mixing Improvements](#basic-mixing-improvements)
2. [Phase Alignment](#phase-alignment)
3. [Dynamic Processing](#dynamic-processing)
4. [Frequency-Aware Processing](#frequency-aware-processing)
5. [Spatial Processing](#spatial-processing)
6. [Multi-band Processing](#multi-band-processing)
7. [Integration with YuE](#integration-with-yue)
8. [Post-Processing Enhancements](#post-processing-enhancements)

## Basic Mixing Improvements

### Level Balancing

Replace the simple arithmetic averaging with controllable level balancing:

```python
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
```

### Proper Gain Staging

Add proper gain staging to maintain optimal signal levels:

```python
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
```

### Pre-mix Normalization

Normalize tracks before mixing to ensure consistent levels:

```python
def normalize_before_mixing(vocal, instrumental, target_lufs=-16.0):
    """
    Normalize tracks to consistent loudness before mixing
    
    Args:
        vocal: Vocal track tensor
        instrumental: Instrumental track tensor
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized vocal and instrumental tracks
    """
    # Calculate LUFS for each track
    vocal_lufs = measure_lufs(vocal)
    instrumental_lufs = measure_lufs(instrumental)
    
    # Calculate gain adjustments
    vocal_gain = 10 ** ((target_lufs - vocal_lufs) / 20.0)
    instrumental_gain = 10 ** ((target_lufs - instrumental_lufs) / 20.0)
    
    # Apply normalization
    vocal_normalized = vocal * vocal_gain
    instrumental_normalized = instrumental * instrumental_gain
    
    return vocal_normalized, instrumental_normalized
```

## Phase Alignment

### Basic Phase Alignment

Align phases between vocal and instrumental tracks:

```python
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
    """Align a single audio channel"""
    # Compute STFTs
    ref_stft = torch.stft(reference, fft_size, hop_size, window=torch.hann_window(fft_size), 
                           return_complex=True)
    target_stft = torch.stft(target, fft_size, hop_size, window=torch.hann_window(fft_size), 
                              return_complex=True)
    
    # Extract magnitudes and phases
    ref_mag = torch.abs(ref_stft)
    ref_phase = torch.angle(ref_stft)
    target_mag = torch.abs(target_stft)
    
    # Create new STFT with target magnitude but reference phase
    aligned_stft = torch.polar(target_mag, ref_phase)
    
    # Convert back to time domain
    aligned_signal = torch.istft(aligned_stft, fft_size, hop_size, 
                                 window=torch.hann_window(fft_size))
    
    return aligned_signal
```

### Time Alignment through Cross-Correlation

Find and correct timing offsets between tracks:

```python
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
        
    if mode == 'shift':
        # Simple shifting with zero-padding
        result = torch.zeros_like(audio)
        if offset > 0:
            # Delay
            result[:, offset:] = audio[:, :-offset]
        else:
            # Advance
            result[:, :offset] = audio[:, -offset:]
        return result
    else:
        # Phase vocoder time stretching to preserve length while shifting
        # Implementation would go here
        pass
```

### Multi-band Phase Alignment

Align phases differently for different frequency bands:

```python
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

def apply_bandpass(audio, low_freq, high_freq, sr):
    """
    Apply bandpass filter to audio
    
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
    
    # Create filter
    filter_length = 1024
    band_filter = torch.hamming_window(filter_length)
    
    for i in range(filter_length):
        # Sinc filter design (simplified)
        if i != filter_length // 2:  # Avoid division by zero
            band_filter[i] *= (torch.sin(torch.tensor(3.14159 * high_normalized * (i - filter_length // 2))) - 
                              torch.sin(torch.tensor(3.14159 * low_normalized * (i - filter_length // 2)))) / (3.14159 * (i - filter_length // 2))
    
    # Apply filter using convolution
    filtered = torch.nn.functional.conv1d(
        audio.unsqueeze(0) if audio.dim() == 1 else audio.unsqueeze(1),
        band_filter.view(1, 1, -1),
        padding=filter_length // 2
    )
    
    # Reshape output to match input
    if audio.dim() == 1:
        filtered = filtered.squeeze(0).squeeze(0)
    else:
        filtered = filtered.squeeze(1)
    
    return filtered
```

## Dynamic Processing

### Basic Compression

Apply dynamics processing to control levels:

```python
def apply_compression(audio, threshold=-20.0, ratio=2.0, attack=0.005, release=0.05, sr=44100):
    """
    Apply basic compression to audio
    
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
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate attack and release coefficients
    attack_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * attack))
    release_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * release))
    
    # Process sample by sample
    output = torch.zeros_like(audio)
    env = 0.0  # Envelope follower
    
    for i in range(len(audio)):
        # Calculate instantaneous level
        level = abs(audio[i])
        
        # Envelope follower with different time constants for attack and release
        if level > env:
            # Attack phase
            env = attack_coeff * env + (1 - attack_coeff) * level
        else:
            # Release phase
            env = release_coeff * env + (1 - release_coeff) * level
        
        # Calculate gain reduction
        if env > threshold_linear:
            gain_reduction = threshold_linear + (env - threshold_linear) / ratio
            gain = gain_reduction / env
        else:
            gain = 1.0
        
        # Apply gain
        output[i] = audio[i] * gain
    
    return output
```

### Multi-band Compression

Apply different compression to different frequency bands:

```python
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
```

### Side-chain Compression

Apply side-chain compression to make vocals sit well over instrumentals:

```python
def sidechain_compression(audio, sidechain_signal, threshold=-20.0, ratio=2.0, 
                           attack=0.005, release=0.05, sr=44100):
    """
    Apply sidechain compression to audio
    
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
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate attack and release coefficients
    attack_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * attack))
    release_coeff = torch.exp(-torch.log(torch.tensor(9.0)) / (sr * release))
    
    # Process sample by sample
    output = torch.zeros_like(audio)
    env = 0.0  # Envelope follower
    
    for i in range(len(audio)):
        # Calculate instantaneous level of sidechain signal
        if i < len(sidechain_signal):
            level = abs(sidechain_signal[i])
        else:
            level = 0.0
        
        # Envelope follower with different time constants for attack and release
        if level > env:
            # Attack phase
            env = attack_coeff * env + (1 - attack_coeff) * level
        else:
            # Release phase
            env = release_coeff * env + (1 - release_coeff) * level
        
        # Calculate gain reduction
        if env > threshold_linear:
            gain_reduction = threshold_linear + (env - threshold_linear) / ratio
            gain = gain_reduction / env
        else:
            gain = 1.0
        
        # Apply gain to the audio (not the sidechain)
        output[i] = audio[i] * gain
    
    return output
```

## Frequency-Aware Processing

### Spectral Balancing

Apply spectral balancing to create a balanced mix:

```python
def spectral_balance(audio, target_spectrum, fft_size=2048, hop_size=512):
    """
    Adjust the spectrum of audio to match a target spectrum
    
    Args:
        audio: Audio tensor
        target_spectrum: Target spectral shape
        fft_size: FFT size
        hop_size: Hop size
        
    Returns:
        Spectrally balanced audio
    """
    # Compute STFT
    stft = torch.stft(audio, fft_size, hop_size, window=torch.hann_window(fft_size), 
                      return_complex=True)
    
    # Extract magnitude and phase
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Compute spectral shape
    mean_spectrum = torch.mean(mag, dim=1)
    
    # Compute scaling factors to match target spectrum
    scaling = target_spectrum / (mean_spectrum + 1e-10)  # Avoid division by zero
    
    # Apply scaling
    scaled_mag = mag * scaling.unsqueeze(1)
    
    # Reconstruct STFT
    scaled_stft = torch.polar(scaled_mag, phase)
    
    # Convert back to time domain
    result = torch.istft(scaled_stft, fft_size, hop_size, window=torch.hann_window(fft_size))
    
    return result
```

### Vocal Enhancement EQ

Apply specialized EQ to enhance vocals:

```python
def enhance_vocals(vocals, level=1.0):
    """
    Apply EQ to enhance vocals
    
    Args:
        vocals: Vocal track tensor
        level: Enhancement level (0.0 to 2.0)
        
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
    
    # Apply EQ bands
    enhanced = torch.zeros_like(vocals)
    for low_freq, high_freq, gain in bands:
        # Apply bandpass filter
        band = apply_bandpass(vocals, low_freq, high_freq, sr=44100)
        
        # Apply gain
        enhanced += band * (1.0 + gain * level)
    
    return enhanced
```

### Instrumental EQ for Vocal Space

Carve out frequency space for vocals in the instrumental:

```python
def carve_space_for_vocals(instrumental, vocal_analyzer, level=1.0):
    """
    Dynamically carve frequency space for vocals in instrumental
    
    Args:
        instrumental: Instrumental track tensor
        vocal_analyzer: Function that analyzes vocal spectrum
        level: Amount of carving (0.0 to 1.0)
        
    Returns:
        Processed instrumental with space for vocals
    """
    # Get vocal spectrum
    vocal_spectrum = vocal_analyzer()
    
    # Find dominant vocal frequencies
    peak_freqs = torch.topk(vocal_spectrum, k=3).indices
    
    # Create dynamic EQ
    eq_result = instrumental.clone()
    
    # Apply notches at vocal peak frequencies
    for freq_idx in peak_freqs:
        freq = freq_idx * 44100 / 2048  # Convert bin index to Hz
        q = 1.5  # Q factor for notch width
        gain = -6.0 * level  # Reduction in dB
        
        eq_result = apply_parametric_eq(eq_result, freq, q, gain)
    
    return eq_result

def apply_parametric_eq(audio, center_freq, q, gain_db, sr=44100):
    """
    Apply parametric EQ to audio
    
    Args:
        audio: Audio tensor
        center_freq: Center frequency in Hz
        q: Q factor
        gain_db: Gain in dB
        sr: Sample rate
        
    Returns:
        EQ'd audio
    """
    # Convert gain to linear
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Compute filter coefficients (simplified biquad filter)
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
    result = torch.zeros_like(audio)
    
    for i in range(len(audio)):
        # Direct form II implementation
        x0 = audio[i]
        w = x0 - a1 * x1 - a2 * x2
        y0 = b0 * w + b1 * x1 + b2 * x2
        
        # Update state
        x2 = x1
        x1 = w
        y2 = y1
        y1 = y0
        
        result[i] = y0
    
    return result
```

## Spatial Processing

### Stereo Width Enhancement

Enhance stereo width for immersive experience:

```python
def enhance_stereo_width(audio, width=1.5):
    """
    Enhance stereo width
    
    Args:
        audio: Stereo audio tensor (2 channels)
        width: Width factor (1.0 = normal, > 1.0 = wider)
        
    Returns:
        Width-enhanced stereo audio
    """
    if audio.shape[0] < 2:
        # Convert mono to stereo first
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
```

### Vocal Panning

Apply controlled panning to vocals for spatial positioning:

```python
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
    pan = torch.clamp(pan_position, -1.0, 1.0)
    angle = (pan + 1.0) * math.pi / 4.0  # 0 to pi/2
    
    left_gain = math.cos(angle)
    right_gain = math.sin(angle)
    
    # Apply gains
    left = audio[0] * left_gain
    right = audio[1] * right_gain
    
    return torch.stack([left, right])
```

### Reverb and Ambience

Add spatial reverb for depth and ambience:

```python
def apply_reverb(audio, mix=0.3, room_size=0.8, damping=0.5, sr=44100):
    """
    Apply reverb to audio
    
    Args:
        audio: Audio tensor
        mix: Dry/wet mix (0.0 to 1.0)
        room_size: Room size (0.0 to 1.0)
        damping: Damping factor (0.0 to 1.0)
        sr: Sample rate
        
    Returns:
        Reverberated audio
    """
    # Simple feedback delay network reverb implementation
    delay_lengths = [int(sr * t) for t in [0.0297, 0.0371, 0.0411, 0.0437]]
    max_delay = max(delay_lengths)
    
    # Feedback matrix (Hadamard)
    feedback_matrix = torch.tensor([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ]) * 0.5
    
    # Apply room size
    feedback_gain = 0.7 * room_size
    
    # Initialize delay lines
    delay_lines = [torch.zeros(length) for length in delay_lengths]
    
    # Process audio
    output = torch.zeros_like(audio)
    
    for i in range(len(audio)):
        # Get input sample
        input_sample = audio[i]
        
        # Read from delay lines
        delay_outputs = torch.tensor([line[-1] for line in delay_lines])
        
        # Apply feedback matrix
        feedback = torch.matmul(feedback_matrix, delay_outputs) * feedback_gain
        
        # Write to delay lines (input + feedback)
        for j in range(len(delay_lines)):
            delay_lines[j] = torch.cat([torch.tensor([input_sample + feedback[j]]), delay_lines[j][:-1]])
        
        # Apply damping
        delay_outputs = delay_outputs * (1.0 - damping * 0.5)
        
        # Mix dry and wet
        output[i] = input_sample * (1.0 - mix) + torch.sum(delay_outputs) * mix / 4.0
    
    return output
```

## Multi-band Processing

### Multi-band Saturation

Apply different saturation to different frequency bands:

```python
def multi_band_saturation(audio, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], 
                           amounts=[0.8, 0.4, 0.2, 0.1], sr=44100):
    """
    Apply different saturation amounts to different frequency bands
    
    Args:
        audio: Audio tensor
        bands: List of (low_freq, high_freq) tuples defining bands
        amounts: Saturation amount for each band (0.0 to 1.0)
        sr: Sample rate
        
    Returns:
        Multi-band saturated audio
    """
    # Split into bands
    band_signals = []
    for low_freq, high_freq in bands:
        band_signal = apply_bandpass(audio, low_freq, high_freq, sr)
        band_signals.append(band_signal)
    
    # Apply saturation to each band
    saturated_bands = []
    for i, band_signal in enumerate(band_signals):
        saturated = apply_saturation(band_signal, amounts[i])
        saturated_bands.append(saturated)
    
    # Sum bands back together
    result = sum(saturated_bands)
    
    return result

def apply_saturation(audio, amount):
    """
    Apply saturation effect to audio
    
    Args:
        audio: Audio tensor
        amount: Saturation amount (0.0 to 1.0)
        
    Returns:
        Saturated audio
    """
    # Tanh saturation with blend
    saturated = torch.tanh(audio * (1.0 + 3.0 * amount))
    
    # Blend with dry signal
    result = audio * (1.0 - amount) + saturated * amount
    
    return result
```

### Multi-band Stereo Processing

Apply different stereo processing to different bands:

```python
def multi_band_stereo(audio, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)], 
                       widths=[0.5, 0.8, 1.2, 1.5], sr=44100):
    """
    Apply different stereo widths to different frequency bands
    
    Args:
        audio: Stereo audio tensor
        bands: List of (low_freq, high_freq) tuples defining bands
        widths: Stereo width for each band
        sr: Sample rate
        
    Returns:
        Multi-band stereo processed audio
    """
    if audio.shape[0] < 2:
        # Convert mono to stereo first
        audio = audio.repeat(2, 1)
    
    # Process each channel
    left = audio[0]
    right = audio[1]
    
    left_bands = []
    right_bands = []
    
    # Split into bands
    for low_freq, high_freq in bands:
        left_band = apply_bandpass(left, low_freq, high_freq, sr)
        right_band = apply_bandpass(right, low_freq, high_freq, sr)
        
        left_bands.append(left_band)
        right_bands.append(right_band)
    
    # Apply stereo processing to each band
    processed_left_bands = []
    processed_right_bands = []
    
    for i in range(len(bands)):
        # Extract mid and side for this band
        mid = (left_bands[i] + right_bands[i]) / 2
        side = (left_bands[i] - right_bands[i]) / 2
        
        # Apply width
        side_enhanced = side * widths[i]
        
        # Recombine
        left_processed = mid + side_enhanced
        right_processed = mid - side_enhance