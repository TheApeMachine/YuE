import os
import torch
import torchaudio
import numpy as np
from torchaudio.transforms import Resample

# Import our enhanced processing functions
from audio_mixing import (
    align_phases, 
    multiband_phase_alignment, 
    enhance_stereo_width
)

def multi_resolution_stft(audio, fft_sizes=(256, 1024, 4096), hop_ratios=(0.25, 0.25, 0.25), sr=44100):
    """
    Perform multi-resolution STFT analysis using different window sizes for
    different frequency bands, optimizing time-frequency resolution tradeoffs.
    
    Args:
        audio: Audio tensor (mono)
        fft_sizes: Tuple of FFT sizes for different bands (small to large)
        hop_ratios: Ratio of hop size to window size for each FFT size
        sr: Sample rate
        
    Returns:
        Dictionary with STFT results for each resolution
    """
    results = {}
    
    # Ensure audio is single-channel for analysis
    if audio.dim() > 1 and audio.shape[0] > 1:
        analysis_audio = audio.mean(dim=0)
    else:
        analysis_audio = audio.squeeze(0) if audio.dim() > 1 else audio
    
    # Compute STFTs at different resolutions
    for i, fft_size in enumerate(fft_sizes):
        hop_size = int(fft_size * hop_ratios[i])
        window = torch.hann_window(fft_size)
        
        if torch.cuda.is_available():
            window = window.to(analysis_audio.device)
        
        # Compute STFT
        stft = torch.stft(
            analysis_audio, 
            n_fft=fft_size, 
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=True
        )
        
        # Calculate frequency resolution
        freq_resolution = sr / fft_size
        time_resolution = hop_size / sr
        
        # Store results
        results[fft_size] = {
            'stft': stft,
            'magnitude': torch.abs(stft),
            'phase': torch.angle(stft),
            'freq_resolution': freq_resolution,
            'time_resolution': time_resolution,
            'hop_size': hop_size,
            'window': window
        }
    
    return results

def multi_resolution_istft(mr_stft_data, audio_length=None):
    """
    Reconstruct audio from multi-resolution STFT data
    
    Args:
        mr_stft_data: Multi-resolution STFT data from multi_resolution_stft
        audio_length: Target audio length (if None, determined automatically)
        
    Returns:
        Reconstructed audio
    """
    reconstructions = []
    
    # Reconstruct from each resolution
    for fft_size, data in mr_stft_data.items():
        # Get parameters
        stft = data['stft']
        hop_size = data['hop_size']
        window = data['window']
        
        # Inverse STFT
        audio_recon = torch.istft(
            stft,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            length=audio_length
        )
        
        reconstructions.append(audio_recon)
    
    # Average the reconstructions
    # This simple approach works surprisingly well for many audio signals
    return torch.stack(reconstructions).mean(dim=0)

def multi_resolution_band_extraction(audio, band_ranges, sr=44100, crossfade_ratio=0.25):
    """
    Extract frequency bands using multi-resolution analysis, with appropriate
    window sizes for each frequency range to optimize time-frequency resolution.
    
    Args:
        audio: Audio tensor
        band_ranges: List of (low_freq, high_freq, fft_size) tuples
        sr: Sample rate
        crossfade_ratio: Amount of crossfade between adjacent bands (0-0.5)
        
    Returns:
        List of extracted bands
    """
    # Handle multi-channel audio
    if audio.dim() > 1:
        channels = []
        for ch in range(audio.shape[0]):
            ch_bands = multi_resolution_band_extraction(audio[ch], band_ranges, sr, crossfade_ratio)
            channels.append(ch_bands)
        
        # Reorganize by band rather than by channel
        result = []
        for band_idx in range(len(band_ranges)):
            band_channels = []
            for ch_idx in range(len(channels)):
                band_channels.append(channels[ch_idx][band_idx])
            result.append(torch.stack(band_channels))
        
        return result
    
    # Process mono audio
    extracted_bands = []
    audio_length = audio.shape[-1]
    
    # Sort bands by frequency
    sorted_bands = sorted(band_ranges, key=lambda x: x[0])
    
    for i, (low_freq, high_freq, fft_size) in enumerate(sorted_bands):
        # Calculate hop size
        hop_size = fft_size // 4
        
        # Prepare window
        window = torch.hann_window(fft_size)
        if torch.cuda.is_available():
            window = window.to(audio.device)
        
        # Calculate STFT
        stft = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=True
        )
        
        # Calculate frequencies for each bin
        freqs = torch.linspace(0, sr/2, stft.shape[0])
        
        # Create band mask with crossfade/overlap
        mask = torch.zeros(stft.shape[0])
        
        # Regular mask (box filter)
        in_band = (freqs >= low_freq) & (freqs <= high_freq)
        mask[in_band] = 1.0
        
        # Add crossfade transitions if this isn't the first or last band
        if i > 0:
            # Crossfade with previous band
            _, prev_high, _ = sorted_bands[i-1]
            overlap_width = (low_freq - prev_high) * crossfade_ratio
            if overlap_width > 0:
                # Create crossfade region
                overlap_mask = (freqs >= prev_high) & (freqs < low_freq)
                # Linear crossfade
                crossfade_values = (freqs[overlap_mask] - prev_high) / (low_freq - prev_high)
                mask[overlap_mask] = crossfade_values
        
        if i < len(sorted_bands) - 1:
            # Crossfade with next band
            next_low, _, _ = sorted_bands[i+1]
            overlap_width = (next_low - high_freq) * crossfade_ratio
            if overlap_width > 0:
                # Create crossfade region
                overlap_mask = (freqs > high_freq) & (freqs <= next_low)
                # Linear crossfade
                crossfade_values = 1.0 - (freqs[overlap_mask] - high_freq) / (next_low - high_freq)
                mask[overlap_mask] = crossfade_values
        
        # Apply mask to STFT
        masked_stft = stft * mask.view(-1, 1).to(stft.dtype)
        
        # Inverse STFT
        band = torch.istft(
            masked_stft,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            length=audio_length
        )
        
        extracted_bands.append(band)
    
    return extracted_bands

def replace_low_freq_with_energy_matched_single(a_waveform, b_waveform, cutoff_freq=5500.0, sr_a=16000, sr_b=44100):
    """
    Process a single audio channel by replacing low frequencies
    
    Args:
        a_waveform: Lower quality waveform (single channel)
        b_waveform: Higher quality waveform (single channel)
        cutoff_freq: Frequency cutoff point
        sr_a: Sample rate of a_waveform
        sr_b: Sample rate of b_waveform
        
    Returns:
        Processed single channel audio
    """
    # Convert to the same sample rate if needed
    if sr_a != sr_b:
        resampler = Resample(orig_freq=sr_a, new_freq=sr_b)
        a_waveform = resampler(a_waveform)
    
    # Ensure both have the same length
    min_length = min(a_waveform.shape[-1], b_waveform.shape[-1])
    a_waveform = a_waveform[..., :min_length]
    b_waveform = b_waveform[..., :min_length]
    
    # Apply phase alignment before frequency domain processing
    b_waveform = align_phases(a_waveform, b_waveform)
    
    # Convert to frequency domain
    a_fft = torch.fft.rfft(a_waveform)
    b_fft = torch.fft.rfft(b_waveform)
    
    # Calculate the frequency bin corresponding to the cutoff
    freq_bins = torch.fft.rfftfreq(min_length, d=1.0/sr_b)
    cutoff_bin = torch.argmin(torch.abs(freq_bins - cutoff_freq), dim=0)
    
    # Combine: use b's low frequencies and a's high frequencies
    combined_fft = b_fft.clone()
    combined_fft[cutoff_bin:] = a_fft[cutoff_bin:]
    
    # Match energy levels
    a_energy = torch.mean(torch.abs(a_fft[cutoff_bin:]) ** 2)
    b_energy = torch.mean(torch.abs(b_fft[:cutoff_bin]) ** 2)
    scale_factor = torch.sqrt(a_energy / (b_energy + 1e-8))
    
    # Apply scaling for smooth energy matching
    combined_fft[:cutoff_bin] *= scale_factor
    
    # Apply a gradual crossfade between the two at the cutoff region
    # Define a transition width (10% of the cutoff bin)
    transition_width = max(int(cutoff_bin * 0.1), 1)
    transition_start = max(0, cutoff_bin - transition_width)
    transition_end = min(len(combined_fft), cutoff_bin + transition_width)
    
    # Create crossfade weights
    weights = torch.linspace(0.0, 1.0, transition_end - transition_start)
    
    # Apply weighted crossfade
    crossfade_region = torch.linspace(transition_start, transition_end, transition_end - transition_start).long()
    combined_fft[crossfade_region] = (1.0 - weights) * (b_fft[crossfade_region] * scale_factor) + weights * a_fft[crossfade_region]
    
    # Convert back to time domain
    combined_waveform = torch.fft.irfft(combined_fft, n=min_length)
    
    return combined_waveform

def replace_low_freq_with_energy_matched(a_file, b_file, c_file, cutoff_freq=5500.0):
    """
    Original mono post-processing function
    
    Args:
        a_file: Lower quality audio file
        b_file: Higher quality audio file
        c_file: Output file path
        cutoff_freq: Frequency cutoff point
    """
    # Load audio
    a, sr_a = torchaudio.load(a_file)  # 16kHz
    b, sr_b = torchaudio.load(b_file)  # 44kHz
    
    # Convert to mono if stereo
    if a.shape[0] > 1:
        a = torch.mean(a, dim=0, keepdim=True)
    if b.shape[0] > 1:
        b = torch.mean(b, dim=0, keepdim=True)
    
    # Process the audio
    processed = replace_low_freq_with_energy_matched_single(
        a.squeeze(0), b.squeeze(0), cutoff_freq, sr_a, sr_b
    )
    
    # Save the result
    torchaudio.save(c_file, processed.unsqueeze(0), sr_b)

def replace_low_freq_with_energy_matched_stereo(a_file, b_file, c_file, cutoff_freq=5500.0):
    """
    Process stereo audio files with frequency-based enhancement
    
    Args:
        a_file: Lower quality stereo audio file
        b_file: Higher quality stereo audio file
        c_file: Output file path
        cutoff_freq: Frequency cutoff point
    """
    # Load audio
    a, sr_a = torchaudio.load(a_file)  # 16kHz
    b, sr_b = torchaudio.load(b_file)  # 44kHz
    
    # Ensure both are stereo
    if a.shape[0] == 1:
        a = a.repeat(2, 1)
    if b.shape[0] == 1:
        b = b.repeat(2, 1)
    
    # Process each channel separately
    output_channels = []
    for ch in range(a.shape[0]):
        processed_channel = replace_low_freq_with_energy_matched_single(
            a[ch], b[ch], cutoff_freq, sr_a, sr_b
        )
        output_channels.append(processed_channel)
    
    # Stack channels back together
    output = torch.stack(output_channels)
    
    # Apply stereo width enhancement
    output = enhance_stereo_width(output, width=1.2)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(c_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the stereo result
    torchaudio.save(c_file, output, sr_b)

# Add a new multiband version of the post-processing
def multiband_enhanced_stereo(a_file, b_file, c_file, bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)]):
    """
    Apply multiband processing and phase alignment for enhanced quality
    
    Args:
        a_file: Lower quality audio file
        b_file: Higher quality audio file
        c_file: Output file path
        bands: Frequency bands for multiband processing
    """
    # Load audio
    a, sr_a = torchaudio.load(a_file)
    b, sr_b = torchaudio.load(b_file)
    
    # Ensure consistent sample rate
    if sr_a != sr_b:
        resampler = Resample(orig_freq=sr_a, new_freq=sr_b)
        a = resampler(a)
        sr = sr_b
    else:
        sr = sr_a
    
    # Ensure both are stereo
    if a.shape[0] == 1:
        a = a.repeat(2, 1)
    if b.shape[0] == 1:
        b = b.repeat(2, 1)
    
    # Apply multiband phase alignment
    aligned = multiband_phase_alignment(a, b, bands, sr)
    
    # Enhance stereo width
    enhanced = enhance_stereo_width(aligned, width=1.3)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(c_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the enhanced stereo result
    torchaudio.save(c_file, enhanced, sr) 