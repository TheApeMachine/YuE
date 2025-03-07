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
    cutoff_bin = torch.argmin(torch.abs(freq_bins - cutoff_freq))
    
    # Combine: use b's low frequencies and a's high frequencies
    combined_fft = b_fft.clone()
    combined_fft[cutoff_bin:] = a_fft[cutoff_bin:]
    
    # Match energy levels
    a_energy = torch.mean(torch.abs(a_fft[cutoff_bin:]) ** 2)
    b_energy = torch.mean(torch.abs(b_fft[:cutoff_bin]) ** 2)
    scale_factor = torch.sqrt(a_energy / (b_energy + 1e-8))
    combined_fft[:cutoff_bin] *= scale_factor
    
    # Convert back to time domain
    processed = torch.fft.irfft(combined_fft, n=min_length)
    
    return processed

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