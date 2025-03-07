import os
import torch
import torchaudio
import soundfile as sf
from torchaudio.transforms import Resample
import numpy as np

# Import our enhanced mixing functions
from audio_mixing import process_files_with_enhancements, enhanced_audio_mix

def load_audio_mono(filepath, sampling_rate=16000):
    """
    Load audio file with mono channel
    
    Args:
        filepath: Path to audio file
        sampling_rate: Target sampling rate
    
    Returns:
        Tensor of shape [1, samples]
    """
    audio, sr = torchaudio.load(filepath)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    
    return audio

def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    """
    Save audio with mono or stereo preservation
    
    Args:
        wav: Audio tensor of shape [channels, samples]
        path: Output file path
        sample_rate: Sampling rate
        rescale: Whether to rescale audio to avoid clipping
    """
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    
    torchaudio.save(str(path), wav, sample_rate)

def process_stereo_mix(vocal_path, instrumental_path, output_path):
    """
    Process stereo vocal and instrumental tracks into a mixed output
    with enhanced mixing quality
    
    Args:
        vocal_path: Path to vocal audio file
        instrumental_path: Path to instrumental audio file
        output_path: Path for mixed output
    """
    # Use our enhanced processing function instead of basic mixing
    return process_files_with_enhancements(vocal_path, instrumental_path, output_path)

def mix_tracks(tracks, output_dir):
    """
    Mix instrumental and vocal tracks with enhanced quality
    
    Args:
        tracks: List of track paths
        output_dir: Output directory for mixed tracks
    
    Returns:
        List of mixed track paths
    """
    mixed_tracks = []
    for inst_path in tracks:
        try:
            if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                and '_itrack' in inst_path:
                # find pair
                vocal_path = inst_path.replace('_itrack', '_vtrack')
                if not os.path.exists(vocal_path):
                    continue
                # mix with enhanced quality
                recons_mix = os.path.join(output_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
                process_files_with_enhancements(vocal_path, inst_path, recons_mix)
                mixed_tracks.append(recons_mix)
        except Exception as e:
            print(e)
    return mixed_tracks 

def apply_dither(audio, bits=16, dither_type='tpdf', noise_shaping=True):
    """
    Apply dithering to audio when converting to lower bit depth to prevent
    quantization distortion, especially in quiet passages.
    
    Args:
        audio: Audio tensor (float32, expected to be in range [-1, 1])
        bits: Target bit depth
        dither_type: Type of dithering to apply:
            'none': No dithering
            'rpdf': Rectangular PDF noise (basic dithering)
            'tpdf': Triangular PDF noise (higher quality dithering, recommended)
            'gaussian': Gaussian noise dithering
        noise_shaping: Whether to apply simple noise shaping (error feedback)
        
    Returns:
        Dithered audio ready for bit depth conversion
    """
    if bits == 32 or dither_type == 'none':
        # No dithering needed for 32-bit float
        return audio
    
    # Calculate step size for target bit depth (in normalized -1 to 1 range)
    # For 16-bit, this is 2/65536
    step = 2.0 / (2 ** bits)
    
    # Scale audio to appropriate range for bit depth conversion
    scale = (2 ** (bits - 1) - 1)
    
    # Create output tensor
    output = audio.clone()
    
    # Generate dither noise based on selected type
    if dither_type == 'rpdf':
        # Rectangular PDF (uniform) dither, amplitude +/- 0.5 LSB
        noise = torch.rand_like(audio) * step - (step / 2)
    elif dither_type == 'tpdf':
        # Triangular PDF dither (sum of two uniform distributions), +/- 1 LSB
        # This is the higher quality option recommended for audio
        noise = (torch.rand_like(audio) + torch.rand_like(audio)) * (step / 2) - step/2
    elif dither_type == 'gaussian':
        # Gaussian noise dither, standard deviation = 0.5 LSB
        noise = torch.randn_like(audio) * (step / 3)
    else:
        raise ValueError(f"Unknown dither type: {dither_type}")
    
    # Apply dither noise
    output = output + noise
    
    if noise_shaping:
        # Simple error feedback (first-order noise shaping)
        # Process each channel separately
        error_feedback = torch.zeros_like(output)
        
        # Need to process sample by sample for error feedback
        if output.dim() > 1:
            for ch in range(output.shape[0]):
                for i in range(output.shape[1]):
                    # Quantize the current sample
                    quantized = torch.round(output[ch, i] * scale) / scale
                    # Compute error
                    error = output[ch, i] - quantized
                    # Store quantized value
                    output[ch, i] = quantized
                    # Apply error feedback to next sample if not at the end
                    if i < output.shape[1] - 1:
                        output[ch, i + 1] += error * 0.5  # Feedback coefficient
        else:
            for i in range(output.shape[0]):
                # Quantize the current sample
                quantized = torch.round(output[i] * scale) / scale
                # Compute error
                error = output[i] - quantized
                # Store quantized value
                output[i] = quantized
                # Apply error feedback to next sample if not at the end
                if i < output.shape[0] - 1:
                    output[i + 1] += error * 0.5  # Feedback coefficient
    else:
        # Standard quantization without noise shaping
        output = torch.round(output * scale) / scale
    
    # Ensure output is in valid range [-1, 1]
    output = torch.clamp(output, -1.0, 1.0 - step)
    
    return output

def save_audio_with_dithering(audio, file_path, sample_rate=44100, bits=16, dither_type='tpdf', noise_shaping=True, format="WAV"):
    """
    Save audio with proper dithering to the specified bit depth.
    
    Args:
        audio: Audio tensor (float32, expected to be in range [-1, 1])
        file_path: Output file path
        sample_rate: Sample rate in Hz
        bits: Target bit depth (16 for CD quality, 24 for high-res)
        dither_type: Type of dithering to apply ('none', 'rpdf', 'tpdf', 'gaussian')
        noise_shaping: Whether to apply noise shaping for better quality at low bit depths
        format: Output file format (default WAV)
        
    Returns:
        None
    """
    # Make sure input is a tensor
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    
    # Apply dithering
    dithered_audio = apply_dither(audio, bits=bits, dither_type=dither_type, noise_shaping=noise_shaping)
    
    # Save audio file
    torchaudio.save(
        file_path,
        dithered_audio,
        sample_rate,
        bits_per_sample=bits,
        format=format
    )
    
    print(f"Saved audio to {file_path} with {bits}-bit depth and {dither_type} dithering")

def load_audio_stereo(file_path, target_sr=None):
    """
    Load audio file and ensure it's stereo
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (if None, use original)
        
    Returns:
        Stereo audio tensor and sample rate
    """
    # Load audio
    audio, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr
    
    # Convert to stereo if mono
    if audio.shape[0] == 1:
        # Duplicate mono channel to create stereo
        audio = audio.repeat(2, 1)
    
    return audio, sr

def save_audio_stereo(audio, file_path, sample_rate=44100, bits=16, dither_type='tpdf', noise_shaping=True, rescale=False):
    """
    Save stereo audio with proper dithering
    
    Args:
        audio: Audio tensor (expected stereo)
        file_path: Output file path
        sample_rate: Sample rate in Hz
        bits: Bit depth
        dither_type: Dithering type
        noise_shaping: Whether to apply noise shaping for better quality
        rescale: Whether to rescale audio to avoid clipping
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Rescale audio if requested to avoid clipping
    if rescale:
        limit = 0.99
        max_val = audio.abs().max()
        audio = audio * min(limit / max_val, 1) if max_val > 0 else audio
    else:
        # Just clamp to safe range
        audio = audio.clamp(-0.99, 0.99)
    
    # Ensure audio is stereo
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).repeat(2, 1)
    elif audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    
    # Use dithering-enhanced save function
    save_audio_with_dithering(audio, file_path, sample_rate, bits, dither_type, noise_shaping) 