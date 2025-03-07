import os
import torch
import torchaudio
import soundfile as sf
from torchaudio.transforms import Resample

# Import our enhanced mixing functions
from audio_mixing import process_files_with_enhancements, enhanced_audio_mix

def load_audio_mono(filepath, sampling_rate=16000):
    """
    Load audio file and convert to mono
    
    Args:
        filepath: Path to audio file
        sampling_rate: Target sampling rate
    
    Returns:
        Tensor of shape [1, samples]
    """
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def load_audio_stereo(filepath, sampling_rate=16000):
    """
    Load audio file with stereo channel preservation
    
    Args:
        filepath: Path to audio file
        sampling_rate: Target sampling rate
    
    Returns:
        Tensor of shape [channels, samples]
    """
    audio, sr = torchaudio.load(filepath)
    
    # Keep stereo if it exists, otherwise duplicate mono to create stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)  # Duplicate mono to stereo
        
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    
    return audio

def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    """
    Save audio to file
    
    Args:
        wav: Audio tensor
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
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def save_audio_stereo(wav, path, sample_rate, rescale=False):
    """
    Save audio with stereo preservation
    
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
    
    # Ensure stereo format (2 channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)  # Convert mono to stereo
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # Convert mono to stereo
        
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

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