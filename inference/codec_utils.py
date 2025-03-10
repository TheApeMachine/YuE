import os
import numpy as np
import torch
from einops import rearrange
from transformers import LogitsProcessor, LogitsProcessorList

def seed_everything(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BlockTokenRangeProcessor(LogitsProcessor):
    """
    LogitsProcessor that blocks specific token ranges
    """
    def __init__(self, start_id, end_id):
        self.start_id = start_id
        self.end_id = end_id
    
    def __call__(self, input_ids, scores):
        scores[:, self.start_id:self.end_id] = -float('inf')
        return scores

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    """
    Encode mono audio into codec tokens
    
    Args:
        codec_model: Neural codec model
        audio_prompt: Audio tensor
        device: Processing device
        target_bw: Target bandwidth
    
    Returns:
        Encoded tokens
    """
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
        
    with torch.no_grad():
        codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
        
    codes = codes.transpose(0, 1).cpu().numpy().astype(np.int16)
    return codes

def encode_audio_stereo(codec_model, audio_prompt, device, target_bw=0.5):
    """
    Encode stereo audio into codec tokens
    
    Args:
        codec_model: Neural codec model
        audio_prompt: Audio tensor of shape [channels, samples]
        device: Processing device
        target_bw: Target bandwidth
        
    Returns:
        Encoded tokens with stereo information preserved
    """
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
        
    # Split stereo channels
    left_channel = audio_prompt[:, 0:1, :]
    right_channel = audio_prompt[:, 1:2, :]
    
    with torch.no_grad():
        # Encode each channel separately
        left_codes = codec_model.encode(left_channel.to(device), target_bw=target_bw)
        right_codes = codec_model.encode(right_channel.to(device), target_bw=target_bw)
        
    # Transform to format expected by the tokenizer
    left_codes = left_codes.transpose(0, 1)
    right_codes = right_codes.transpose(0, 1)
    
    # Convert to numpy arrays
    left_codes = left_codes.cpu().numpy().astype(np.int16)
    right_codes = right_codes.cpu().numpy().astype(np.int16)
    
    return left_codes, right_codes

def decode_audio(codec_model, codes, device):
    """
    Decode audio from codec tokens
    
    Args:
        codec_model: Neural codec model
        codes: Encoded tokens
        device: Processing device
        
    Returns:
        Decoded audio tensor
    """
    with torch.no_grad():
        waveform = codec_model.decode(
            torch.as_tensor(codes.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
    
    return waveform.cpu().squeeze(0)

def decode_stereo_audio(codec_model, left_codes, right_codes, device):
    """
    Decode stereo audio from separate channel codes
    
    Args:
        codec_model: Neural codec model
        left_codes: Encoded tokens for left channel
        right_codes: Encoded tokens for right channel
        device: Processing device
        
    Returns:
        Stereo audio tensor
    """
    with torch.no_grad():
        # Decode each channel
        left_waveform = codec_model.decode(
            torch.as_tensor(left_codes.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
        
        right_waveform = codec_model.decode(
            torch.as_tensor(right_codes.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
    
    # Combine channels
    left_waveform = left_waveform.cpu().squeeze(0)
    right_waveform = right_waveform.cpu().squeeze(0)
    
    # Stack to create stereo
    stereo_waveform = torch.stack([left_waveform, right_waveform], dim=0)
    
    return stereo_waveform

def split_lyrics(lyrics):
    """
    Split lyrics into phrases based on punctuation
    
    Args:
        lyrics: Input lyrics string
        
    Returns:
        List of lyric phrases
    """
    # Replace common punctuation with period to standardize splits
    for punct in [",", ";", ":", "!", "?"]:
        lyrics = lyrics.replace(punct, ".")
    
    # Remove double periods
    while ".." in lyrics:
        lyrics = lyrics.replace("..", ".")
    
    # Split by period and filter out empty strings
    phrases = [phrase.strip() for phrase in lyrics.split(".") if phrase.strip()]
    
    return phrases 