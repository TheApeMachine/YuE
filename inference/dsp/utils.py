import torch

def to_mono(audio):
    """
    Convert multi-channel to mono by averaging. If already mono, no change.
    """
    if audio.dim()>1 and audio.shape[0]>1:
        return audio.mean(dim=0)
    return audio.squeeze(0) if audio.dim()>1 else audio

# Import after to_mono is defined to avoid circular import
from dsp.metering import measure_lufs

def normalize(vocal, instrumental, target_lufs=-16.0, sr=44100):
    """
    Same logic from original: measure LUFS of each track, compute gain to bring each to target.
    """
    vocal_lufs = measure_lufs(vocal, sr)
    inst_lufs = measure_lufs(instrumental, sr)
    
    v_gain = 10**((target_lufs - vocal_lufs)/20)
    i_gain = 10**((target_lufs - inst_lufs)/20)
    
    return vocal*v_gain, instrumental*i_gain

def apply_per_channel(audio, func, *args, **kwargs):
    """
    Apply a single-channel function across all channels in the audio tensor,
    returning the same shape. 
    shape(audio) => [C, N] or [N].
    """
    if audio.dim()==1:
        # Single channel
        return func(audio, *args, **kwargs)
    else:
        # multiple channels
        outs = []
        for ch_idx in range(audio.shape[0]):
            out_ch = func(audio[ch_idx], *args, **kwargs)
            outs.append(out_ch)
        return torch.stack(outs, dim=0)

def deep_merge_dicts(defaults, user):
    """
    Recursively merge two dicts so that user can override values in defaults.
    """
    merged = {}
    for k,v in defaults.items():
        if k not in user:
            merged[k] = v
        else:
            if isinstance(v, dict) and isinstance(user[k], dict):
                merged[k] = deep_merge_dicts(v, user[k])
            else:
                merged[k] = user[k]
    # Add any additional keys from user that are not in defaults
    for k,v in user.items():
        if k not in merged:
            merged[k] = v
    return merged
