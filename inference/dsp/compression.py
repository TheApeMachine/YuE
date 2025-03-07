import math
import torch
from YuE.inference.dsp.filtering import apply_bandpass
from dsp.utils import apply_per_channel

def apply_look_ahead_limiter(audio, threshold_db=-1.0, release_time=0.05,
                             attack_time=0.001, look_ahead_time=0.005, sr=44100):
    """
    True look-ahead limiter, sample-by-sample, with a single pass over time. 
    We handle multi-channel in parallel. 
    """
    # shape => [channels, samples]
    if audio.dim()==1:
        audio = audio.unsqueeze(0)
    thr_lin = 10**(threshold_db/20)
    
    a_samps = max(1, int(attack_time*sr))
    r_samps = max(1, int(release_time*sr))
    la_samps = max(1, int(look_ahead_time*sr))
    
    device = audio.device
    dtype = audio.dtype
    ch, nsamps = audio.shape
    
    out = torch.zeros_like(audio)
    # pad
    padded = torch.nn.functional.pad(audio, (0, la_samps))
    # current gain
    current_gain = torch.ones(ch, device=device, dtype=dtype)
    
    # Attack/Release
    a_coeff = math.exp(-1.0/a_samps)
    r_coeff = math.exp(-1.0/r_samps)
    
    for i in range(nsamps):
        # Look-ahead peak across la_samps
        windowed = padded[:, i:i+la_samps].abs().max(dim=1).values
        # required gain
        target_gain = torch.ones_like(windowed)
        above = windowed>thr_lin
        target_gain[above] = thr_lin/windowed[above]
        # Smooth
        need_more_reduction = target_gain<current_gain
        current_gain[need_more_reduction] = (a_coeff*current_gain[need_more_reduction] +
                                             (1-a_coeff)*target_gain[need_more_reduction])
        need_less_reduction = ~need_more_reduction
        current_gain[need_less_reduction] = (r_coeff*current_gain[need_less_reduction] +
                                             (1-r_coeff)*target_gain[need_less_reduction])
        
        out[:, i] = audio[:, i]*current_gain
    
    return out

def apply_soft_clipper(audio, threshold=0.8, softness=0.1):
    """
    Tanh-based soft clipping above the threshold region.
    shape => [C, N] or [N].
    """
    def _soft_clip_single(ch):
        out = ch.clone()
        mask = ch.abs()>threshold
        over = ch[mask]
        x_norm = (over.abs()-threshold)/softness
        curve = threshold + softness*torch.tanh(x_norm)
        out[mask] = torch.sign(over)*curve
        return out
    
    return apply_per_channel(audio, _soft_clip_single)

def apply_compression(
    audio: torch.Tensor,
    threshold_db: float=-20.0,
    ratio: float=2.0,
    attack_time: float=0.005,
    release_time: float=0.05,
    sr: int=44100,
    sidechain_signal: torch.Tensor=None,
    multiband_bands=None,
    multiband_thresholds=None,
    multiband_ratios=None,
    filter_order: int=4,
    chunk_size_s: float=None
):
    """
    Unified compressor that can:
      - Do single-band compression
      - Do sidechain compression (if `sidechain_signal` is provided)
      - Do multi-band compression if you pass lists for bands/thresholds/ratios
      - Optionally chunk for very large signals, preserving envelope state across chunks

    Args:
      audio: shape [C, N] or [N]
      threshold_db, ratio, attack_time, release_time: typical compressor params
      sr: sample rate
      sidechain_signal: optional, same shape or can be shorter/longer
      multiband_bands: list of (low_freq, high_freq)
      multiband_thresholds: list of dB thresholds
      multiband_ratios: list of ratios
      filter_order: order for bandpass
      chunk_size_s: if not None, we do chunk-based processing to handle extremely large signals
                    without huge memory usage. e.g. chunk_size_s=2.0 => 2s chunks

    Returns:
      compressed audio, same shape as input
    """
    # If multi-band is requested
    is_multiband = (multiband_bands is not None and len(multiband_bands)>0)
    if is_multiband:
        # Validate length match
        if not (len(multiband_bands)==len(multiband_thresholds)==len(multiband_ratios)):
            raise ValueError("Mismatch in multi-band compressor param lengths.")
    
    # If chunking is desired
    if chunk_size_s is not None and chunk_size_s>0:
        return _compress_in_chunks(
            audio, threshold_db, ratio, attack_time, release_time, sr,
            sidechain_signal, multiband_bands, multiband_thresholds, multiband_ratios,
            filter_order, chunk_size_s
        )
    else:
        # Single pass
        return _compress_entire(
            audio, threshold_db, ratio, attack_time, release_time, sr,
            sidechain_signal, multiband_bands, multiband_thresholds, multiband_ratios,
            filter_order
        )

def _compress_in_chunks(audio, threshold_db, ratio, attack_time, release_time, sr,
                        sidechain_signal, bands, thrs, rats, filter_order, chunk_size_s):
    """
    Chunk-based compression for extremely long signals. We keep an 'envelope state' 
    across chunk boundaries so it’s seamless.
    """
    # Force shape => [C, N]
    if audio.dim()==1:
        audio = audio.unsqueeze(0)
    length = audio.shape[-1]
    chunk_len = int(chunk_size_s*sr)
    
    # Prepare sidechain if needed
    if sidechain_signal is not None:
        if sidechain_signal.dim()==1:
            sidechain_signal = sidechain_signal.unsqueeze(0)
        side_len = sidechain_signal.shape[-1]
    else:
        side_len = 0
    
    out = torch.zeros_like(audio)
    start = 0
    env_state = None  # we’ll keep a dictionary if needed for multi-band or single band
    while start<length:
        end = min(start+chunk_len, length)
        audio_chunk = audio[:, start:end]
        if sidechain_signal is not None:
            side_chunk = sidechain_signal[:, start:end] if end<=side_len else sidechain_signal[:, start:side_len]
        else:
            side_chunk = None
        
        # compress
        compressed_chunk, env_state = _compress_entire(
            audio_chunk, threshold_db, ratio, attack_time, release_time, sr,
            side_chunk, bands, thrs, rats, filter_order, env_state=env_state
        )
        out[:, start:end] = compressed_chunk
        start=end
    return out

def _compress_entire(
    audio, threshold_db, ratio, attack_time, release_time, sr,
    sidechain_signal, bands, thrs, rats, filter_order,
    env_state=None
):
    """
    Actually do the single pass (or multi-band) compression. If env_state is provided,
    we continue from that envelope across chunk boundaries.
    """
    # multi-band vs single-band
    if bands and len(bands)>0:
        # multi-band
        # Force shape => [C, N]
        if audio.dim()==1:
            audio = audio.unsqueeze(0)
        band_outputs = []
        for i, (lf, hf) in enumerate(bands):
            band_thresh = thrs[i]
            band_ratio = rats[i]
            band_audio = apply_bandpass(audio, lf, hf, sr, filter_order)
            # compress band
            cband, env_state_sub = _compress_single_band(
                band_audio, band_thresh, band_ratio, attack_time, release_time, sr,
                sidechain_signal, env_state=env_state.get(f"band_{i}") if env_state else None
            )
            if env_state is None:
                env_state = {}
            env_state[f"band_{i}"] = env_state_sub
            band_outputs.append(cband)
        # sum
        out = torch.stack(band_outputs, dim=0).sum(dim=0)
        return out, env_state
    else:
        # single-band
        out, new_state = _compress_single_band(
            audio, threshold_db, ratio, attack_time, release_time, sr,
            sidechain_signal, env_state=env_state
        )
        return out, new_state

def _compress_single_band(audio, threshold_db, ratio, attack_time, release_time, sr,
                          sidechain_signal=None, env_state=None):
    """
    The core compressor for a single band or wideband. 
    If sidechain_signal is present, we drive the envelope from that instead. 
    env_state is a dict containing e.g. {"env": <torch tensor [channels]>} so we can pass it across chunks.
    """
    # shape => [C, N]
    if audio.dim()==1:
        audio = audio.unsqueeze(0)
    if sidechain_signal is not None:
        if sidechain_signal.dim()==1:
            sidechain_signal = sidechain_signal.unsqueeze(0)
        length = min(audio.shape[-1], sidechain_signal.shape[-1])
        audio = audio[..., :length]
        sidechain_signal = sidechain_signal[..., :length]
        envelope_source = sidechain_signal
    else:
        envelope_source = audio
        length = audio.shape[-1]
    
    # set up
    dev = audio.device
    dtp = audio.dtype
    thr_lin = 10**(threshold_db/20)
    ratio = max(1.0, ratio)
    
    a_time = max(1e-9, attack_time)
    r_time = max(1e-9, release_time)
    a_coeff = math.exp(-1.0/(a_time*sr))
    r_coeff = math.exp(-1.0/(r_time*sr))
    
    # allocate
    out = torch.empty_like(audio)
    channels = audio.shape[0]
    # if we have an existing state
    if env_state and "env" in env_state:
        env = env_state["env"].clone()
    else:
        env = torch.zeros(channels, device=dev, dtype=dtp)
    
    # do the recursion
    abs_src = envelope_source.abs()
    for n in range(length):
        bigger = abs_src[:, n]>env
        env[bigger] = a_coeff*env[bigger] + (1-a_coeff)*abs_src[bigger, n]
        smaller = ~bigger
        env[smaller] = r_coeff*env[smaller] + (1-r_coeff)*abs_src[smaller, n]
        
        # gain
        above = env>thr_lin
        gain = torch.ones_like(env)
        over = env[above]-thr_lin
        gain[above] = (thr_lin + over/ratio)/(env[above]+1e-10)
        
        out[:, n] = audio[:, n]*gain
    
    new_state = {"env": env}
    return out, new_state