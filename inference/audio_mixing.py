import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np

from dsp.compression import (
    apply_compression,
    apply_look_ahead_limiter, 
    apply_soft_clipper
)
from dsp.phase import (
    align_phases, 
    multiband_phase_alignment
)
from dsp.utils import normalize, deep_merge_dicts
from dsp.imaging import enhance_stereo_width
from dsp.filtering import (
    enhance_vocals,
    carve_space_for_vocals,
    spectral_balance,
    apply_bandpass
)
from dsp.saturation import (
    apply_saturation,
    multi_band_saturation,
    exciter
)
from dsp.reverb import (
    apply_reverb,
    add_space
)

def mix_tracks_basic(vocal, instrumental, vocal_gain=1.0, instrumental_gain=0.8):
    return ((vocal*vocal_gain)+(instrumental*instrumental_gain))/(vocal_gain+instrumental_gain)

def enhanced_audio_mix(vocal, instrumental, mix_params=None, sr=44100):
    """
    Integrated pipeline that:
     1. Normalizes
     2. Phase aligns (optionally multi-band)
     3. (Optional) compress vocals & instrumentals individually
     4. (Optional) enhance vocals with vocal-specific EQ
     5. (Optional) carve space for vocals in instrumental
     6. (Optional) apply saturation to instrumental for warmth
     7. Gains
     8. (Optional) sidechain
     9. Summation
     10. (Optional) multi-band compression
     11. (Optional) exciter for high-end clarity
     12. (Optional) multi-band saturation
     13. (Optional) spectral balancing for professional frequency curve
     14. (Optional) reverb/ambience
     15. (Optional) stereo width
     16. (Optional) look-ahead limiting or soft clip
     ...
    """
    # default
    DEFAULT_PARAMS = {
        'vocal_gain': 1.0,
        'instrumental_gain': 0.8,
        'vocal_compression': {'enabled': True, 'threshold': -20.0, 'ratio': 2.0},
        'instrumental_compression': {'enabled': False, 'threshold': -24.0, 'ratio':1.5},
        'multiband_compression': {
            'enabled': True,
            'bands': [(0,250),(250,2000),(2000,8000),(8000,22050)],
            'thresholds':[-24,-18,-18,-16],
            'ratios':[2.5,2.0,1.8,1.5]
        },
        'phase_alignment': {
            'enabled': True,
            'multiband': True,
            'freq_dependent': True,
            'transient_preservation': True,
            'phase_locking': True
        },
        'normalization': {
            'enabled': True,
            'target_lufs': -14.0
        },
        'sidechain': {'enabled':False, 'threshold':-24.0, 'ratio':2.0},
        'stereo_width': {'enabled':True, 'width':1.2},
        'lookahead_limiter': {'enabled':True, 'threshold': -1.0, 'attack':0.001, 'release':0.05},
        'soft_clip': {'enabled':False, 'threshold':0.8, 'softness':0.1},
        # New features
        'vocal_enhancement': {
            'enabled': True, 
            'level': 0.7
        },
        'vocal_space_carving': {
            'enabled': True,
            'level': 0.6
        },
        'instrumental_saturation': {
            'enabled': True,
            'amount': 0.3,
            'type': 'tube'
        },
        'exciter': {
            'enabled': True,
            'amount': 0.4,
            'frequency': 3000
        },
        'multiband_saturation': {
            'enabled': False,
            'bands': [(0,250),(250,2000),(2000,8000),(8000,22050)],
            'amounts': [0.6, 0.4, 0.2, 0.1],
            'types': ['tube', 'tanh', 'soft_clip', 'tanh']
        },
        'spectral_balance': {
            'enabled': True,
            'strength': 0.7
        },
        'reverb': {
            'enabled': False,
            'mix': 0.2,
            'room_size': 0.7,
            'damping': 0.5,
            'pre_delay_ms': 20
        },
        'ambience': {
            'enabled': False,
            'space_type': 'medium_room',
            'mix': 0.15
        }
    }
    if mix_params is None:
        params = DEFAULT_PARAMS
    else:
        # merge
        params = deep_merge_dicts(DEFAULT_PARAMS, mix_params)

    try:
        # Quick checks
        if not isinstance(vocal, torch.Tensor) or not isinstance(instrumental, torch.Tensor):
            raise TypeError("Vocal / Instrumental must be Tensors.")
        if torch.isnan(vocal).any() or torch.isnan(instrumental).any():
            raise ValueError("NaN in input signals.")
        
        # 1) normalization
        if params['normalization']['enabled']:
            try:
                target_lufs = params['normalization']['target_lufs']
                vocal, instrumental = normalize(vocal, instrumental, target_lufs, sr)
            except Exception as e:
                print(f"Warning: normalization failed: {e}")
        
        # 2) phase alignment
        pa = params['phase_alignment']
        if pa['enabled']:
            try:
                if pa['multiband']:
                    # user-chosen bands for alignment or default
                    align_bands = [(0,250),(250,1200),(1200,4000),(4000,20000)]
                    instrumental = multiband_phase_alignment(vocal, instrumental, align_bands, sr,
                                                            pa['freq_dependent'],
                                                            pa['transient_preservation'],
                                                            pa['phase_locking'])
                else:
                    instrumental = align_phases(vocal, instrumental,
                                                enable_freq_dependent=pa['freq_dependent'],
                                                enable_transient_preservation=pa['transient_preservation'],
                                                enable_phase_locking=pa['phase_locking'])
            except Exception as e:
                print(f"Warning: phase alignment failed: {e}")
        
        # 3) compress individually
        vc = params['vocal_compression']
        if vc['enabled']:
            vocal = apply_compression(vocal, vc['threshold'], vc['ratio'], sr=sr)
        ic = params['instrumental_compression']
        if ic['enabled']:
            instrumental = apply_compression(instrumental, ic['threshold'], ic['ratio'], sr=sr)
        
        # 4) vocal enhancement
        ve = params['vocal_enhancement']
        if ve['enabled']:
            try:
                vocal = enhance_vocals(vocal, level=ve['level'], sr=sr)
            except Exception as e:
                print(f"Warning: vocal enhancement failed: {e}")
        
        # 5) carve space for vocals in instrumental
        vsc = params['vocal_space_carving']
        if vsc['enabled']:
            try:
                instrumental = carve_space_for_vocals(instrumental, vocal, level=vsc['level'], sr=sr)
            except Exception as e:
                print(f"Warning: vocal space carving failed: {e}")
        
        # 6) instrumental saturation for warmth
        is_params = params['instrumental_saturation']
        if is_params['enabled']:
            try:
                instrumental = apply_saturation(
                    instrumental, 
                    amount=is_params['amount'], 
                    saturation_type=is_params['type']
                )
            except Exception as e:
                print(f"Warning: instrumental saturation failed: {e}")
        
        # 7) Gains
        vg = params['vocal_gain']
        ig = params['instrumental_gain']
        vocal = vocal*vg
        instrumental = instrumental*ig
        
        # 8) sidechain
        sc = params['sidechain']
        if sc['enabled']:
            # sidechain compress instrumental with vocal
            instrumental = apply_compression(instrumental, sc['threshold'], sc['ratio'], sr=sr,
                                             sidechain_signal=vocal)
        
        # 9) sum
        mixed = vocal + instrumental
        max_val = mixed.abs().max()
        if max_val>1.0:
            mixed = mixed*(1.0/max_val)  # quick fix
        
        # 10) multi-band comp on the mix
        mbc = params['multiband_compression']
        if mbc['enabled']:
            if len(mbc['bands'])==len(mbc['thresholds'])==len(mbc['ratios']):
                mixed = apply_compression(
                    mixed, sr=sr,
                    multiband_bands=mbc['bands'],
                    multiband_thresholds=mbc['thresholds'],
                    multiband_ratios=mbc['ratios']
                )
            else:
                print("Warning: multi-band compression param mismatch.")
        
        # 11) exciter for high-end clarity
        ex = params['exciter']
        if ex['enabled']:
            try:
                mixed = exciter(mixed, amount=ex['amount'], freq=ex['frequency'], sr=sr)
            except Exception as e:
                print(f"Warning: exciter failed: {e}")
        
        # 12) multi-band saturation
        mbs = params['multiband_saturation']
        if mbs['enabled']:
            try:
                if len(mbs['bands'])==len(mbs['amounts']):
                    mixed = multi_band_saturation(
                        mixed, 
                        bands=mbs['bands'],
                        amounts=mbs['amounts'],
                        types=mbs.get('types', None),
                        sr=sr
                    )
                else:
                    print("Warning: multi-band saturation param mismatch.")
            except Exception as e:
                print(f"Warning: multi-band saturation failed: {e}")
        
        # 13) spectral balancing
        sb = params['spectral_balance']
        if sb['enabled']:
            try:
                mixed = spectral_balance(mixed, strength=sb['strength'], sr=sr)
            except Exception as e:
                print(f"Warning: spectral balancing failed: {e}")
        
        # 14) reverb/ambience
        rv = params['reverb']
        if rv['enabled']:
            try:
                mixed = apply_reverb(
                    mixed, 
                    mix=rv['mix'],
                    room_size=rv['room_size'],
                    damping=rv['damping'],
                    pre_delay_ms=rv['pre_delay_ms'],
                    sr=sr
                )
            except Exception as e:
                print(f"Warning: reverb failed: {e}")
        
        amb = params['ambience']
        if amb['enabled'] and not rv['enabled']:  # only apply if reverb isn't already applied
            try:
                mixed = add_space(
                    mixed,
                    space_type=amb['space_type'],
                    mix=amb['mix'],
                    sr=sr
                )
            except Exception as e:
                print(f"Warning: ambience failed: {e}")
        
        # 15) stereo width
        sw = params['stereo_width']
        if sw['enabled'] and mixed.dim()>1 and mixed.shape[0]>=2:
            w = max(0.0, min(2.0, sw['width']))
            mixed = enhance_stereo_width(mixed, w)
        
        # 16) final limiting or soft clip
        la = params['lookahead_limiter']
        scp = params['soft_clip']
        
        final = mixed
        if la['enabled']:
            try:
                final = apply_look_ahead_limiter(final, la['threshold'], la['release'], la['attack'], 0.005, sr=sr)
            except Exception as e:
                print(f"Warning: limiter failed: {e}")
                final = torch.clamp(final, -1.0, 1.0)
        
        if scp['enabled']:
            try:
                final = apply_soft_clipper(final, scp['threshold'], scp['softness'])
            except Exception as e:
                print(f"Warning: soft clip failed: {e}")
                final = torch.clamp(final, -1.0, 1.0)
        
        # Safety check
        if torch.isnan(final).any() or torch.isinf(final).any():
            print("Warning: NaN/Inf in final. Fallback to simpler half-gain mix.")
            final = 0.5*(vocal + instrumental)
        
        return final
    except Exception as e:
        print(f"Critical error in enhanced_audio_mix: {e}")
        # fallback
        safe_mix = 0.5*(vocal + instrumental)
        return safe_mix

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
        compressed = apply_compression(band_signal, thresholds[i], ratios[i], sr=sr)
        compressed_bands.append(compressed)
    
    # Sum the compressed bands
    result = sum(compressed_bands)
    
    return result

def apply_gain_staging(audio, gain_db=0.0):
    """
    Apply gain staging to audio
    
    Args:
        audio: Audio tensor
        gain_db: Gain in dB to apply
        
    Returns:
        Gain-staged audio
    """
    return audio * (10 ** (gain_db / 20))

def process_files_with_enhancements(vocal_path, instrumental_path, output_path, mix_params=None):
    """
    Loads vocal, loads instrumental, ensures same SR, calls enhanced_audio_mix,
    then saves the result. Keeps your advanced approach for ensuring the user's 
    chosen sample rate is consistent, etc.
    """
    vocal, sr_v = torchaudio.load(vocal_path)
    instrumental, sr_i = torchaudio.load(instrumental_path)
    
    # unify SR
    if sr_v != sr_i:
        rsmpl = Resample(orig_freq=sr_i, new_freq=sr_v)
        instrumental = rsmpl(instrumental)
        sr = sr_v
    else:
        sr = sr_v
    
    final = enhanced_audio_mix(vocal, instrumental, mix_params, sr)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, final.cpu(), sr)
    return output_path