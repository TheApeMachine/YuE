import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np

from YuE.inference.dsp.compression import apply_compression
from YuE.inference.dsp.phase import (
    align_phases, 
    multiband_phase_alignment, 
    apply_look_ahead_limiter, 
    apply_soft_clipper
)
from dsp.utils import normalize, deep_merge_dicts
from dsp.imaging import enhance_stereo_width

def mix_tracks_basic(vocal, instrumental, vocal_gain=1.0, instrumental_gain=0.8):
    return ((vocal*vocal_gain)+(instrumental*instrumental_gain))/(vocal_gain+instrumental_gain)

def enhanced_audio_mix(vocal, instrumental, mix_params=None, sr=44100):
    """
    Integrated pipeline that:
     1. Normalizes
     2. Phase aligns (optionally multi-band)
     3. (Optional) compress vocals & instrumentals individually
     4. Gains
     5. (Optional) sidechain
     6. Summation
     7. (Optional) multi-band compression
     8. (Optional) stereo width
     9. (Optional) look-ahead limiting or soft clip
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
        'soft_clip': {'enabled':False, 'threshold':0.8, 'softness':0.1}
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
        
        # 4) Gains
        vg = params['vocal_gain']
        ig = params['instrumental_gain']
        vocal = vocal*vg
        instrumental = instrumental*ig
        
        # 5) sidechain
        sc = params['sidechain']
        if sc['enabled']:
            # sidechain compress instrumental with vocal
            instrumental = apply_compression(instrumental, sc['threshold'], sc['ratio'], sr=sr,
                                             sidechain_signal=vocal)
        
        # 6) sum
        mixed = vocal + instrumental
        max_val = mixed.abs().max()
        if max_val>1.0:
            mixed = mixed*(1.0/max_val)  # quick fix
        
        # 7) multi-band comp on the mix
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
        
        # 8) stereo width
        sw = params['stereo_width']
        if sw['enabled'] and mixed.dim()>1 and mixed.shape[0]>=2:
            w = max(0.0, min(2.0, sw['width']))
            mixed = enhance_stereo_width(mixed, w)
        
        # 9) final limiting or soft clip
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

def process_files_with_enhancements(vocal_path, instrumental_path, output_path, mix_params=None):
    """
    Loads vocal, loads instrumental, ensures same SR, calls enhanced_audio_mix,
    then saves the result. Keeps your advanced approach for ensuring the user’s 
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