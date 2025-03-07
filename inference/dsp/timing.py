import torch
import numpy as np
from dsp.utils import to_mono, apply_per_channel

def find_time_offset(reference, target, max_offset_ms=100, sr=44100):
    max_off_samps = int(sr*max_offset_ms/1000)
    ref_mono = to_mono(reference)
    tgt_mono = to_mono(target)
    length = min(ref_mono.shape[-1], tgt_mono.shape[-1])
    ref_mono = ref_mono[..., :length]
    tgt_mono = tgt_mono[..., :length]
    
    corr = torch.nn.functional.conv1d(
        ref_mono.unsqueeze(0).unsqueeze(0),
        tgt_mono.flip(0).unsqueeze(0).unsqueeze(0),
        padding=max_off_samps
    )
    _, peak_idx = torch.max(corr, dim=2)
    offset = peak_idx.item() - max_off_samps
    return offset

def apply_time_offset(audio, offset, mode='shift', sr=44100, fft_size=2048, hop_size=512):
    """
    Apply time offset to audio signal.
    
    Args:
        audio: Input audio tensor with shape [C, N] or [N]
        offset: Time offset in samples (positive = delay, negative = advance)
        mode: 'shift' for simple zero-padding shift, 'stretch' for phase vocoder approach
        sr: Sample rate for stretch mode
        fft_size: FFT size for STFT in stretch mode
        hop_size: Hop size for STFT in stretch mode
        
    Returns:
        Time-offset audio
    """
    if mode == 'shift':
        def _shift_single(ch):
            if offset==0:
                return ch
            out = torch.zeros_like(ch)
            if offset>0:
                out[offset:] = ch[:-offset]
            else:
                out[:offset] = ch[-offset:]
            return out
        return apply_per_channel(audio, _shift_single)
    
    elif mode == 'stretch':
        def _stretch_single(ch):
            if offset == 0:
                return ch
                
            # Calculate the stretch factor based on the offset and length
            length = ch.shape[-1]
            target_length = length - offset
            stretch_factor = target_length / length
            
            # Apply phase vocoder for time stretching
            window = torch.hann_window(fft_size, device=ch.device if ch.is_cuda else 'cpu')
            
            # Compute STFT
            stft = torch.stft(ch, fft_size, hop_size, window=window, return_complex=True)
            
            # Get magnitude and phase
            mag = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Calculate phase advance (horizontal phase coherence)
            phase_advance = torch.zeros_like(phase)
            phase_advance[:, 1:] = phase[:, 1:] - phase[:, :-1]
            
            # Unwrap phase advance to avoid discontinuities
            phase_advance_unwrap = torch.from_numpy(np.unwrap(phase_advance.cpu().numpy(), axis=1)).to(phase.device)
            
            # Calculate new number of frames based on stretch factor
            n_frames = stft.shape[1]
            new_n_frames = int(n_frames / stretch_factor)
            
            # Create new STFT with modified time scale
            new_stft = torch.zeros((stft.shape[0], new_n_frames), dtype=torch.complex64, device=stft.device)
            new_mag = torch.zeros((stft.shape[0], new_n_frames), device=mag.device)
            new_phase = torch.zeros((stft.shape[0], new_n_frames), device=phase.device)
            
            # Initialize first frame
            new_phase[:, 0] = phase[:, 0]
            new_mag[:, 0] = mag[:, 0]
            
            # Phase vocoder: time-stretch by adjusting phase increment
            for f in range(1, new_n_frames):
                # Map to the original frame index (non-integer)
                orig_frame = f * stretch_factor
                
                # Get the two closest frames for interpolation
                frame_idx1 = int(np.floor(orig_frame))
                frame_idx2 = min(frame_idx1 + 1, n_frames - 1)
                alpha = orig_frame - frame_idx1  # interpolation weight
                
                # Interpolate magnitude
                if frame_idx1 < n_frames:
                    new_mag[:, f] = (1 - alpha) * mag[:, frame_idx1] + alpha * mag[:, frame_idx2]
                    
                    # Calculate phase based on the phase advance
                    new_phase[:, f] = new_phase[:, f-1] + phase_advance_unwrap[:, frame_idx1] * stretch_factor
                
            # Reconstruct complex STFT
            new_stft = torch.polar(new_mag, new_phase)
            
            # Convert back to time domain
            stretched_signal = torch.istft(new_stft, fft_size, hop_size, window=window)
            
            # Resize to match target length if needed
            if stretched_signal.shape[-1] != target_length:
                if stretched_signal.shape[-1] > target_length:
                    stretched_signal = stretched_signal[:target_length]
                else:
                    padded = torch.zeros(target_length, device=stretched_signal.device)
                    padded[:stretched_signal.shape[-1]] = stretched_signal
                    stretched_signal = padded
            
            return stretched_signal
            
        return apply_per_channel(audio, _stretch_single)
    
    else:
        raise ValueError(f"Unknown mode: {mode}, should be 'shift' or 'stretch'")