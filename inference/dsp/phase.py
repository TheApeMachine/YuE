import torch
import numpy as np
from dsp.filtering import apply_bandpass
from dsp.utils import to_mono

def align_phases(reference, target, fft_size=2048, hop_size=512,
                 enable_freq_dependent=True, enable_transient_preservation=True,
                 enable_phase_locking=True, sr=44100):
    """
    Advanced phase alignment that preserves transients and maintains horizontal phase coherence.
    
    This implementation avoids the artifacts common in simplistic phase replacement by:
    1. Using frequency-dependent phase processing to reduce timbral distortion
    2. Preserving transients in the target signal to maintain impact
    3. Applying horizontal phase consistency (phase locking) to reduce artifacts
    4. Handling multi-channel audio properly
    
    Args:
        reference: Reference audio signal [C, N] or [N]
        target: Target audio signal to be phase-aligned [C, N] or [N]
        fft_size: Size of FFT window
        hop_size: Hop size for STFT
        enable_freq_dependent: Apply frequency-dependent phase processing
        enable_transient_preservation: Preserve transients in the target signal
        enable_phase_locking: Apply phase locking within critical bands
        sr: Sample rate in Hz
    
    Returns:
        Phase-aligned audio with target magnitude and reference phase characteristics
    """
    # Handle multi-channel cases
    if reference.dim() > 1 and reference.shape[0] > 1:
        # Multi-channel reference
        out_channels = []
        for ch in range(reference.shape[0]):
            if target.dim() > 1 and target.shape[0] > 1:
                # Multi-channel target
                aligned_ch = _align_single_channel(
                    reference[ch], target[ch], fft_size, hop_size,
                    enable_freq_dependent, enable_transient_preservation,
                    enable_phase_locking, sr
                )
            else:
                # Mono target
                aligned_ch = _align_single_channel(
                    reference[ch], target, fft_size, hop_size,
                    enable_freq_dependent, enable_transient_preservation,
                    enable_phase_locking, sr
                )
            out_channels.append(aligned_ch)
        return torch.stack(out_channels, dim=0)
    
    elif target.dim() > 1 and target.shape[0] > 1:
        # Mono reference, multi-channel target
        out_channels = []
        for ch in range(target.shape[0]):
            aligned_ch = _align_single_channel(
                reference, target[ch], fft_size, hop_size,
                enable_freq_dependent, enable_transient_preservation,
                enable_phase_locking, sr
            )
            out_channels.append(aligned_ch)
        return torch.stack(out_channels, dim=0)
    
    else:
        # Both mono or single channel
        return _align_single_channel(
            reference, target, fft_size, hop_size,
            enable_freq_dependent, enable_transient_preservation,
            enable_phase_locking, sr
        )

def multiband_phase_alignment(reference, target, bands, sr=44100,
                             enable_freq_dependent=True,
                             enable_transient_preservation=True,
                             enable_phase_locking=True):
    """
    Splits signals into bands, calls align_phases on each band, sums them up.
    Preserves logic from your original "multiband_phase_alignment" function.
    """
    # If multi-channel, handle channels separately (like the original code).
    if reference.dim()>1 and reference.shape[0]>1:
        # For each channel
        out_channels = []
        for ch in range(reference.shape[0]):
            if target.dim()>1 and target.shape[0]>1:
                aligned_ch = multiband_phase_alignment(reference[ch], target[ch], bands, sr,
                                                      enable_freq_dependent, enable_transient_preservation,
                                                      enable_phase_locking)
            else:
                aligned_ch = multiband_phase_alignment(reference[ch], target, bands, sr,
                                                      enable_freq_dependent, enable_transient_preservation,
                                                      enable_phase_locking)
            out_channels.append(aligned_ch)
        return torch.stack(out_channels, dim=0)
    if target.dim()>1 and target.shape[0]>1:
        # multiple channels in target
        out_channels = []
        for ch in range(target.shape[0]):
            aligned_ch = multiband_phase_alignment(reference, target[ch], bands, sr,
                                                  enable_freq_dependent, enable_transient_preservation,
                                                  enable_phase_locking)
            out_channels.append(aligned_ch)
        return torch.stack(out_channels, dim=0)
    
    # single-channel or effectively so
    filtered_refs = []
    filtered_tars = []
    for (lo, hi) in bands:
        ref_band = apply_bandpass(reference, lo, hi, sr)
        tar_band = apply_bandpass(target, lo, hi, sr)
        filtered_refs.append(ref_band)
        filtered_tars.append(tar_band)
    
    aligned_bands = []
    for i, (lo, hi) in enumerate(bands):
        # Decide FFT size from freq range
        if lo<250:
            fft_s = 4096
        elif lo<2000:
            fft_s = 2048
        else:
            fft_s = 1024
        hop_s = fft_s//4
        
        band_aligned = align_phases(filtered_refs[i], filtered_tars[i],
                                    fft_size=fft_s, hop_size=hop_s,
                                    enable_freq_dependent=enable_freq_dependent,
                                    enable_transient_preservation=enable_transient_preservation,
                                    enable_phase_locking=enable_phase_locking,
                                    sr=sr)
        aligned_bands.append(band_aligned)
    
    return sum(aligned_bands)

def _align_single_channel(reference, target, fft_size, hop_size,
                          enable_freq_dependent=True,
                          enable_transient_preservation=True,
                          enable_phase_locking=True,
                          sr=44100):
    """
    The main single-channel alignment from your original code.
    We keep your advanced flags (freq-dep, transient, etc.).
    Parameters:
        reference: Reference audio signal
        target: Target audio signal to be phase-aligned
        fft_size: Size of FFT window
        hop_size: Hop size for STFT
        enable_freq_dependent: Apply frequency-dependent phase processing
        enable_transient_preservation: Preserve transients in the target signal
        enable_phase_locking: Apply phase locking within critical bands
        sr: Sample rate in Hz
    """
    
    ref_m = to_mono(reference)
    tgt_m = to_mono(target)
    length = min(ref_m.shape[-1], tgt_m.shape[-1])
    ref_m = ref_m[..., :length]
    tgt_m = tgt_m[..., :length]
    
    window = torch.hann_window(fft_size, device=ref_m.device if ref_m.is_cuda else 'cpu')
    ref_stft = torch.stft(ref_m, fft_size, hop_size, window=window, return_complex=True)
    tgt_stft = torch.stft(tgt_m, fft_size, hop_size, window=window, return_complex=True)
    
    ref_mag = ref_stft.abs()
    ref_phase = torch.angle(ref_stft)
    tgt_mag = tgt_stft.abs()
    tgt_phase = torch.angle(tgt_stft)
    
    # Phase diff
    rp_diff = torch.zeros_like(ref_phase)
    rp_diff[:, 1:] = ref_phase[:, 1:] - ref_phase[:, :-1]
    
    # unwrap
    rp_diff_unwrap = torch.from_numpy(np.unwrap(rp_diff.cpu().numpy(), axis=1)).to(rp_diff.device)
    
    # Initialize frequency dependent scaling factors if needed
    freq_scaling = None
    if enable_freq_dependent:
        # Create frequency-dependent scaling factors (more conservative in low frequencies, 
        # more aggressive in high frequencies where phase is less perceptually important)
        num_freqs = ref_phase.shape[0]
        freq_scaling = torch.linspace(0.7, 1.0, num_freqs, device=ref_phase.device)
        freq_scaling = freq_scaling.unsqueeze(1).expand_as(ref_phase)
    
    # Define critical bands for phase locking if needed
    critical_bands = None
    if enable_phase_locking:
        # Advanced critical bands based on the Bark scale formula
        # Using the improved Bark scale formula: z = 13 * arctan(0.00076*f) + 3.5 * arctan((f/7500)²)
        def hz_to_bark(f):
            return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
        
        def bark_to_hz(z):
            # Inverse mapping using numerical approximation (more accurate than simplified formula)
            # This is an accurate approximation of the inverse Bark scale
            if z < 2:
                return 200 * z
            elif z < 20.1:
                return 104.6 * np.exp(0.403 * z)
            else:
                return 14000
        
        # Create 24 critical bands in the Bark domain (standard psychoacoustic model)
        bark_bands = np.linspace(0, 24, 25)  # 0 to 24 Bark (covering human hearing range)
        freq_bands = [bark_to_hz(bark) for bark in bark_bands]
        
        # Convert Hz to bin indices based on sample rate and FFT size
        sample_rate = sr  # Use the actual sample rate from arguments
        nyquist = sample_rate / 2
        bin_edges = [int(min(edge / nyquist * (fft_size // 2), ref_phase.shape[0]-1)) for edge in freq_bands]
        
        # Ensure bins are unique and sorted
        bin_edges = sorted(list(set(bin_edges)))
        
        # Create bands
        critical_bands = []
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] < bin_edges[i+1]:
                critical_bands.append((bin_edges[i], bin_edges[i+1]))
    
    # advanced features
    aligned_phase = torch.zeros_like(tgt_phase)
    aligned_phase[:, 0] = tgt_phase[:, 0]
    n_frames = tgt_phase.shape[1]
    
    # Possibly do a transient mask
    transient_mask = None
    if enable_transient_preservation:
        # naive approach: measure frame energy in tgt_mag
        frame_energy = torch.sum(tgt_mag**2, dim=0)
        # find large upward jumps
        e_diff = torch.zeros_like(frame_energy)
        e_diff[1:] = frame_energy[1:] - frame_energy[:-1]
        # threshold
        threshold = 1.5*torch.mean(torch.abs(e_diff))
        transient_mask = (e_diff>threshold)
        # optionally extend a few frames
        extended = torch.zeros_like(transient_mask)
        for i in range(len(transient_mask)):
            if i<len(transient_mask)-3 and (transient_mask[i] or transient_mask[i+1] or transient_mask[i+2]):
                extended[i:i+3] = True
        transient_mask = extended
    
    for f in range(1, n_frames):
        # Default phase propagation
        phase_diff = rp_diff_unwrap[:, f]
        
        # Apply frequency-dependent scaling if enabled
        if enable_freq_dependent and freq_scaling is not None:
            phase_diff = phase_diff * freq_scaling[:, f]
        
        # Compute base aligned phase
        aligned_phase[:, f] = aligned_phase[:, f-1] + phase_diff
        
        # Apply phase locking within critical bands if enabled
        if enable_phase_locking and critical_bands:
            for low_bin, high_bin in critical_bands:
                if high_bin - low_bin <= 1:
                    continue
                    
                # Calculate average phase derivative within this critical band
                avg_phase_diff = torch.mean(phase_diff[low_bin:high_bin])
                
                # Lock phases within band to this average (with some tapering at edges)
                band_width = high_bin - low_bin
                
                # Create weights that taper at the edges (1.0 in middle, 0.5 at edges)
                weights = torch.linspace(0.5, 1.0, band_width//2 + 1, device=phase_diff.device)
                if band_width % 2 == 0:
                    # Even width
                    weights = torch.cat([weights, weights.flip(0)[1:]], dim=0)
                else:
                    # Odd width
                    weights = torch.cat([weights, weights.flip(0)[:-1]], dim=0)
                
                # Apply weighted average of individual phase difference and band average
                lock_strength = 0.7  # How strongly to lock phases (0=none, 1=complete lock)
                band_phase_diff = (1 - lock_strength * weights) * phase_diff[low_bin:high_bin] + \
                                 (lock_strength * weights) * avg_phase_diff
                
                # Update the aligned phase with the locked version
                aligned_phase[low_bin:high_bin, f] = aligned_phase[low_bin:high_bin, f-1] + band_phase_diff
        
        # If transient => override with original target phase
        if enable_transient_preservation and transient_mask is not None and transient_mask[f]:
            aligned_phase[:, f] = tgt_phase[:, f]
    
    aligned_stft = torch.polar(tgt_mag, aligned_phase)
    aligned_signal = torch.istft(aligned_stft, fft_size, hop_size, window=window, length=length)
    return aligned_signal
