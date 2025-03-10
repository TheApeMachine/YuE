import numpy as np
import torch
from audio_utils import save_audio

from audio_mixing import align_phases, enhance_stereo_width, multi_band_compression, apply_gain_staging

def post_process_generated_audio(audio, sr=44100, apply_enhancements=True, audio_processing_level="full"):
    """
    Apply audio mixing enhancements to generated audio
    
    Args:
        audio: Generated audio tensor (mono or stereo)
        sr: Sample rate
        apply_enhancements: Whether to apply enhancements
        audio_processing_level: Level of audio post-processing to apply ("minimal", "standard", "full")
        
    Returns:
        Enhanced audio
    """
    if not apply_enhancements:
        return audio
        
    # Skip processing if audio is invalid
    if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
        return audio
    
    # If audio is a numpy array, convert to tensor for processing
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # 1. Convert to stereo if mono
    if audio.dim() == 1 or (audio.dim() > 1 and audio.shape[0] == 1):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.repeat(2, 1)
    
    # For minimal processing, just return stereo conversion
    if audio_processing_level == "minimal":
        return audio
    
    # 2. Apply phase alignment between channels to improve stereo image
    if audio.shape[0] > 1:
        # Use the first channel as reference for phase alignment
        reference = audio[0].unsqueeze(0)
        target = audio[1].unsqueeze(0)
        # Align the second channel to the first
        aligned_channel = align_phases(reference, target)
        # Reconstruct stereo signal with aligned phases
        audio = torch.cat([reference, aligned_channel], dim=0)
    
    # Apply different levels of processing based on audio_processing_level
    if audio_processing_level == "standard":
        # Moderate stereo width enhancement
        audio = enhance_stereo_width(audio, width=1.15)
        
        # Apply simpler multi-band compression for balanced dynamics
        audio = multi_band_compression(
            audio,
            bands=[(0, 250), (250, 8000), (8000, 22050)],  # Fewer bands
            thresholds=[-22, -18, -16],
            ratios=[2.0, 1.8, 1.5],
            sr=sr
        )
    else:  # "full" processing
        # 3. Apply stereo width enhancement
        audio = enhance_stereo_width(audio, width=1.3)
        
        # 4. Apply multi-band compression for balanced dynamics
        audio = multi_band_compression(
            audio,
            bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)],
            thresholds=[-24, -18, -18, -16],
            ratios=[2.5, 2.0, 1.8, 1.5],
            sr=sr
        )
    
    # 5. Final gain staging (apply to all levels except minimal)
    audio = apply_gain_staging(audio, gain_db=-0.3)
    
    return audio

def process_and_save_audio(
    audio, 
    output_path, 
    codectool, 
    sr=44100, 
    apply_enhancements=True, 
    diffusion_postproc_model=None, 
    diffusion_steps=50, 
    diffusion_sampling_method='ddpm', 
    audio_processing_level="full"
):
    """
    Process and save generated audio
    
    Args:
        audio: Generated audio data
        output_path: Path to save the audio
        codectool: Codec tool for audio processing
        sr: Sample rate
        apply_enhancements: Whether to apply audio enhancements
        diffusion_postproc_model: Optional diffusion model for post-processing
        diffusion_steps: Number of steps for diffusion process
        diffusion_sampling_method: Sampling method for diffusion
        audio_processing_level: Level of audio post-processing to apply
        
    Returns:
        Path to processed audio
    """
    # Convert from tensor or numpy array to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Save the codec tokens or intermediate format if needed
    np.save(output_path, audio)
    
    # Decode to audio waveform
    audio = codectool.ids2npy(audio)
    
    # Apply diffusion post-processing if available
    if diffusion_postproc_model is not None:
        print("Applying diffusion post-processing...")
        audio = diffusion_postproc_model.denoise(
            audio, 
            steps=diffusion_steps,
            sampling_method=diffusion_sampling_method
        )
    
    # Apply audio enhancements
    if apply_enhancements:
        print(f"Applying audio enhancements (level: {audio_processing_level})...")
        audio = post_process_generated_audio(audio, sr=sr, audio_processing_level=audio_processing_level)
    
    # Save the audio
    wav_output_path = output_path.replace('.npy', '.wav')
    save_audio(audio, wav_output_path, sample_rate=sr)
    
    return output_path
