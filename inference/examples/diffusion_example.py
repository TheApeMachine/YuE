import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav

# Import our diffusion models
from diffusion_models import (
    HybridArchitectureDiffusion,
    PostProcessingDiffusion,
    ConditionalDiffusion
)

def plot_audio_waveform(audio, title, filename=None):
    """Plot an audio waveform for visualization"""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    if audio.ndim > 1:
        # Plot stereo audio
        plt.plot(audio[0], alpha=0.7, label='Left')
        plt.plot(audio[1], alpha=0.7, label='Right')
        plt.legend()
    else:
        # Plot mono audio
        plt.plot(audio)
    
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def save_audio(audio, sample_rate=44100, filename='output.wav'):
    """Save audio tensor as WAV file"""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    
    # Ensure audio is in [-1, 1] range
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    # Convert to int16 format for WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Transpose if needed (scipy expects [samples, channels])
    if audio_int16.ndim > 1 and audio_int16.shape[0] < audio_int16.shape[1]:
        audio_int16 = audio_int16.T
    
    # Save file
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    write_wav(filename, sample_rate, audio_int16)
    print(f"Saved audio to {filename}")

def example_hybrid_diffusion():
    """Demonstrate the hybrid architecture diffusion model"""
    print("\n=== Testing Hybrid Architecture Diffusion ===")
    
    # Initialize model (will download from HuggingFace if needed)
    model = HybridArchitectureDiffusion(
        model_path='',  # No local path - will use HuggingFace
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy structure tokens (in real usage, these would come from a transformer model)
    structure_tokens = torch.randn(1, 16, 512)  # [batch_size, sequence_length, embedding_dim]
    
    # Generate audio
    print("Generating audio from structure tokens...")
    audio = model.generate_from_tokens(
        structure_tokens,
        steps=20,  # Fewer steps for faster generation during testing
        sampling_method='ddim',  # DDIM is faster than DDPM
        guidance_scale=3.0
    )
    
    # Save and visualize the result
    os.makedirs('outputs/diffusion', exist_ok=True)
    save_audio(audio[0], filename='outputs/diffusion/hybrid_output.wav')
    plot_audio_waveform(audio[0], "Hybrid Diffusion Output", 
                        filename='outputs/diffusion/hybrid_waveform.png')
    
    return audio

def example_post_processing():
    """Demonstrate the post-processing diffusion model"""
    print("\n=== Testing Post-Processing Diffusion ===")
    
    # Initialize model (will download from HuggingFace if needed)
    model = PostProcessingDiffusion(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create some noisy/low-quality audio for enhancement
    audio_length = 44100 * 2  # 2 seconds
    noisy_audio = torch.randn(2, audio_length) * 0.1  # Low volume noise
    
    # Add a simple sine wave to simulate a basic audio signal
    t = torch.linspace(0, 4 * np.pi, audio_length)
    sine_wave = torch.sin(440 * t).unsqueeze(0).repeat(2, 1)  # 440 Hz sine, stereo
    noisy_audio += sine_wave * 0.5
    
    # Enhance audio
    print("Enhancing audio quality...")
    enhanced_audio = model.enhance_audio(
        noisy_audio,
        steps=20,  # Fewer steps for faster processing during testing
        sampling_method='ddim'
    )
    
    # Save and visualize the results
    os.makedirs('outputs/diffusion', exist_ok=True)
    
    # Original noisy audio
    save_audio(noisy_audio, filename='outputs/diffusion/noisy_audio.wav')
    plot_audio_waveform(noisy_audio, "Noisy Audio Input", 
                        filename='outputs/diffusion/noisy_waveform.png')
    
    # Enhanced audio
    save_audio(enhanced_audio, filename='outputs/diffusion/enhanced_audio.wav')
    plot_audio_waveform(enhanced_audio, "Enhanced Audio Output", 
                        filename='outputs/diffusion/enhanced_waveform.png')
    
    return enhanced_audio

def example_conditional_diffusion():
    """Demonstrate the conditional diffusion model"""
    print("\n=== Testing Conditional Diffusion ===")
    
    # Initialize model (will download from HuggingFace if needed)
    model = ConditionalDiffusion(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy codec tokens (in real usage, these would be actual codec tokens)
    # Adjust dimensions based on the model's expected input
    codec_tokens = torch.randn(1, 16, 1024)  # [batch_size, sequence_length, embedding_dim]
    
    # Generate audio with natural transitions
    print("Generating audio with natural transitions...")
    audio = model.generate_conditioned(
        codec_tokens,
        steps=20,  # Fewer steps for faster generation during testing
        sampling_method='ddim',
        guidance_scale=3.0
    )
    
    # Save and visualize the result
    os.makedirs('outputs/diffusion', exist_ok=True)
    save_audio(audio[0], filename='outputs/diffusion/conditional_output.wav')
    plot_audio_waveform(audio[0], "Conditional Diffusion Output", 
                        filename='outputs/diffusion/conditional_waveform.png')
    
    return audio

def main():
    print("YuE Diffusion Models Example")
    print("----------------------------")
    print("This script demonstrates the three diffusion models in YuE:")
    print("1. Hybrid Architecture Diffusion")
    print("2. Post-Processing Diffusion")
    print("3. Conditional Diffusion")
    print("\nModels will be downloaded automatically from HuggingFace if needed.\n")
    
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    hybrid_audio = example_hybrid_diffusion()
    enhanced_audio = example_post_processing()
    conditional_audio = example_conditional_diffusion()
    
    print("\nAll examples completed successfully!")
    print("Check the outputs/diffusion directory for the generated audio files and waveform plots.")

if __name__ == "__main__":
    main() 