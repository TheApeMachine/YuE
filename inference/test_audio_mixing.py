import os
import torch
import torchaudio
import argparse
from YuE.inference.audio_mixing import enhanced_audio_mix, process_files_with_enhancements

def test_audio_mixing(vocal_path, instrumental_path, output_path):
    """
    Test the enhanced audio mixing features with custom settings.
    
    Args:
        vocal_path: Path to vocal file
        instrumental_path: Path to instrumental file
        output_path: Path to save mixed output
    """
    print(f"Processing {os.path.basename(vocal_path)} + {os.path.basename(instrumental_path)}")
    
    # Load audio files
    vocal, sr_v = torchaudio.load(vocal_path)
    instrumental, sr_i = torchaudio.load(instrumental_path)
    
    # Ensure same sample rate
    if sr_v != sr_i:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_i, new_freq=sr_v)
        instrumental = resampler(instrumental)
        sr = sr_v
    else:
        sr = sr_v
        
    print(f"Sample rate: {sr}Hz")
    print(f"Vocal shape: {vocal.shape}, Instrumental shape: {instrumental.shape}")
    
    # Create custom mixing parameters to showcase the new features
    custom_params = {
        'vocal_enhancement': {
            'enabled': True,
            'level': 0.8  # Slightly stronger enhancement
        },
        'vocal_space_carving': {
            'enabled': True,
            'level': 0.7  # More aggressive carving
        },
        'instrumental_saturation': {
            'enabled': True,
            'amount': 0.4,
            'type': 'tube'
        },
        'exciter': {
            'enabled': True,
            'amount': 0.5,
            'frequency': 3500  # Higher frequency focus
        },
        'spectral_balance': {
            'enabled': True,
            'strength': 0.8
        },
        'reverb': {
            'enabled': True,
            'mix': 0.15,
            'room_size': 0.6,
            'damping': 0.4,
            'pre_delay_ms': 15
        },
        'stereo_width': {
            'enabled': True,
            'width': 1.3  # Wider stereo image
        }
    }
    
    # Process with our new features
    print("Applying enhanced mix with new features...")
    mixed = enhanced_audio_mix(vocal, instrumental, custom_params, sr)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, mixed.cpu(), sr)
    print(f"Enhanced mix saved to: {output_path}")
    
    # Also create a basic mix for comparison
    basic_mix = (vocal + instrumental) * 0.5
    basic_path = output_path.replace('.wav', '_basic.wav')
    torchaudio.save(basic_path, basic_mix.cpu(), sr)
    print(f"Basic mix saved to: {basic_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YuE audio mixing improvements")
    parser.add_argument("--vocal", type=str, required=True, help="Path to vocal file")
    parser.add_argument("--instrumental", type=str, required=True, help="Path to instrumental file")
    parser.add_argument("--output", type=str, default="output/enhanced_mix.wav", help="Output path")
    
    args = parser.parse_args()
    
    test_audio_mixing(args.vocal, args.instrumental, args.output) 