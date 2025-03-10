#!/usr/bin/env python3
"""
YuE Music Generation - Audio Enhancement Script

This script applies advanced audio mixing techniques to improve the quality
of generated music.
"""

import os
import torch
import argparse
import torchaudio
from inference.audio_mixing import (
    process_files_with_enhancements,
    align_phases,
    enhance_stereo_width,
    apply_compression,
    multi_band_compression
)
from inference.post_process_audio import multiband_enhanced_stereo

def parse_args():
    parser = argparse.ArgumentParser(description="Enhance YuE audio quality with advanced mixing techniques")
    parser.add_argument(
        "--vocal", type=str, required=False, help="Path to vocal audio file"
    )
    parser.add_argument(
        "--instrumental", type=str, required=False, help="Path to instrumental audio file"
    )
    parser.add_argument(
        "--audio", type=str, required=False, help="Path to audio file to enhance (without mixing)"
    )
    parser.add_argument(
        "--reference", type=str, required=False, help="Path to reference audio file (for phase alignment)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output audio file"
    )
    parser.add_argument(
        "--mode", type=str, default="mix", choices=["mix", "enhance", "align", "multiband"], 
        help="Processing mode: mix (vocals+instrumental), enhance (single file), align (with reference), multiband"
    )
    parser.add_argument(
        "--vocal-gain", type=float, default=1.0, help="Gain for vocal track"
    )
    parser.add_argument(
        "--instrumental-gain", type=float, default=0.8, help="Gain for instrumental track"
    )
    parser.add_argument(
        "--target-lufs", type=float, default=-16.0, help="Target LUFS loudness"
    )
    parser.add_argument(
        "--compression", action="store_true", help="Apply compression"
    )
    parser.add_argument(
        "--multiband", action="store_true", help="Use multiband compression"
    )
    parser.add_argument(
        "--sidechain", action="store_true", help="Apply sidechain compression (instrumental ducking under vocals)"
    )
    parser.add_argument(
        "--phase-align", action="store_true", help="Align phases between tracks"
    )
    parser.add_argument(
        "--stereo-width", type=float, default=1.0, help="Stereo width enhancement (1.0 = normal, 1.5 = wider)"
    )
    parser.add_argument(
        "--pan-position", type=float, default=0.0, help="Pan position for vocals (-1.0 = left, 0.0 = center, 1.0 = right)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    if args.mode == "mix" and args.vocal and args.instrumental:
        # Mixing mode - combine vocals and instrumentals with enhancements
        mix_params = {
            'vocal_gain': args.vocal_gain,
            'instrumental_gain': args.instrumental_gain,
            'target_lufs': args.target_lufs,
            'vocal_compression': {
                'threshold': -20.0,
                'ratio': 2.0,
                'attack': 0.005,
                'release': 0.05
            } if args.compression else None,
            'sidechain': {
                'enabled': args.sidechain,
                'threshold': -24.0,
                'ratio': 2.5
            },
            'stereo_width': args.stereo_width,
            'pan_position': args.pan_position,
            'phase_align': args.phase_align
        }
        
        print("Mixing audio with enhancements:")
        print(f"  - Vocal: {args.vocal}")
        print(f"  - Instrumental: {args.instrumental}")
        print(f"  - Output: {args.output}")
        print(f"  - Vocal gain: {args.vocal_gain}")
        print(f"  - Instrumental gain: {args.instrumental_gain}")
        print(f"  - Target LUFS: {args.target_lufs}")
        print(f"  - Compression: {'Yes' if args.compression else 'No'}")
        print(f"  - Sidechain: {'Yes' if args.sidechain else 'No'}")
        print(f"  - Phase alignment: {'Yes' if args.phase_align else 'No'}")
        print(f"  - Stereo width: {args.stereo_width}")
        print(f"  - Pan position: {args.pan_position}")
        
        process_files_with_enhancements(args.vocal, args.instrumental, args.output, mix_params)
        print(f"Enhanced mix saved to {args.output}")
        
    elif args.mode == "enhance" and args.audio:
        # Enhancement mode - apply processing to a single audio file
        audio, sr = torchaudio.load(args.audio)
        
        print("Enhancing audio:")
        print(f"  - Input: {args.audio}")
        print(f"  - Output: {args.output}")
        print(f"  - Target LUFS: {args.target_lufs}")
        print(f"  - Compression: {'Multiband' if args.multiband else 'Standard' if args.compression else 'No'}")
        print(f"  - Stereo width: {args.stereo_width}")
        
        # Apply enhancements
        # 1. Make stereo if mono
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
            
        # 2. Apply compression if requested
        if args.compression:
            if args.multiband:
                audio = multi_band_compression(
                    audio,
                    bands=[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)],
                    thresholds=[-24, -18, -18, -16],
                    ratios=[2.5, 2.0, 1.8, 1.5],
                    sr=sr
                )
            else:
                audio = apply_compression(audio, threshold=-18.0, ratio=2.0, sr=sr)
        
        # 3. Enhance stereo width if requested
        if abs(args.stereo_width - 1.0) > 1e-6:  # Use small epsilon for float comparison
            audio = enhance_stereo_width(audio, width=args.stereo_width)
        
        # 4. Apply gain staging to reach target LUFS
        # We'll approximate this with peak normalization for simplicity
        peak_target = 10 ** (args.target_lufs / 20.0)
        current_peak = audio.abs().max()
        audio = audio * (peak_target / current_peak)
        
        # Save the enhanced audio
        torchaudio.save(args.output, audio, sr)
        print(f"Enhanced audio saved to {args.output}")
        
    elif args.mode == "align" and args.audio and args.reference:
        # Alignment mode - align audio to reference
        audio, sr_audio = torchaudio.load(args.audio)
        reference, sr_ref = torchaudio.load(args.reference)
        
        # Ensure matching sample rates
        if sr_audio != sr_ref:
            print(f"Resampling reference from {sr_ref}Hz to {sr_audio}Hz")
            resampler = torchaudio.transforms.Resample(sr_ref, sr_audio)
            reference = resampler(reference)
        
        print("Aligning audio to reference:")
        print(f"  - Audio: {args.audio}")
        print(f"  - Reference: {args.reference}")
        print(f"  - Output: {args.output}")
        
        # Align phases
        aligned = align_phases(reference, audio)
        
        # Save aligned audio
        torchaudio.save(args.output, aligned, sr_audio)
        print(f"Phase-aligned audio saved to {args.output}")
        
    elif args.mode == "multiband" and args.audio and args.reference:
        # Multiband alignment mode
        print("Applying multiband processing:")
        print(f"  - Audio: {args.audio}")
        print(f"  - Reference: {args.reference}")
        print(f"  - Output: {args.output}")
        
        # Use the multiband processing function
        multiband_enhanced_stereo(args.audio, args.reference, args.output)
        print(f"Multiband enhanced audio saved to {args.output}")
        
    else:
        print("Error: Invalid combination of arguments for the selected mode.")
        print("For 'mix' mode, provide --vocal and --instrumental")
        print("For 'enhance' mode, provide --audio")
        print("For 'align' mode, provide --audio and --reference")
        print("For 'multiband' mode, provide --audio and --reference")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 