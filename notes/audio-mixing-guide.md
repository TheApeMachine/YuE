# YuE Audio Mixing Enhancements Guide

This document explains how to use the newly implemented audio mixing enhancements in YuE. These improvements address common issues in AI-generated music and bring the output closer to professional studio quality.

## Table of Contents

1. [Overview](#overview)
2. [Integrated Enhancements](#integrated-enhancements)
3. [Standalone Enhancement Tool](#standalone-enhancement-tool)
4. [Advanced Configuration](#advanced-configuration)
5. [Technical Details](#technical-details)

## Overview

The audio mixing enhancements implement various professional mixing techniques:

- **Phase Alignment**: Ensures proper phase relationships between audio components
- **Loudness Normalization**: Balances loudness levels for consistent output
- **Dynamic Processing**: Applies compression and dynamic control
- **Stereo Enhancement**: Improves spatial characteristics of the audio
- **Multiband Processing**: Processes different frequency ranges independently

These enhancements are available in two ways:

1. Integrated directly into the YuE inference pipeline
2. As a standalone tool for post-processing audio files

## Integrated Enhancements

The easiest way to use these enhancements is by enabling them in the main inference script.

### Basic Usage

Simply add the `--enhance-audio` flag when running the inference:

```bash
python inference/main.py --prompt "Your prompt here" --enhance-audio
```

This applies the default enhancements to all generated audio.

### Advanced Options

You can customize certain aspects of the enhancement process:

```bash
python inference/main.py --prompt "Your prompt here" \
    --enhance-audio \
    --stereo-width 1.5 \
    --apply-compression
```

Available options:

- `--enhance-audio`: Enable audio enhancements (required)
- `--stereo-width`: Control stereo width (1.0 = normal, >1.0 = wider)
- `--apply-compression`: Apply multiband compression

## Standalone Enhancement Tool

For more control over the enhancement process, or to process existing audio files, use the standalone `enhance_audio.py` script.

### Processing Existing Audio

```bash
# Enhance a single audio file
python enhance_audio.py --mode enhance --audio path/to/audio.wav --output path/to/output.wav

# Add compression and stereo widening
python enhance_audio.py --mode enhance --audio path/to/audio.wav --output path/to/output.wav \
    --compression --stereo-width 1.4
```

### Mixing Vocal and Instrumental Tracks

```bash
python enhance_audio.py --mode mix \
    --vocal path/to/vocal.wav \
    --instrumental path/to/instrumental.wav \
    --output path/to/mixed.wav \
    --vocal-gain 1.0 \
    --instrumental-gain 0.8 \
    --compression \
    --sidechain \
    --phase-align
```

### Phase Alignment

```bash
python enhance_audio.py --mode align \
    --audio path/to/audio.wav \
    --reference path/to/reference.wav \
    --output path/to/aligned.wav
```

### Multiband Processing

```bash
python enhance_audio.py --mode multiband \
    --audio path/to/audio.wav \
    --reference path/to/reference.wav \
    --output path/to/processed.wav
```

## Advanced Configuration

The default parameters are optimized for most use cases, but you can customize:

1. **Frequency Bands**: The default bands are `[(0, 250), (250, 2000), (2000, 8000), (8000, 22050)]`, dividing audio into low, low-mid, high-mid, and high frequencies.

2. **Compression Settings**: Different thresholds and ratios are applied to each band. You can modify these in `audio_mixing.py` for specialized applications.

3. **Phase Alignment Parameters**: FFT size and hop size can be adjusted for different time-frequency resolution tradeoffs.

## Technical Details

These enhancements implement techniques from the professional audio mixing field:

### Phase Alignment

The phase alignment uses the STFT (Short-Time Fourier Transform) to match the phase properties of one audio signal to another, preserving the magnitude characteristics of the original signal while improving coherence.

### Multiband Processing

Multiband processing applies different processing to different frequency ranges, allowing for targeted enhancement. This is particularly important for achieving a balanced mix where each frequency range needs different treatment.

### Dynamic Processing

The dynamic processing uses standard compression techniques based on envelope following with customizable attack and release times. Both standard compression and sidechain compression (for ducking) are implemented.

### Stereo Enhancement

The stereo enhancement uses mid-side processing to increase the perceived stereo width while maintaining mono compatibility.

For more technical details, please refer to the code comments in `inference/audio_mixing.py`.
