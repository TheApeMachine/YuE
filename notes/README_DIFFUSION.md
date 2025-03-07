# YuE Diffusion Models

This document explains how to use the diffusion models in the YuE audio generation system.

## Overview

YuE includes three types of diffusion models to enhance audio generation quality:

1. **Hybrid Architecture Diffusion** - Enhances structure tokens from transformer models to generate high-quality audio
2. **Post-Processing Diffusion** - Improves the quality of existing audio through diffusion-based enhancement
3. **Conditional Diffusion** - Generates audio with natural transitions using codec token conditioning

## Automatic Model Loading

The diffusion models will automatically download pre-trained weights from HuggingFace when you first use them. No separate download script is needed. Simply initialize the models with either:

- A local model path
- A HuggingFace model ID 
- Or both (will try local path first, then download if needed)

## Usage Examples

### Hybrid Architecture Diffusion

```python
from YuE.inference.diffusion_models import HybridArchitectureDiffusion

# Initialize with automatic download from HuggingFace
model = HybridArchitectureDiffusion(
    model_path='',  # No local path - will use HuggingFace
    device='cuda',  # Use GPU if available
    hf_model_id='facebook/musicgen-small'  # Default HuggingFace model
)

# Generate high-quality audio from structure tokens
structure_tokens = ...  # From transformer model
audio = model.generate_from_tokens(
    structure_tokens,
    steps=50,
    sampling_method='ddim',  # Options: 'ddpm', 'ddim', 'plms'
    guidance_scale=3.0
)
```

### Post-Processing Diffusion

```python
from YuE.inference.diffusion_models import PostProcessingDiffusion

# Initialize with automatic download
model = PostProcessingDiffusion(
    device='cuda',
    hf_model_id='facebook/audiocraft-base'  # Default HuggingFace model
)

# Enhance existing audio
audio = ...  # Your audio tensor or numpy array
enhanced_audio = model.enhance_audio(
    audio,
    steps=50,
    sampling_method='ddim'
)
```

### Conditional Diffusion

```python
from YuE.inference.diffusion_models import ConditionalDiffusion

# Initialize with local model path
model = ConditionalDiffusion(
    model_path='YuE/models/diffusion/conditional_diffusion.pt',
    device='cuda'
)

# Generate audio with natural transitions
codec_tokens = ...  # Your codec tokens
audio = model.generate_conditioned(
    codec_tokens,
    steps=50,
    sampling_method='ddpm',
    guidance_scale=3.0
)
```

## Sampling Methods

The diffusion models support three sampling methods:

- **DDPM** - Original diffusion process, high quality but slower
- **DDIM** - Deterministic sampling, faster with similar quality
- **PLMS** - Pseudo Linear Multi-Step, improved stability

## Model Compatibility

The models expect pre-trained weights with a specific format. If you're using custom pre-trained weights, they should include:

- `model_state_dict`: The actual model parameters
- `diffusion_steps`: Number of steps in the diffusion process
- `beta_start` and `beta_end`: Noise schedule parameters

When downloading from HuggingFace, the models will attempt to adapt different formats to work with YuE's implementation.

## Running the Example

To test all three diffusion models with automatic downloading:

```bash
python -m YuE.inference.diffusion_example
```

This will demonstrate all three models, generate sample audio files, and create waveform visualizations in the `outputs/diffusion` directory.

## Requirements

- PyTorch 1.7+
- tqdm
- huggingface_hub (for automatic downloading)
- matplotlib (for visualization)
- scipy (for audio saving)

You can install the dependencies with:

```bash
pip install torch tqdm huggingface_hub matplotlib scipy
``` 