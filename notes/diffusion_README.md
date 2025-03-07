# YuE Diffusion Enhancements

This extension to YuE adds three different diffusion-based approaches for enhancing music generation quality:

1. **Hybrid Architecture**: Combines transformer models for musical structure with diffusion models for audio quality
2. **Post-Processing Enhancement**: Applies diffusion-based audio enhancement after standard generation
3. **Conditional Generation**: Uses diffusion models conditioned on codec tokens for better transitions

These enhancements are completely optional and can be enabled via command-line parameters.

## Requirements

In addition to the standard YuE requirements, you'll need a pre-trained diffusion model for audio. The code is designed to work with various diffusion models, but you'll need to provide the model weights.

## Usage

### Command-line Arguments

The following additional arguments have been added to YuE's main.py:

```
--use_diffusion                 Enable diffusion model enhancements (required for any diffusion method)
--diffusion_model_path PATH     Path to pre-trained diffusion model weights
--use_hybrid_architecture       Use hybrid transformer-diffusion architecture
--use_diffusion_postprocessing  Apply diffusion-based enhancement after generation
--use_conditional_diffusion     Use diffusion models conditioned on codec tokens
--diffusion_guidance_scale N    Guidance scale for classifier-free guidance (default: 3.0)
--diffusion_steps N             Number of diffusion steps (default: 50)
--diffusion_sampling_method M   Sampling method (ddpm, ddim, or plms) (default: ddpm)
```

### Example Commands

#### 1. Using Post-Processing Enhancement Only

This is the simplest approach - it keeps the existing YuE pipeline but enhances the audio quality at the end:

```bash
python YuE/inference/main.py \
  --genre_txt path/to/genre.txt \
  --use_diffusion \
  --diffusion_model_path path/to/diffusion_model.pt \
  --use_diffusion_postprocessing \
  --diffusion_steps 50
```

#### 2. Using All Three Enhancements

For maximum quality improvement, you can enable all three approaches:

```bash
python YuE/inference/main.py \
  --genre_txt path/to/genre.txt \
  --use_diffusion \
  --diffusion_model_path path/to/diffusion_model.pt \
  --use_hybrid_architecture \
  --use_diffusion_postprocessing \
  --use_conditional_diffusion \
  --diffusion_steps 100 \
  --diffusion_sampling_method ddim
```

### Demo Script

For convenience, a demo script is provided that makes it easy to try different diffusion enhancements:

```bash
python YuE/demo_diffusion.py \
  --genre_txt path/to/genre.txt \
  --diffusion_model_path path/to/diffusion_model.pt \
  --all_enhancements \
  --diffusion_steps 50
```

Or to try just one enhancement method:

```bash
python YuE/demo_diffusion.py \
  --genre_txt path/to/genre.txt \
  --diffusion_model_path path/to/diffusion_model.pt \
  --postproc_only
```

## Technical Details

### Diffusion Models Integration

The diffusion models are integrated at three different points in the YuE pipeline:

1. **Hybrid Architecture**: Integrates between the transformer token generation and audio synthesis
2. **Post-Processing**: Applied after the standard YuE generation pipeline
3. **Conditional Generation**: Integrates during the token-to-audio decoding process

### Implementation Notes

- All diffusion enhancements are implemented as optional modules that don't affect the original pipeline when not used
- The code includes placeholder implementations that can be replaced with actual diffusion models
- Performance will vary based on the quality and type of diffusion model used

## Contributing

This is an experimental extension to YuE. If you improve the diffusion models or integration, please consider contributing your changes! 