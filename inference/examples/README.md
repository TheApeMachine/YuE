# YuE Examples

This directory contains example scripts that demonstrate how to use YuE with various configurations and for specific use cases.

## Available Examples

### 1. Running on Older GPUs (Quadro, etc.)

The `run_on_quadro_example.py` script demonstrates how to run YuE on older GPUs like the Quadro M6000 that don't support Flash Attention.

```bash
python run_on_quadro_example.py
```

Key features:

-   Disables Flash Attention
-   Enables PyTorch compile optimization
-   Uses 8-bit quantization
-   Configures model distribution across multiple GPUs

### 2. Audio Mixing Testing

The `test_audio_mixing_example.py` script allows you to test the audio mixing functionality without running the full model.

```bash
python test_audio_mixing_example.py --vocal_path /path/to/vocal.wav --instrumental_path /path/to/instrumental.wav --output_path mixed_output.wav
```

Options:

-   `--processing_level`: Choose between "minimal", "standard", or "full" processing (default: "full")

### 3. Safe Mode Example

The `safe_run_example.py` script demonstrates how to run YuE with conservative settings for resource-constrained environments.

```bash
python safe_run_example.py
```

Key features:

-   Uses fewer tokens
-   Generates a single segment
-   Uses 4-bit quantization
-   Applies minimal audio processing

## Using the Main Entry Point Directly

Instead of using these example scripts, you can also use the main entry point directly with the appropriate flags:

```bash
# For Quadro GPUs:
python ../infer.py --genre_txt ../../prompt_egs/genre.txt --lyrics_txt ../../prompt_egs/lyrics.txt --disable_flash_attention --enable_torch_compile --quantization 8bit

# For audio mixing test:
python ../infer.py --test_audio_mixing --vocal_path /path/to/vocal.wav --instrumental_path /path/to/instrumental.wav --output_path mixed_output.wav

# For safe mode:
python ../infer.py --genre_txt ../../prompt_egs/genre.txt --lyrics_txt ../../prompt_egs/lyrics.txt --safe_mode
```

## Hardware-Specific Configuration

For optimal performance on your specific hardware, use the auto-configuration feature:

```bash
python ../infer.py --genre_txt ../../prompt_egs/genre.txt --lyrics_txt ../../prompt_egs/lyrics.txt --auto_config
```

This will automatically detect your hardware capabilities and apply the recommended settings.

## Multi-GPU Configuration

For systems with multiple GPUs, you can distribute the workload:

```bash
python ../infer.py --genre_txt ../../prompt_egs/genre.txt --lyrics_txt ../../prompt_egs/lyrics.txt --model_split_strategy model_type --transformer_device cuda:0,1 --diffusion_device cuda:0 --codec_device cuda:1
```

This distributes the transformer model across both GPUs while placing the diffusion model on the first GPU and the codec model on the second GPU.
