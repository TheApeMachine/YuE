# Implementing Stereo Support in YuE Music Generation System

This guide explains how to modify the YuE music generation system to fully support stereo audio throughout the pipeline. The current implementation converts all audio to mono, but with these changes, you can preserve stereo information for richer, more immersive music generation.

## Why Stereo Matters

Stereo audio provides several benefits for music generation:

- Spatial positioning of instruments and vocals
- Increased sense of depth and immersion
- Better separation between elements
- More professional and polished sound

## Implementation Overview

To support stereo in YuE, we need to modify several components:

1. **Audio Loading**: Preserve stereo channels when loading reference audio
2. **Encoding**: Handle left and right channels separately during encoding
3. **Token Representation**: Store stereo information in the token space
4. **Generation**: Have the model generate tokens for both channels
5. **Decoding**: Convert stereo tokens back to waveforms correctly
6. **Mixing**: Properly mix stereo vocal and instrumental tracks
7. **Post-processing**: Ensure stereo is preserved in all post-processing steps

## Step-by-Step Implementation Guide

### 1. Modify Audio Loading Functions

Replace the current mono conversion:

```python
# Original code (converts to mono)
def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio
```

With stereo-preserving code:

```python
def load_audio_stereo(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)

    # Keep stereo if it exists, otherwise duplicate mono to create stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)  # Duplicate mono to stereo

    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)

    return audio
```

### 2. Extend the CodecManipulator Class

Create a `StereoCodecManipulator` class that extends the existing `CodecManipulator`:

```python
class StereoCodecManipulator(CodecManipulator):
    """Extension of CodecManipulator to handle stereo audio tokens"""

    def process_stereo(self, left_codes, right_codes):
        """Process and interleave stereo channel codes"""
        left_tokens = self.npy2ids(left_codes[0])
        right_tokens = self.npy2ids(right_codes[0])

        # Use existing special tokens as channel markers
        special_tokens = self.mm_v0_2_cfg["special_tokens"]
        left_marker = special_tokens["<s_local>"]
        right_marker = special_tokens["<s_global>"]

        # Format as: [LEFT_MARKER] [left_tokens] [RIGHT_MARKER] [right_tokens]
        stereo_tokens = [left_marker] + left_tokens + [right_marker] + right_tokens

        return stereo_tokens

    def deprocess_stereo(self, stereo_tokens):
        """Split stereo tokens back into separate channels"""
        special_tokens = self.mm_v0_2_cfg["special_tokens"]
        left_marker = special_tokens["<s_local>"]
        right_marker = special_tokens["<s_global>"]

        # Find marker positions
        left_start = stereo_tokens.index(left_marker) + 1
        right_start = stereo_tokens.index(right_marker) + 1

        # Extract tokens for each channel
        if right_marker in stereo_tokens[left_start:]:
            right_idx = stereo_tokens[left_start:].index(right_marker) + left_start
            left_tokens = stereo_tokens[left_start:right_idx]
        else:
            left_tokens = stereo_tokens[left_start:]

        right_tokens = stereo_tokens[right_start:]

        # Convert back to numpy arrays
        left_data = self.ids2npy(left_tokens)
        right_data = self.ids2npy(right_tokens)

        return left_data, right_data
```

### 3. Modify the Encoding Function

Change the encoding function to handle stereo:

```python
def encode_audio_stereo(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)

    # Split stereo channels
    left_channel = audio_prompt[:, 0:1, :]
    right_channel = audio_prompt[:, 1:2, :]

    with torch.no_grad():
        # Encode each channel separately
        left_codes = codec_model.encode(left_channel.to(device), target_bw=target_bw)
        right_codes = codec_model.encode(right_channel.to(device), target_bw=target_bw)

    left_codes = left_codes.transpose(0, 1).cpu().numpy().astype(np.int16)
    right_codes = right_codes.transpose(0, 1).cpu().numpy().astype(np.int16)

    return left_codes, right_codes
```

### 4. Update the Decoding Function

Add a stereo-aware decoding function:

```python
def decode_stereo_audio(codec_model, left_codes, right_codes, device):
    with torch.no_grad():
        # Decode each channel
        left_waveform = codec_model.decode(
            torch.as_tensor(left_codes.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )

        right_waveform = codec_model.decode(
            torch.as_tensor(right_codes.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )

    # Combine channels
    left_waveform = left_waveform.cpu().squeeze(0)
    right_waveform = right_waveform.cpu().squeeze(0)

    # Stack to create stereo
    stereo_waveform = torch.stack([left_waveform, right_waveform], dim=0)

    return stereo_waveform
```

### 5. Update the Audio Saving Function

Ensure stereo is preserved when saving:

```python
def save_audio_stereo(wav, path, sample_rate, rescale=False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)

    # Ensure stereo format (2 channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
```

### 6. Modify Prompt Processing

Update how audio prompts are processed:

```python
# In the main generation loop where audio prompts are processed:
if args.use_audio_prompt:
    audio_prompt = load_audio_stereo(args.audio_prompt_path)
    left_codes, right_codes = encode_audio_stereo(codec_model, audio_prompt, device)

    # Create stereo-aware codec manipulator
    stereo_codectool = StereoCodecManipulator("xcodec", 0, 1)
    audio_prompt_codec = stereo_codectool.process_stereo(left_codes, right_codes)

    # Trim to desired segment
    start_idx = int(args.prompt_start_time * 50 * 2)  # Double for stereo
    end_idx = int(args.prompt_end_time * 50 * 2)
    audio_prompt_codec = audio_prompt_codec[start_idx:end_idx]

    # Finish processing prompt as before
    audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
    # ...continue with the rest of prompt processing
```

### 7. Update the Stage 2 Processing

Modify Stage 2 processing to handle stereo:

```python
def stage2_generate_stereo(model, prompt_left, prompt_right, batch_size=16):
    # Process left and right channels separately
    left_output = stage2_generate(model, prompt_left, batch_size)
    right_output = stage2_generate(model, prompt_right, batch_size)

    # Return both channels
    return left_output, right_output
```

### 8. Update Post-Processing

Ensure the post-processing preserves stereo:

```python
# Update the post-processing function to handle stereo
def replace_low_freq_with_energy_matched_stereo(a_file, b_file, c_file, cutoff_freq=5500.0):
    # Load audio
    a, sr_a = torchaudio.load(a_file)  # 16kHz
    b, sr_b = torchaudio.load(b_file)  # 44kHz

    # Process each channel separately
    output_channels = []
    for ch in range(a.shape[0]):
        # Apply existing processing to each channel
        # (Your existing frequency-domain processing logic here,
        #  but applied to each channel separately)
        processed_channel = replace_low_freq_with_energy_matched_single(
            a[ch], b[ch], cutoff_freq, sr_a, sr_b
        )
        output_channels.append(processed_channel)

    # Stack channels back together
    output = torch.stack(output_channels)

    # Save the stereo result
    torchaudio.save(c_file, output, sr_b)
```

## Testing Your Stereo Implementation

1. **Verify Channel Separation**: Generate output with distinct left/right channels and confirm the separation is maintained
2. **Test with Stereo References**: Use stereo audio as prompts and verify the output respects the spatial characteristics
3. **Compare with Mono**: Generate both stereo and mono versions to confirm the quality improvement

## Advanced Considerations

1. **Pan Automation**: Consider adding controlled panning between channels for different instruments
2. **Stereo Width Control**: Implement a parameter to control how wide the stereo field should be
3. **Mid-Side Processing**: Add mid-side processing capabilities for more nuanced stereo control
4. **Multi-Channel Support**: Extend beyond stereo to surround sound formats for even more immersive music
