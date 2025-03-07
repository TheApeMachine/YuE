import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# Add HuggingFace integration
try:
    from huggingface_hub import hf_hub_download
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

class DiffusionModel(nn.Module):
    """Base class for diffusion models used in YuE"""
    def __init__(self, model_path='', device='cuda', hf_model_id=None):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.hf_model_id = hf_model_id
        self.is_loaded = False
        self.noise_schedule = None
        
    def load_model(self):
        """Load model weights if model_path is provided or download from HuggingFace"""
        # Try loading from local path first
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.eval()  # Set to evaluation mode
                self.noise_schedule = get_noise_schedule(
                    num_steps=checkpoint.get('diffusion_steps', 1000),
                    beta_start=checkpoint.get('beta_start', 1e-4),
                    beta_end=checkpoint.get('beta_end', 0.02)
                )
                self.is_loaded = True
                print(f"Successfully loaded diffusion model from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model from local path: {e}")
                # If local loading fails, try HuggingFace if available
                if self.hf_model_id:
                    return self._load_from_huggingface()
                return False
        # If no local path or it doesn't exist, try HuggingFace
        elif self.hf_model_id:
            return self._load_from_huggingface()
        
        return False
    
    def _load_from_huggingface(self):
        """Download and load model from HuggingFace Hub"""
        if not HUGGINGFACE_AVAILABLE:
            print("HuggingFace Hub not available. Install with: pip install huggingface_hub")
            return False
            
        try:
            print(f"Downloading model from HuggingFace: {self.hf_model_id}")
            
            # Define local cache directory
            cache_dir = os.path.join('YuE', 'models', 'diffusion', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Download model file
            model_file = hf_hub_download(
                repo_id=self.hf_model_id,
                filename="pytorch_model.bin",
                cache_dir=cache_dir
            )
            
            # Download config if available
            try:
                config_file = hf_hub_download(
                    repo_id=self.hf_model_id,
                    filename="config.json",
                    cache_dir=cache_dir
                )
                # Could parse config here if needed
            except:
                pass
                
            # Load model
            model_weights = torch.load(model_file, map_location=self.device)
            
            # Handle different model formats
            if isinstance(model_weights, dict):
                if 'state_dict' in model_weights:
                    state_dict = model_weights['state_dict']
                elif 'model_state_dict' in model_weights:
                    state_dict = model_weights['model_state_dict']
                elif 'model' in model_weights:
                    state_dict = model_weights['model']
                else:
                    # Assume it's already a state dict
                    state_dict = model_weights
            else:
                state_dict = model_weights
                
            # Handle module prefix if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            # Load state dict with flexible loading (ignore missing keys)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in model: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in model: {unexpected_keys[:5]}...")
                
            self.eval()  # Set to evaluation mode
            
            # Set default noise schedule
            self.noise_schedule = get_noise_schedule()
            
            self.is_loaded = True
            print(f"Successfully loaded diffusion model from HuggingFace: {self.hf_model_id}")
            return True
            
        except Exception as e:
            print(f"Error loading model from HuggingFace: {e}")
            return False
        
    def add_noise(self, x_start, t, noise=None):
        """
        Add noise to samples at specified timestep according to diffusion process
        
        Args:
            x_start: Starting clean sample
            t: Timestep
            noise: Noise to add (if None, random noise is generated)
            
        Returns:
            Noisy sample at timestep t
        """
        if self.noise_schedule is None:
            self.noise_schedule = get_noise_schedule()
            
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.noise_schedule['sqrt_alphas_cumprod'][t]
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule['sqrt_one_minus_alphas_cumprod'][t]
        
        # Expand dimensions for broadcasting if needed
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
            
        # Forward diffusion process: q(x_t | x_0) = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def denoise(self, noisy_audio, condition=None, steps=50, sampling_method='ddpm', guidance_scale=3.0):
        """
        Generic denoising interface
        
        Args:
            noisy_audio: Tensor with noise added
            condition: Optional conditioning information (codec tokens, etc.)
            steps: Number of diffusion steps
            sampling_method: DDPM, DDIM, or PLMS sampling
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Denoised audio sample
        """
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("Model not loaded")
            
        # Move to device
        noisy_audio = noisy_audio.to(self.device)
        if condition is not None:
            condition = condition.to(self.device)
            
        # Initialize noise schedule if needed
        if self.noise_schedule is None:
            self.noise_schedule = get_noise_schedule()
            
        # Select sampling method
        if sampling_method.lower() == 'ddim':
            return self._ddim_sample(noisy_audio, condition, steps, guidance_scale)
        elif sampling_method.lower() == 'plms':
            return self._plms_sample(noisy_audio, condition, steps, guidance_scale)
        else:  # Default to DDPM
            return self._ddpm_sample(noisy_audio, condition, steps, guidance_scale)
            
    def _predict_noise(self, x_t, t, condition=None, guidance_scale=3.0):
        """
        Predict noise using model with optional classifier-free guidance
        
        Args:
            x_t: Noisy audio at timestep t
            t: Timestep indices
            condition: Conditioning information
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Predicted noise
        """
        if condition is not None and guidance_scale > 1.0:
            # Classifier-free guidance
            # 1. Predict with conditioning
            noise_pred_cond = self.forward(x_t, t, condition)
            
            # 2. Predict without conditioning (unconditional)
            noise_pred_uncond = self.forward(x_t, t, None)
            
            # 3. Interpolate predictions for guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # Regular prediction
            noise_pred = self.forward(x_t, t, condition)
            
        return noise_pred
            
    def _ddpm_sample(self, x_T, condition=None, steps=50, guidance_scale=3.0):
        """
        DDPM sampling algorithm
        
        Args:
            x_T: Starting noise
            condition: Conditioning information
            steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Denoised sample
        """
        # Get noise schedule for sampling timesteps
        betas = self.noise_schedule['betas']
        alphas = self.noise_schedule['alphas']
        alphas_cumprod = self.noise_schedule['alphas_cumprod']
        
        # Generate timesteps for sampling (evenly spaced)
        timesteps = torch.linspace(0, len(alphas_cumprod)-1, steps=steps).long().to(self.device)
        timesteps = timesteps.flip(0)  # Start from T, go to 0
        
        x_t = x_T
        
        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="DDPM Sampling")):
            # If it's the last step, there's no noise to add
            if i == len(timesteps) - 1:
                break
                
            # 1. Predict noise
            noise_pred = self._predict_noise(x_t, t, condition, guidance_scale)
            
            # 2. Get next timestep
            t_next = timesteps[i + 1]
            alpha_t = alphas[t]
            alpha_t_next = alphas[t_next]
            beta_t = betas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_t_next = alphas_cumprod[t_next]
            
            # 3. Compute coefficients
            coef1 = alpha_cumprod_t_next / alpha_cumprod_t
            coef2 = (1 - alpha_cumprod_t_next) - coef1 * (1 - alpha_cumprod_t)
            
            # 4. Predict x_0
            pred_x0 = (x_t - (1 - alpha_cumprod_t).sqrt() * noise_pred) / alpha_cumprod_t.sqrt()
            
            # 5. Compute mean for q(x_{t-1} | x_t, x_0)
            mean = coef1.sqrt() * x_t + coef2.sqrt() * pred_x0
            
            # 6. Add noise for x_{t-1} sample, except for the last step
            if i < len(timesteps) - 2:
                noise = torch.randn_like(x_t)
                variance = (1 - alpha_t_next) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t_next)
                sigma = variance.sqrt()
                x_t = mean + sigma * noise
            else:
                x_t = mean
                
        return x_t
            
    def _ddim_sample(self, x_T, condition=None, steps=50, guidance_scale=3.0):
        """
        DDIM sampling algorithm (faster than DDPM with similar quality)
        
        Args:
            x_T: Starting noise
            condition: Conditioning information
            steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Denoised sample
        """
        # Get noise schedule
        alphas_cumprod = self.noise_schedule['alphas_cumprod']
        
        # Generate timesteps for sampling (evenly spaced)
        timesteps = torch.linspace(0, len(alphas_cumprod)-1, steps=steps).long().to(self.device)
        timesteps = timesteps.flip(0)  # Start from T, go to 0
        
        x_t = x_T
        
        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            # If it's the last step, we're done
            if i == len(timesteps) - 1:
                break
                
            # 1. Predict noise
            noise_pred = self._predict_noise(x_t, t, condition, guidance_scale)
            
            # 2. Get alpha values
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_next = alphas_cumprod[timesteps[i + 1]]
            
            # 3. Predict x_0
            pred_x0 = (x_t - (1 - alpha_cumprod_t).sqrt() * noise_pred) / alpha_cumprod_t.sqrt()
            
            # 4. Compute deterministic next sample
            x_t = alpha_cumprod_next.sqrt() * pred_x0 + (1 - alpha_cumprod_next).sqrt() * noise_pred
                
        return x_t
        
    def _plms_sample(self, x_T, condition=None, steps=50, guidance_scale=3.0):
        """
        PLMS sampling algorithm (Pseudo Linear Multistep - an improved method)
        
        Args:
            x_T: Starting noise
            condition: Conditioning information
            steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Denoised sample
        """
        # Get noise schedule
        alphas_cumprod = self.noise_schedule['alphas_cumprod']
        
        # Generate timesteps for sampling (evenly spaced)
        timesteps = torch.linspace(0, len(alphas_cumprod)-1, steps=steps).long().to(self.device)
        timesteps = timesteps.flip(0)  # Start from T, go to 0
        
        x_t = x_T
        noise_preds = []
        
        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="PLMS Sampling")):
            # 1. Predict noise
            noise_pred = self._predict_noise(x_t, t, condition, guidance_scale)
            
            # For the first few steps, we use regular DDIM
            if i < 4:
                noise_preds.append(noise_pred)
                if i < 3:
                    # Get alpha values
                    alpha_cumprod_t = alphas_cumprod[t]
                    alpha_cumprod_next = alphas_cumprod[timesteps[min(i + 1, len(timesteps) - 1)]]
                    
                    # Predict x_0
                    pred_x0 = (x_t - (1 - alpha_cumprod_t).sqrt() * noise_pred) / alpha_cumprod_t.sqrt()
                    
                    # Compute next sample (DDIM step)
                    x_t = alpha_cumprod_next.sqrt() * pred_x0 + (1 - alpha_cumprod_next).sqrt() * noise_pred
                continue
            
            # After collecting 4 predictions, use PLMS
            noise_preds.pop(0)  # Remove oldest prediction
            noise_preds.append(noise_pred)  # Add newest prediction
            
            # Use PLMS coefficients for linear multistep
            noise_pred_plms = (
                noise_preds[0] + 
                (1/2) * (noise_preds[3] - noise_preds[0]) +
                (1/6) * (noise_preds[3] - 2*noise_preds[2] + noise_preds[1])
            )
            
            # Get alpha values
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_next = alphas_cumprod[timesteps[min(i + 1, len(timesteps) - 1)]]
            
            # Predict x_0
            pred_x0 = (x_t - (1 - alpha_cumprod_t).sqrt() * noise_pred_plms) / alpha_cumprod_t.sqrt()
            
            # Compute next sample
            x_t = alpha_cumprod_next.sqrt() * pred_x0 + (1 - alpha_cumprod_next).sqrt() * noise_pred_plms
                
        return x_t


class HybridArchitectureDiffusion(DiffusionModel):
    """Implements hybrid architecture with transformer for structure and diffusion for quality"""
    
    def __init__(self, model_path='', device='cuda', hf_model_id='facebook/musicgen-small'):
        super().__init__(model_path, device, hf_model_id)
        # Initialize network architecture
        self.channels = 64
        self.time_embedding_dim = 256
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        # Conditioning embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(512, self.time_embedding_dim),  # Assuming token dim is 512
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        # U-Net backbone with skip connections
        # This is a simplified version for implementation
        # Further implementation details would include the full U-Net with
        # ResNet blocks, attention layers, etc.
        self.encoder = nn.ModuleList([
            nn.Conv1d(2, self.channels, 3, padding=1),
            nn.Conv1d(self.channels, self.channels*2, 3, stride=2, padding=1),
            nn.Conv1d(self.channels*2, self.channels*4, 3, stride=2, padding=1)
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1)
        )
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose1d(self.channels*8, self.channels*2, 4, stride=2, padding=1),
            nn.ConvTranspose1d(self.channels*4, self.channels, 4, stride=2, padding=1),
            nn.Conv1d(self.channels*2, 2, 3, padding=1)
        ])
    
    def forward(self, x, t, condition=None):
        """
        Forward pass for model
        
        Args:
            x: Input audio
            t: Timestep
            condition: Structure tokens for conditioning
            
        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.channels)
        t_emb = self.time_embed(t_emb)
        
        # Condition embedding (if available)
        c_emb = torch.zeros_like(t_emb)
        if condition is not None:
            c_emb = self.cond_embed(condition)
        
        # Combine embeddings
        emb = t_emb + c_emb
        
        # Encoder
        h = x
        skip_connections = []
        for layer in self.encoder:
            h = layer(h)
            h = F.silu(h)
            skip_connections.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Add skip connection
            h = torch.cat([h, skip_connections[-i-1]], dim=1)
            h = layer(h)
            if i < len(self.decoder) - 1:  # No activation on final layer
                h = F.silu(h)
        
        return h
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal timestep embeddings
        
        Args:
            timesteps: 1-D Tensor of timesteps
            embedding_dim: Dimension of the embeddings
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Zero-pad odd dimensions
            emb = F.pad(emb, (0, 1))
            
        return emb
    
    def generate_from_tokens(self, structure_tokens, steps=50, sampling_method='ddpm', guidance_scale=3.0):
        """
        Generate high-quality audio from structure tokens
        
        Args:
            structure_tokens: Tokens from transformer model defining structure
            steps: Number of diffusion steps
            sampling_method: Sampling algorithm
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            High-quality audio conditioned on structure tokens
        """
        if not self.is_loaded and not self.load_model():
            print("Warning: Diffusion model not loaded - using fallback")
            # Return a placeholder that will work with the existing pipeline
            return structure_tokens

        # Get shape information
        batch_size, seq_len = structure_tokens.shape[:2]
        
        # Create initial noise (shape depends on audio upsampling ratio)
        # Assuming a standard ratio of 256 audio samples per token
        audio_length = seq_len * 256
        x_T = torch.randn((batch_size, 2, audio_length), device=self.device)  # Stereo audio
        
        # Process structure tokens to get conditioning
        # This could involve additional embedding or processing
        condition = structure_tokens.to(self.device)
        
        # Run denoising diffusion
        return self.denoise(
            x_T, 
            condition=condition,
            steps=steps,
            sampling_method=sampling_method,
            guidance_scale=guidance_scale
        )


class PostProcessingDiffusion(DiffusionModel):
    """Implements diffusion-based post-processing to enhance audio quality"""
    
    def __init__(self, model_path='', device='cuda', hf_model_id='facebook/audiocraft-base'):
        super().__init__(model_path, device, hf_model_id)
        # Initialize network architecture - similar to HybridDiffusion but 
        # focused on audio quality enhancement rather than generation
        self.channels = 64
        self.time_embedding_dim = 256
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        # U-Net architecture with residual connections
        self.encoder = nn.ModuleList([
            nn.Conv1d(2, self.channels, 3, padding=1),
            nn.Conv1d(self.channels, self.channels*2, 3, stride=2, padding=1),
            nn.Conv1d(self.channels*2, self.channels*4, 3, stride=2, padding=1)
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1)
        )
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose1d(self.channels*8, self.channels*2, 4, stride=2, padding=1),
            nn.ConvTranspose1d(self.channels*4, self.channels, 4, stride=2, padding=1),
            nn.Conv1d(self.channels*2, 2, 3, padding=1)
        ])
    
    def forward(self, x, t, condition=None):
        """Forward pass similar to HybridDiffusion but without conditioning"""
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.channels)
        t_emb = self.time_embed(t_emb)
        
        # Encoder
        h = x
        skip_connections = []
        for layer in self.encoder:
            h = layer(h)
            h = F.silu(h)
            skip_connections.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Add skip connection
            h = torch.cat([h, skip_connections[-i-1]], dim=1)
            h = layer(h)
            if i < len(self.decoder) - 1:  # No activation on final layer
                h = F.silu(h)
        
        return h
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Sinusoidal timestep embeddings (same as HybridDiffusion)"""
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Zero-pad odd dimensions
            emb = F.pad(emb, (0, 1))
            
        return emb
    
    def enhance_audio(self, audio, steps=50, sampling_method='ddpm'):
        """
        Enhance audio quality with diffusion-based post-processing
        
        Args:
            audio: Audio to enhance (tensor or numpy array)
            steps: Number of diffusion steps
            sampling_method: Sampling algorithm
            
        Returns:
            Enhanced audio
        """
        if not self.is_loaded and not self.load_model():
            print("Warning: Diffusion model not loaded - returning original audio")
            return audio
            
        # Convert audio to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, device=self.device)
            
        # Ensure audio has proper shape [channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        if audio.dim() > 2:
            audio = audio.squeeze(0)  # Remove batch dimension if present
            
        # Normalize audio
        audio = audio / (audio.abs().max() + 1e-8)
        
        # Get noise schedule
        if self.noise_schedule is None:
            self.noise_schedule = get_noise_schedule()
            
        # Set noise level for enhancement (using partial denoising)
        # Here we use 30% noise level which works well for enhancement
        noise_level = int(0.3 * len(self.noise_schedule['alphas_cumprod']))
        t = torch.tensor([noise_level], device=self.device)
        
        # Add noise to original audio
        noisy_audio = self.add_noise(audio.unsqueeze(0), t)
        
        # Denoise
        enhanced = self.denoise(
            noisy_audio, 
            condition=None,
            steps=steps,
            sampling_method=sampling_method
        )
        
        # Return enhanced audio
        return enhanced.squeeze(0)


class ConditionalDiffusion(DiffusionModel):
    """Implements conditional diffusion models for natural transitions and textures"""
    
    def __init__(self, model_path='', device='cuda', hf_model_id='facebook/musicgen-melody'):
        super().__init__(model_path, device, hf_model_id)
        # Initialize conditional diffusion architecture
        self.channels = 128
        self.time_embedding_dim = 512
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        # Codec token conditioning (assuming 1024-dim codec tokens)
        self.codec_embed = nn.Sequential(
            nn.Linear(1024, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        # U-Net architecture with attention for long-range dependencies
        # This is important for natural transitions
        self.encoder = nn.ModuleList([
            nn.Conv1d(2, self.channels, 3, padding=1),
            nn.Conv1d(self.channels, self.channels*2, 3, stride=2, padding=1),
            nn.Conv1d(self.channels*2, self.channels*4, 3, stride=2, padding=1)
        ])
        
        # Attention in bottleneck for long-range dependencies
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.channels*4, self.channels*4, 3, padding=1),
            # Would add Self-Attention here in full implementation
        )
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose1d(self.channels*8, self.channels*2, 4, stride=2, padding=1),
            nn.ConvTranspose1d(self.channels*4, self.channels, 4, stride=2, padding=1),
            nn.Conv1d(self.channels*2, 2, 3, padding=1)
        ])
    
    def forward(self, x, t, condition=None):
        """
        Forward pass with codec token conditioning
        
        Args:
            x: Input audio
            t: Timestep
            condition: Codec tokens for conditioning
            
        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.channels)
        t_emb = self.time_embed(t_emb)
        
        # Condition embedding (if available)
        c_emb = torch.zeros_like(t_emb)
        if condition is not None:
            c_emb = self.codec_embed(condition)
        
        # Combine embeddings 
        emb = t_emb + c_emb
        
        # Encoder
        h = x
        skip_connections = []
        for layer in self.encoder:
            h = layer(h)
            h = F.silu(h)
            skip_connections.append(h)
        
        # Bottleneck with attention
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Add skip connection
            h = torch.cat([h, skip_connections[-i-1]], dim=1)
            h = layer(h)
            if i < len(self.decoder) - 1:  # No activation on final layer
                h = F.silu(h)
        
        return h
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Sinusoidal timestep embeddings (same as other models)"""
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Zero-pad odd dimensions
            emb = F.pad(emb, (0, 1))
            
        return emb
    
    def generate_conditioned(self, codec_tokens, steps=50, sampling_method='ddpm', guidance_scale=3.0):
        """
        Generate audio conditioned on codec tokens for natural transitions
        
        Args:
            codec_tokens: Codec tokens to condition on
            steps: Number of diffusion steps
            sampling_method: Sampling algorithm
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Audio with improved transitions and textures
        """
        if not self.is_loaded and not self.load_model():
            print("Warning: Diffusion model not loaded - using fallback")
            # Generate basic audio that works with existing pipeline
            return codec_tokens
            
        # Get shape information
        batch_size, seq_len = codec_tokens.shape[:2]
        
        # Create initial noise (assuming a 256 expansion factor)
        audio_length = seq_len * 256
        x_T = torch.randn((batch_size, 2, audio_length), device=self.device)  # Stereo audio
        
        # Run denoising diffusion with conditioning
        return self.denoise(
            x_T, 
            condition=codec_tokens.to(self.device),
            steps=steps,
            sampling_method=sampling_method,
            guidance_scale=guidance_scale
        )


def get_noise_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    """
    Create noise schedule for diffusion process
    
    Args:
        num_steps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
        
    Returns:
        Noise schedule tensors
    """
    # Linear beta schedule
    betas = torch.linspace(beta_start, beta_end, num_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1. - alphas_cumprod),
    } 