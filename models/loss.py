# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Penalises the latent distribution for drifting away from an isotropic
    Gaussian by comparing sliced characteristic-function statistics.
    Accepts a 3-D latent tensor [B, T, C] or [T, B, C].
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        Args:
            proj: [B, T, C] or [T, B, C] latent sequence.
        Returns:
            Scalar SIGReg loss.
        """
        if proj.dim() != 3:
            raise ValueError(f"SIGReg expects a 3-D tensor, got shape {tuple(proj.shape)}")
        if proj.size(0) <= proj.size(1):   # [B, T, C] -> transpose to [T, B, C]
            proj = proj.transpose(0, 1)

        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(torch.finfo(proj.dtype).eps))

        x_t = (proj @ A).unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(device=proj.device, dtype=proj.dtype)).square()
        err = err + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(device=proj.device, dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


def gaussian_latent_loss(latent, eps=1e-6):
    """Lightweight per-channel mean/variance regulariser for the latent.

    Penalises per-channel mean drifting from 0 and variance drifting from 1.

    Args:
        latent: [B, T, C]
    Returns:
        Scalar loss.
    """
    if latent.dim() != 3:
        raise ValueError(f"gaussian_latent_loss expects [B, T, C], got {tuple(latent.shape)}")
    channel_mean = latent.mean(dim=(0, 1))
    channel_var = latent.var(dim=(0, 1), unbiased=False)
    channel_log_var = torch.log(channel_var.clamp_min(eps))
    return channel_mean.square().mean() + channel_log_var.square().mean()


def phase_losses(phase_r, phase_g, cfg):
    """
    Calculate phase losses including in-phase loss, gradient delay loss, 
    and integrated absolute frequency loss between reference and generated phases.
    
    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        h (object): Configuration object containing parameters like n_fft.
    
    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension
    
    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    
    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    
    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    
    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    
    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    
    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    """
    Anti-wrapping function to adjust phase values within the range of -pi to pi.
    
    Args:
        x (torch.Tensor): Input tensor representing phase differences.
    
    Returns:
        torch.Tensor: Adjusted tensor with phase values wrapped within -pi to pi.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def compute_stft(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool, compress_factor: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Short-Time Fourier Transform (STFT) and return magnitude, phase, and complex components.

    Args:
        y (torch.Tensor): Input signal tensor.
        n_fft (int): Number of FFT points.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        center (bool): Whether to pad the input on both sides.
        compress_factor (float, optional): Compression factor for magnitude. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Magnitude, phase, and complex components.
    """
    eps = torch.finfo(y.dtype).eps
    hann_window = torch.hann_window(win_size).to(y.device)
    
    stft_spec = torch.stft(
        y, 
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window, 
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        return_complex=True
    )
    
    real_part = stft_spec.real
    imag_part = stft_spec.imag

    # Match the project STFT path: magnitude is sqrt(re^2 + im^2)
    mag_sq = real_part.pow(2) + imag_part.pow(2) + eps
    mag = torch.sqrt(mag_sq)
    pha = torch.atan2(imag_part, real_part) # For return/logging purposes only

    compressed_mag = torch.pow(mag, compress_factor)
    
    # Bypass atan2 for complex generation
    ratio = compressed_mag / mag
    com_real = real_part * ratio
    com_imag = imag_part * ratio
    com = torch.stack((com_real, com_imag), dim=-1)
    mag = compressed_mag
    
    return mag, pha, com

def pesq_score(utts_r, utts_g, cfg):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score for pairs of reference and generated utterances.
    
    Args:
        utts_r (list of torch.Tensor): List of reference utterances.
        utts_g (list of torch.Tensor): List of generated utterances.
        h (object): Configuration object containing parameters like sampling_rate.
    
    Returns:
        float: Mean PESQ score across all pairs of utterances.
    """
    def eval_pesq(clean_utt, esti_utt, sr):
        """
        Evaluate PESQ score for a single pair of clean and estimated utterances.
        
        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.
        
        Returns:
            float: PESQ score or -1 in case of an error.
        """
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt, 'wb')
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1
        return pesq_score
    
    # Parallel processing of PESQ score computation
    num_worker = cfg['env_setting']['num_workers']
    pesq_scores = Parallel(n_jobs=num_worker)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))
    
    pesq_scores = np.array(pesq_scores)
    valid_scores = pesq_scores[pesq_scores != -1]
    if valid_scores.size == 0:
        return np.nan

    # Calculate mean PESQ score over valid utterances only.
    pesq_score = np.mean(valid_scores)
    return pesq_score
