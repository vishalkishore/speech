import torch
import torch.nn as nn

def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False):
    """
    Compute magnitude and phase using STFT.

    Args:
        y (torch.Tensor): Input audio signal.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.
        eps (bool, optional): Whether adding epsilon to magnitude and phase or not. Defaults to False. 

    Returns:
        tuple: Magnitude, phase, and complex representation of the STFT.
    """
    # DPDFNet-style causal STFT: only compute when enough samples are present, no initial zero-padding
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y, n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
        # Compress the magnitude
        mag = torch.pow(mag, compress_factor)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag_sq = real_part.pow(2) + imag_part.pow(2) + eps
        mag = torch.sqrt(mag_sq)
        pha = torch.atan2(imag_part, real_part) # For return purposes only
        
        # Compress the magnitude
        compressed_mag = torch.pow(mag, compress_factor)
        
        # Analytically stable complex reconstruction bypassing unstable atan2/trig gradients:
        # cos(pha) = real / mag, sin(pha) = imag / mag
        # com_real = compressed_mag * cos(pha) = compressed_mag * (real / mag) = real * (compressed_mag / mag)
        ratio = compressed_mag / mag
        com_real = real_part * ratio
        com_imag = imag_part * ratio
        com = torch.stack((com_real, com_imag), dim=-1)
        mag = compressed_mag

    # Note: causal STFT introduces a delay of win_size samples before first output
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True, length=None):
    """
    Inverse STFT to reconstruct the audio signal from magnitude and phase.

    Args:
        mag (torch.Tensor): Magnitude of the STFT.
        pha (torch.Tensor): Phase of the STFT.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.
        length (int, optional): Output length for causal (center=False) mode. Ignored if center=True.

    Returns:
        torch.Tensor: Reconstructed audio signal.
    """
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    if not center and length is not None:
        wav = torch.istft(
            com,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            length=length
        )
    else:
        wav = torch.istft(
            com,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center
        )
    return wav
