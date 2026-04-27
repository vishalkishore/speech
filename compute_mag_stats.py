import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_vctk import VCTKDemandDataset
import os


def compute_mag_stats(dataset, compress_factor, batch_size=1, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sum_mag    = 0.0
    sum_mag_sq = 0.0
    total_frames = 0

    print("Computing stats... This might take a while.")
    for i, (_, _, _, _, noisy_mag, _) in enumerate(loader):
        # noisy_mag: [B, F, T] — compressed magnitude |STFT|^cf
        # Must use the same dB formula as SEMamba.encode() to match model input
        log_mag_db = (20.0 / compress_factor) * torch.log10(noisy_mag.clamp(min=1e-7))

        sum_mag    += log_mag_db.sum(dim=(0, 2))            # accumulate over B and T → [F]
        sum_mag_sq += (log_mag_db ** 2).sum(dim=(0, 2))
        total_frames += log_mag_db.shape[0] * log_mag_db.shape[2]

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} batches")

    mean = sum_mag / total_frames
    var  = sum_mag_sq / total_frames - mean ** 2
    std  = var.sqrt()

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'mag_stats.pt'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'mean': mean, 'std': std}, save_path)
    print(f"Saved stats to {save_path}")
    print(f"Mean range: [{mean.min():.1f}, {mean.max():.1f}] dB")
    print(f"Std  range: [{std.min():.1f},  {std.max():.1f}]  dB")
    return mean, std


if __name__ == '__main__':
    # match compress_factor in recipes/SEMamba_advanced/SEMamba_advanced.yaml
    CF = 0.2

    dataset = VCTKDemandDataset(
        clean_json="data/train_clean.json",
        noisy_json="data/train_noisy.json",
        sampling_rate=16000,
        segment_size=32000,
        n_fft=400,
        hop_size=200,
        win_size=400,
        compress_factor=CF,
        split=True,     # use same segmentation as training
        shuffle=False,
    )
    compute_mag_stats(dataset, compress_factor=CF, batch_size=1, num_workers=4)
