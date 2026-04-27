import torch
import argparse
from utils.util import load_config
from models.generator import SEMamba
from models.stfts import mag_phase_stft

def main():
    parser = argparse.ArgumentParser(description="Profile SEMamba GFLOPs and Params")
    parser.add_argument('--config', default='recipes/SEMamba_advanced/SEMamba_advanced.yaml', help='Config yaml')
    parser.add_argument('--seconds', type=float, default=1.0, help='Length of audio in seconds')
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SEMamba(cfg).to(device)
    model.eval()

    # Calculate model parameters manually
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 40)
    print("Model Parameters:")
    print(f"Total Parameters     : {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters : {trainable_params / 1e6:.2f} M")
    print("=" * 40)

    # Audio input setup
    sr = cfg['stft_cfg']['sampling_rate']
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress = cfg['model_cfg']['compress_factor']

    # Dummy input
    T = int(args.seconds * sr)
    dummy_wav = torch.randn(1, T).to(device)

    noisy_amp, noisy_pha, _ = mag_phase_stft(dummy_wav, n_fft, hop_size, win_size, compress)
    
    # Using thop for FLOPs profiling
    try:
        from thop import profile
        # thop doesn't always like kwargs, pass args correctly
        macs, params = profile(model, inputs=(noisy_amp, noisy_pha), verbose=False)
        # 1 MAC = 2 FLOPs usually
        gmacs = macs / 1e9
        
        print("\nComputational Complexity:")
        print(f"For {args.seconds} second(s) of audio:")
        print(f"MACs                 : {gmacs:.3f} GMACs")
        print(f"GFLOPs               : {gmacs * 2:.3f} GFLOPs")
        print("=" * 40)
    except ImportError:
        print("\n[WARN] `thop` is not installed. To get MACs/GFLOPs, run: pip install thop")
    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {e}")
        print("Note: Custom CUDA kernels like Mamba SSM might not be fully supported by profilers out-of-the-box.")

if __name__ == '__main__':
    main()
