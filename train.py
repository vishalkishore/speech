import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import datetime
import time
import argparse
import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.loss import pesq_score, phase_losses, gaussian_latent_loss
from models.discriminator import MetricDiscriminator, batch_pesq
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

torch.backends.cudnn.benchmark = True

def append_eval_log(exp_path: str, record: dict):
    """Append one JSON record (one line) to eval_log.json inside exp_path."""
    log_path = os.path.join(exp_path, 'eval_log.json')
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    optim_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

def create_dataset(cfg, split_name='train', split=True, device='cuda:0'):
    """Create dataset for the requested split."""
    split_keys = {
        'train': ('train_clean_json', 'train_noisy_json'),
        'valid': ('valid_clean_json', 'valid_noisy_json'),
        'test': ('test_clean_json', 'test_noisy_json'),
    }
    if split_name not in split_keys:
        raise ValueError(f"Unsupported split_name: {split_name}")

    clean_key, noisy_key = split_keys[split_name]
    clean_json = cfg['data_cfg'][clean_key]
    noisy_json = cfg['data_cfg'][noisy_key]
    is_train = split_name == 'train'
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if is_train else False
    pcs = cfg['training_cfg']['use_PCS400'] if is_train else False
    
    center = cfg['stft_cfg'].get('center', True)
    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs,
        center=center
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )


def _get_scheduled_weight(base_weight, step, start_steps=0, warmup_steps=0):
    """Linear warm-up scheduler for auxiliary loss weights.

    Returns 0 before start_steps, then linearly ramps from 0 to base_weight
    over warmup_steps steps, and stays at base_weight afterwards.
    """
    if base_weight <= 0.0:
        return 0.0
    if step < start_steps:
        return 0.0
    if warmup_steps <= 0:
        return base_weight
    progress = min(1.0, (step - start_steps + 1) / warmup_steps)
    return base_weight * progress


def evaluate_loader(generator, loader, cfg, device, n_fft, hop_size, win_size, compress_factor):
    """Run one full evaluation pass over a loader and return aggregate metrics."""
    audios_r, audios_g = [], []
    mag_err_tot = 0.0
    pha_err_tot = 0.0
    com_err_tot = 0.0
    gen = generator.module if hasattr(generator, 'module') else generator
    gen.set_streaming(True)

    with torch.no_grad():
        num_batches = 0
        for batch in loader:
            num_batches += 1
            gen.mag_norm.reset_state()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audios_r += torch.split(clean_audio, 1, dim=0)
            audios_g += torch.split(audio_g, 1, dim=0)

            mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
            ip_err, gd_err, iaf_err = phase_losses(clean_pha, pha_g, cfg)
            pha_err_tot += (ip_err + gd_err + iaf_err).item()
            com_err_tot += F.mse_loss(clean_com, com_g).item()

    gen.set_streaming(False)
    return {
        'mag_loss': mag_err_tot / num_batches,
        'phase_loss': pha_err_tot / num_batches,
        'complex_loss': com_err_tot / num_batches,
        'pesq': pesq_score(audios_r, audios_g, cfg).item(),
    }


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']

    # Latent regularisation / prediction settings (all optional, default to off)
    latent_cfg = cfg.get('latent_cfg', {})
    latent_gaussian_weight         = latent_cfg.get('gaussian_weight', 0.0)
    latent_gaussian_start_steps    = latent_cfg.get('gaussian_start_steps', 0)
    latent_gaussian_warmup_steps   = latent_cfg.get('gaussian_warmup_steps', 0)
    latent_prediction_weight       = latent_cfg.get('prediction_weight', 0.0)
    latent_prediction_start_steps  = latent_cfg.get('prediction_start_steps', 0)
    latent_prediction_warmup_steps = latent_cfg.get('prediction_warmup_steps', 0)
    prediction_horizon             = latent_cfg.get('prediction_horizon', 1)
    use_latent = (latent_gaussian_weight > 0.0) or (latent_prediction_weight > 0.0)

    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg['training_cfg'].get('use_pretrainedD', False):
        discriminator.load_state_dict( torch.load('ckpts/pretrained_discriminator.pth') )
        print("Loaded pretrained weight from ckpts/pretrained_discriminator.pth.")

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, split_name='train', split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)

    # Create eval datasets and loaders if rank is 0
    if rank == 0:
        validset = create_dataset(cfg, split_name='valid', split=False, device=device)
        testset = create_dataset(cfg, split_name='test', split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)
        test_loader = create_dataloader(testset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq, best_pesq_step = 0.0, 0
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g, aux = generator(noisy_mag, noisy_pha, return_latent=True)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g, cfg)

            # Discriminator
            # ------------------------------------------------------- #
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            
            loss_disc_all.backward()
            optim_d.step()
            # ------------------------------------------------------- #
            
            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()

            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Time Loss
            # Ensure causal STFT output and target are the same length (like DPDFNet)
            min_len = min(clean_audio.shape[-1], audio_g.shape[-1])
            clean_audio = clean_audio[..., :min_len]
            audio_g = audio_g[..., :min_len]
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            for param in discriminator.parameters():
                param.requires_grad = False
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            for param in discriminator.parameters():
                param.requires_grad = True
            # Consistancy Loss
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            # Latent Gaussian (SIGREG-style) and Prediction losses
            latent_h = aux['latent_h']  # [B, T, hid]
            loss_latent_gaussian = gaussian_latent_loss(latent_h) if latent_gaussian_weight > 0.0 \
                else torch.zeros((), device=device)
            loss_latent_prediction = torch.zeros((), device=device)
            latent_pred = aux.get('latent_pred')
            if latent_pred is not None and prediction_horizon > 0:
                # Accumulate MSE for every horizon h in [1 .. prediction_horizon].
                # latent_pred[t] predicts latent_h[t+h] for all h simultaneously.
                # Only include horizons where the sequence is long enough.
                horizon_losses = []
                for h in range(1, prediction_horizon + 1):
                    if latent_h.size(1) > h:
                        h_start = (h - 1) * latent_h.size(-1)
                        h_end = h * latent_h.size(-1)
                        pred_h = latent_pred[:, :-h, h_start:h_end]
                        horizon_losses.append(F.mse_loss(
                            pred_h,
                            latent_h[:, h:, :].detach(),
                        ))
                if horizon_losses:
                    loss_latent_prediction = torch.stack(horizon_losses).mean()
            current_latent_gaussian_weight = _get_scheduled_weight(
                latent_gaussian_weight, steps,
                latent_gaussian_start_steps, latent_gaussian_warmup_steps,
            )
            current_latent_prediction_weight = _get_scheduled_weight(
                latent_prediction_weight, steps,
                latent_prediction_start_steps, latent_prediction_warmup_steps,
            )

            loss_gen_all = (
                loss_metric * cfg['training_cfg']['loss']['metric'] +
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg['training_cfg']['loss']['phase'] +
                loss_com * cfg['training_cfg']['loss']['complex'] +
                loss_time * cfg['training_cfg']['loss']['time'] + 
                loss_con * cfg['training_cfg']['loss']['consistancy'] +
                loss_latent_gaussian * current_latent_gaussian_weight +
                loss_latent_prediction * current_latent_prediction_weight
            )

            loss_gen_all.backward()
            # Gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
            optim_g.step()
            # ------------------------------------------------------- #

            if rank == 0:
                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g, cfg)
                        pha_error = (loss_ip + loss_gd + loss_iaf).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        con_error = F.mse_loss( com_g, rec_com ).item()
                        lat_g_err = loss_latent_gaussian.item()
                        lat_p_err = loss_latent_prediction.item()

                        print(
                            'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                            'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, '
                            'Cons Loss: {:4.3f}, LatGauss: {:4.3f}(w={:.1e}), LatPred: {:4.3f}(w={:.1e}), '
                            's/b : {:4.3f}'.format(
                                steps, loss_gen_all, loss_disc_all, metric_error, mag_error, pha_error,
                                com_error, time_error, con_error,
                                lat_g_err, current_latent_gaussian_weight,
                                lat_p_err, current_latent_prediction_weight,
                                time.time() - start_b
                            )
                        )

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                        }
                    )
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )

                # Tensorboard summary logging
                if steps % cfg['env_setting']['summary_interval'] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistancy Loss", con_error, steps)
                    sw.add_scalar("Trainiwhat is happening when trainign loss palteau but validation loss in going down? is it happensng/Latent Gaussian Loss", lat_g_err, steps)
                    sw.add_scalar("Training/Latent Gaussian Weight", current_latent_gaussian_weight, steps)
                    sw.add_scalar("Training/Latent Prediction Loss", lat_p_err, steps)
                    sw.add_scalar("Training/Latent Prediction Weight", current_latent_prediction_weight, steps)

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    print(f"DEBUG NaNs: metric={loss_metric.item():.4f}, mag={loss_mag.item():.4f}, pha={loss_pha.item():.4f}, com={loss_com.item():.4f}, time={loss_time.item():.4f}, con={loss_con.item():.4f}, lat_g={loss_latent_gaussian.item():.4f}, lat_p={loss_latent_prediction.item():.4f}")
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_metrics = evaluate_loader(
                        generator, validation_loader, cfg, device,
                        n_fft, hop_size, win_size, compress_factor
                    )
                    test_metrics = evaluate_loader(
                        generator, test_loader, cfg, device,
                        n_fft, hop_size, win_size, compress_factor
                    )
                    generator.train()

                    # Print best validation PESQ score in terminal
                    if val_metrics['pesq'] >= best_pesq:
                        best_pesq = val_metrics['pesq']
                        best_pesq_step = steps
                    print(
                        'Steps : {:d}, Valid PESQ: {:4.3f}, Test PESQ: {:4.3f}, s/b : {:4.3f}'.format(
                            steps, val_metrics['pesq'], test_metrics['pesq'], time.time() - start_b
                        )
                    )
                    sw.add_scalar("Validation/PESQ Score", val_metrics['pesq'], steps)
                    sw.add_scalar("Validation/Magnitude Loss", val_metrics['mag_loss'], steps)
                    sw.add_scalar("Validation/Phase Loss", val_metrics['phase_loss'], steps)
                    sw.add_scalar("Validation/Complex Loss", val_metrics['complex_loss'], steps)
                    sw.add_scalar("Test/PESQ Score", test_metrics['pesq'], steps)

                    # JSON eval log — one record per validation run
                    append_eval_log(args.exp_path, {
                        'step':          steps,
                        'epoch':         epoch,
                        'valid_pesq':    round(float(val_metrics['pesq']), 4),
                        'test_pesq':     round(float(test_metrics['pesq']), 4),
                        'mag_loss':      round(float(val_metrics['mag_loss']), 6),
                        'phase_loss':    round(float(val_metrics['phase_loss']), 6),
                        'complex_loss':  round(float(val_metrics['complex_loss']), 6),
                        'best_pesq':     round(float(best_pesq),      4),
                        'best_pesq_step': best_pesq_step,
                        'timestamp':     datetime.datetime.utcnow().isoformat() + 'Z',
                    })

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='SEMamba_advanced')
    parser.add_argument('--config', default='recipes/SEMamba_advanced/SEMamba_advanced.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)
        

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()
