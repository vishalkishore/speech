import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from .norms import MagNorm   

class CausalLatentPredictor(nn.Module):
    """Causal 1-D predictor over the time-averaged latent sequence.

    Given a [B, T, C] latent sequence, predicts the next-step latent
    using only past context (left-only causal padding).
    """
    def __init__(self, channels, hidden_channels=None, kernel_size=3, prediction_horizon=1):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.kernel_size = kernel_size
        self.prediction_horizon = prediction_horizon
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels * prediction_horizon, kernel_size=1),
        )

    def forward(self, x):
        """Args: x [B, T, C]. Returns: [B, T, C * prediction_horizon] predictions."""
        x = x.transpose(1, 2)                   # [B, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0)) # causal left-pad
        x = self.net(x)
        return x.transpose(1, 2)                 # [B, T, C * H]

class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.

    MagNorm modes
    -------------
    Training   (stateful=False) : EMA state resets every forward call.
                                   A random 0-500 phantom warm-up is applied so
                                   the model never sees a cold EMA start.
    Validation / streaming
                (stateful=True) : EMA state is carried frame-to-frame within
                                   one utterance and reset *between* utterances
                                   via reset_state().  This exactly mirrors
                                   real-time streaming inference and eliminates
                                   the train/eval normalization mismatch.

    Call  model.set_streaming(True)  before the validation loop and
          model.mag_norm.reset_state()  before each utterance.
    Call  model.set_streaming(False)  to restore training behaviour.
    """

    def __init__(self, cfg):
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = (
            cfg['model_cfg']['num_tfmamba']
            if cfg['model_cfg']['num_tfmamba'] is not None
            else 4
        )

        hid = cfg['model_cfg']['hid_feature']

        # Dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Causal per-bin EMA normaliser for log-magnitude input.
        # stateful=False at construction; switched to True for eval/streaming
        # via set_streaming().
        self.mag_norm = MagNorm(
            alpha_attack=cfg['model_cfg'].get('mag_norm_alpha_attack', 0.95),
            alpha_decay=cfg['model_cfg'].get('mag_norm_alpha_decay', 0.995),
            stateful=False,
            dynamic_var=True,
        )

        # Mamba blocks
        self.TSMamba = nn.ModuleList(
            [TFMambaBlock(cfg) for _ in range(self.num_tscblocks)]
        )

        # Decoder-specific encoder→decoder skip fusion (concat + 1×1 conv).
        self.mag_skip_fusion = nn.Conv2d(hid * 2, hid, kernel_size=1)
        self.pha_skip_fusion = nn.Conv2d(hid * 2, hid, kernel_size=1)

        # Decoders
        self.mask_decoder  = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

        # Optional causal latent predictor (SIGREG companion).
        latent_cfg = cfg.get('latent_cfg', {})
        self.use_latent_predictor = latent_cfg.get('enable_prediction', False)
        if self.use_latent_predictor:
            self.latent_predictor = CausalLatentPredictor(
                channels=hid,
                hidden_channels=latent_cfg.get('predictor_hidden_channels', hid),
                kernel_size=latent_cfg.get('predictor_kernel_size', 3),
                prediction_horizon=latent_cfg.get('prediction_horizon', 1),
            )

    def set_streaming(self, streaming: bool) -> None:
        """Switch MagNorm between training mode (stateful=False) and
        streaming / validation mode (stateful=True).
        """
        self.mag_norm.stateful = streaming

    def encode(self, noisy_mag, noisy_pha):
        """Encode inputs and run the Mamba stack.

        Returns
        -------
        x            : post-Mamba spatial features  [B, hid, T, F//2]
        encoder_skip : pre-Mamba encoder output      [B, hid, T, F//2]
        aux          : dict with
                         'latent_h'   [B, T, hid]
                         'latent_pred'[B, T, hid]  (only when predictor enabled)
        """
        cf = self.cfg['model_cfg']['compress_factor']

        # Convert compressed magnitude → log dB, then causal EMA-normalise.
        log_mag_db = (20.0 / cf) * torch.log10(noisy_mag.clamp(1e-7))  # [B, F, T]
        mag_in = log_mag_db.permute(0, 2, 1)       # [B, T, F]
        mag_in = self.mag_norm(mag_in)              # causal EMA norm
        mag_in = mag_in.permute(0, 2, 1)            # [B, F, T]

        noisy_mag_r = rearrange(mag_in,      'b f t -> b t f').unsqueeze(1)  # [B,1,T,F]
        noisy_pha_r = rearrange(noisy_pha,   'b f t -> b t f').unsqueeze(1)  # [B,1,T,F]
        x = torch.cat((noisy_mag_r, noisy_pha_r), dim=1)                     # [B,2,T,F]

        x = self.dense_encoder(x)
        encoder_skip = x                            # [B, hid, T, F//2]

        for block in self.TSMamba:
            x = block(x)

        # Frequency-average → [B, T, hid] for latent regularisation.
        latent_h = x.mean(dim=-1).transpose(1, 2).contiguous()
        aux = {'latent_h': latent_h}
        if self.use_latent_predictor:
            aux['latent_pred'] = self.latent_predictor(latent_h)

        return x, encoder_skip, aux

    def decode(self, x, encoder_skip, noisy_mag):
        """Decoder-specific skip fusion → magnitude mask + phase estimate.

        Args
        ----
        x            : post-Mamba features  [B, hid, T, F//2]
        encoder_skip : pre-Mamba features   [B, hid, T, F//2]
        noisy_mag    : original noisy mag   [B, F, T]
        """
        noisy_mag_r = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)   # [B,1,T,F]

        x_mag = self.mag_skip_fusion(torch.cat([x, encoder_skip], dim=1))   # [B,hid,T,F//2]
        x_pha = self.pha_skip_fusion(torch.cat([x, encoder_skip], dim=1))   # [B,hid,T,F//2]

        denoised_mag = rearrange(
            self.mask_decoder(x_mag) * noisy_mag_r, 'b c t f -> b f t c'
        ).squeeze(-1)
        denoised_pha = rearrange(
            self.phase_decoder(x_pha), 'b c t f -> b f t c'
        ).squeeze(-1)
        denoised_com = torch.stack(
            (
                denoised_mag * torch.cos(denoised_pha),
                denoised_mag * torch.sin(denoised_pha),
            ),
            dim=-1,
        )
        return denoised_mag, denoised_pha, denoised_com

    def forward(self, noisy_mag, noisy_pha, return_latent=False):
        """
        Args
        ----
        noisy_mag     : [B, F, T]
        noisy_pha     : [B, F, T]
        return_latent : if True, also return aux dict

        Returns
        -------
        denoised_mag : [B, F, T]
        denoised_pha : [B, F, T]
        denoised_com : [B, F, T, 2]
        aux          : dict  (only when return_latent=True)
        """
        x, encoder_skip, aux = self.encode(noisy_mag, noisy_pha)
        denoised_mag, denoised_pha, denoised_com = self.decode(
            x, encoder_skip, noisy_mag
        )

        if return_latent:
            return denoised_mag, denoised_pha, denoised_com, aux
        return denoised_mag, denoised_pha, denoised_com