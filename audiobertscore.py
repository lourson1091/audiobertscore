import logging
from typing import List, Tuple, Union
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi_fbank

# --- AST ----------------------------------------------------
from ast_y.src.models.ast_models import ASTModel

# --- ATSTFrame ----------------------------------------------
from audiossl.methods.atstframe.embedding import (
    load_model as load_atst_model,
    get_timestamp_embedding,
)

# --- BYOL-A v2 ----------------------------------------------
from byol_a2.augmentations import PrecomputedNorm
from byol_a2.models import FlexibleAudioEncoder

logging.getLogger("transformers").setLevel(logging.ERROR)


def make_features(
    waveform: torch.Tensor,
    sr: int = 16000,
    mel_bins: int = 128,
    target_length: int = 1024,
) -> torch.Tensor:
    if sr != 16000:
        raise ValueError("Waveform must be 16 kHz")

    fbank = kaldi_fbank.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    pad = target_length - fbank.size(0)
    fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad)) if pad > 0 else fbank[:target_length]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank.unsqueeze(0)


def bert_score(
    gen: torch.Tensor,
    ref: torch.Tensor,
    lam: float,
    p: float,
) -> Tuple[float, float, float]:
    sim = (gen @ ref.T) / (
        torch.norm(gen, dim=1, keepdim=True) * torch.norm(ref, dim=1).unsqueeze(0)
    )
    sim_pos = torch.clamp(sim, min=0.0)
    term1_p = sim.max(dim=1)[0].mean()
    term1_r = sim.max(dim=0)[0].mean()
    term2_p = ((sim_pos.pow(p).mean(dim=1)).pow(1.0 / p)).mean()
    term2_r = ((sim_pos.pow(p).mean(dim=0)).pow(1.0 / p)).mean()
    global_p = lam * term1_p + (1.0 - lam) * term2_p
    global_r = lam * term1_r + (1.0 - lam) * term2_r
    global_f1 = 2 * global_p * global_r / (global_p + global_r + 1e-8)
    return global_p.item(), global_r.item(), global_f1.item()


class AudioBERTScore:
    def __init__(
        self,
        sr: int = 16000,
        model_type: str = "ast",
        *,
        layer: int = 13,
        byola_mode: str = "local",
        lam: float = -3.5,
        p: float = 106.0,
        use_gpu: bool = True,
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.lam = lam
        self.p = p
        self.sr = sr

        # ---------- Error checking ----------
        if model_type in ("ast", "atstframe"):
            if not (1 <= layer <= 13):
                raise ValueError(f"Invalid layer={layer}. Must be between 1 and 13 for model_type='{model_type}'.")

        if model_type == "byola_v2":
            valid_modes = ("local", "global", "concat")
            if byola_mode not in valid_modes:
                raise ValueError(f"Invalid byola_mode='{byola_mode}'. Must be one of {valid_modes}.")
        # ---------------------------------------

        # ---------------- AST ----------------
        if model_type == "ast":
            self.layer = layer
            ckpt = "audioset_10_10_0.4593.pth"
            self.model = ASTModel(
                label_dim=527, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=True
            )
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
            self.model = torch.nn.DataParallel(self.model).to(self.device).eval()

        elif model_type == "atstframe":
            self.layer = layer
            ckpt = "atstframe_base.ckpt"
            self.model = load_atst_model(ckpt).eval()

        elif model_type == "byola_v2":
            self.layer = 1
            self.byola_mode = byola_mode
            self.model = FlexibleAudioEncoder(n_mels=64, mode=self.byola_mode)
            self.model.load_state_dict(
                torch.load("AudioNTT2022-BYOLA-64x96d2048.pth", map_location=self.device)
            )
            self.model.to(self.device).eval()

            self.to_melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                hop_length=160,
                win_length=400,
                n_mels=64,
                f_min=60,
                f_max=7800,
                power=2.0,
            )
            self.normalizer = PrecomputedNorm([-5.49, 5.03])

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def _prep_ast(self, wav: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        wav = torch.from_numpy(wav).float() if isinstance(wav, np.ndarray) else wav.float()
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)
        feats = (
            make_features(wav.unsqueeze(0), sr=16000).permute(0, 2, 1).unsqueeze(1).to(self.device)
        )
        return feats

    def _tokens_ast(self, feats: torch.Tensor) -> List[torch.Tensor]:
        v = self.model.module.v
        x = torch.cat(
            (v.cls_token.expand(feats.size(0), -1, -1), v.dist_token.expand(feats.size(0), -1, -1), v.patch_embed(feats)),
            dim=1,
        )
        x = v.pos_drop(x + v.pos_embed)
        layers = []
        for blk in v.blocks:
            x = blk(x)
            layers.append(x.clone())
        layers.append(v.norm(x))
        return [y[0, 2:] for y in layers]

    def _tokens_atst(self, wav: Union[np.ndarray, torch.Tensor], sr: int) -> List[torch.Tensor]:
        wav = torch.from_numpy(wav).float() if isinstance(wav, np.ndarray) else wav.float()
        wav = wav.unsqueeze(0) if wav.ndim == 1 else wav
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        with torch.no_grad():
            self.model.cpu()
            timestamp_embeddings, _ = get_timestamp_embedding(wav.cpu(), self.model)
            timestamp_embeddings = [t.to(self.device) for t in timestamp_embeddings]
            self.model.to(self.device)

        norm_last = timestamp_embeddings[-1] / (
            timestamp_embeddings[-1].norm(dim=-1, keepdim=True) + 1e-8
        )
        return timestamp_embeddings + [norm_last]

    def _tokens_byola(self, wav: Union[np.ndarray, torch.Tensor], sr: int) -> List[torch.Tensor]:
        wav = torch.from_numpy(wav).float() if isinstance(wav, np.ndarray) else wav.float()
        wav = wav.unsqueeze(0) if wav.ndim == 1 else wav
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.to(self.device)

        with torch.no_grad():
            lms = self.to_melspec(wav.cpu())
            lms = (lms + 1e-8).log().to(self.device)
            lms = self.normalizer(lms)
            feats = self.model(lms.unsqueeze(0))  # (B, T, D)

        return [feats.squeeze(0)]

    def score(
        self,
        ref_wav: Union[np.ndarray, torch.Tensor],
        ref_sr: int,
        gen_wav: Union[np.ndarray, torch.Tensor],
        gen_sr: int,
    ) -> List[Tuple[float, float, float]]:
        if self.model_type == "ast":
            ref_feats = self._tokens_ast(self._prep_ast(ref_wav, ref_sr))
            gen_feats = self._tokens_ast(self._prep_ast(gen_wav, gen_sr))

        elif self.model_type == "atstframe":
            ref_feats = self._tokens_atst(ref_wav, ref_sr)
            gen_feats = self._tokens_atst(gen_wav, gen_sr)

        elif self.model_type == "byola_v2":
            ref_feats = self._tokens_byola(ref_wav, ref_sr)
            gen_feats = self._tokens_byola(gen_wav, gen_sr)

        else:
            raise ValueError("Unsupported model_type")

        layer_idx = 0 if self.model_type == "byola_v2" else self.layer - 1
        if self.model_type != "byola_v2":
            if not 0 <= layer_idx < len(ref_feats):
                raise IndexError(f"Layer {self.layer} out of range (1-{len(ref_feats)})")

        r, g = ref_feats[layer_idx], gen_feats[layer_idx]
        return [bert_score(g, r, lam=self.lam, p=self.p)]
