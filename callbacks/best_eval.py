import os, json, shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

import lpips

# SSIM: compatible con varias versiones de torchmetrics, con fallback a scikit-image
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure  # torchmetrics >=0.11 aprox
except Exception:
    try:
        from torchmetrics import StructuralSimilarityIndexMeasure     # otras versiones
    except Exception:
        StructuralSimilarityIndexMeasure = None

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None

from PIL import Image
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _atomic_write_json(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _rm_and_mkdir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

def _safe_copy(src: str, dst: str):
    """Copia robusta (sin metadata/utime) para FS tipo rclone/FUSE."""
    _ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)

def _to_uint8_img(x_chw: torch.Tensor) -> np.ndarray:
    """
    x_chw: tensor C,H,W en [0,1]
    Devuelve uint8 H,W o H,W,3
    """
    x = x_chw.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        x = x[0]  # H,W
        return (x.numpy() * 255).astype(np.uint8)
    else:
        x = x.permute(1, 2, 0)  # H,W,C
        return (x.numpy() * 255).astype(np.uint8)

def _batch_to_device(batch, device: torch.device):
    """Mueve tensores de un batch dict al device."""
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        return out
    return batch

def _mask_to_1ch_01(mask: torch.Tensor) -> torch.Tensor:
    """
    Convierte mask de posibles layouts a [B,1,H,W] en [0,1].
    Acepta:
      - [B,1,H,W]
      - [B,3,H,W]
      - [B,H,W,1]
      - [B,H,W,3]
    y valores en [-1,1] o [0,1]
    """
    m = mask
    if not torch.is_tensor(m):
        return None

    # Si viene HWC: (B,H,W,C) -> (B,C,H,W)
    if m.dim() == 4 and m.shape[-1] in (1, 3) and (m.shape[1] not in (1, 3)):
        m = m.permute(0, 3, 1, 2).contiguous()

    # Si es 3ch -> 1ch
    if m.dim() == 4 and m.shape[1] != 1:
        m = m.mean(dim=1, keepdim=True)

    # [-1,1] -> [0,1] si hace falta
    if m.min() < -0.1:
        m = (m + 1.0) / 2.0

    return m.clamp(0, 1)


@dataclass
class EvalCfg:
    eval_every_steps: int = 2000
    save_every_steps: int = 2000
    num_samples_metric: int = 64
    num_samples_save: int = 250
    grid_n: int = 16
    w_lpips: float = 1.0
    real_pool_dir: str = ""          # pool fijo de reales (p.ej. testing/)
    real_pool_max: int = 250
    use_fid: bool = False
    fid_real_dir: str = ""           # si calculas FID aparte
    fid_max_real: int = 250
    device: str = "cuda"
    # sampling params (si usas LDM)
    ddim_steps: int = 50
    ddim_eta: float = 1.0
    real_rotate_deg: int = 0
    sample_chunk_size: int = 8   # tamaño de chunk para sampling


class BestEvalCallback(Callback):
    def __init__(self, outdir: str, cfg: EvalCfg):
        super().__init__()
        self.outdir = outdir
        self.cfg = cfg

        self.best_path = os.path.join(outdir, "best.json")
        self.best_dir = os.path.join(outdir, "best")
        self.best_eval_dir = os.path.join(outdir, "best_eval")
        self.best_eval_samples = os.path.join(self.best_eval_dir, "samples")
        self.best_eval_conditions = os.path.join(self.best_eval_dir, "conditions")

        self._lpips = None
        self._ssim = None
        self._real_pool = None  # tensor [M,1,H,W] en [0,1]
        self._real_pool_loaded = False

    def _lazy_init_metrics(self, device: torch.device):
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net="alex").to(device).eval()

        if self._ssim is None:
            if StructuralSimilarityIndexMeasure is not None:
                self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            else:
                self._ssim = "skimage"
                if sk_ssim is None:
                    raise ImportError("No hay StructuralSimilarityIndexMeasure y tampoco skimage instalado.")

    def _load_real_pool(self, device: torch.device):
        """
        Carga un pool fijo de reales en memoria (hasta real_pool_max).
        Misma preparación que el dataset: resize 256, grayscale.
        """
        if self._real_pool_loaded:
            return
        pool_dir = self.cfg.real_pool_dir
        if not pool_dir:
            raise ValueError("real_pool_dir está vacío. Necesito un pool fijo para NN matching.")

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [os.path.join(pool_dir, f) for f in sorted(os.listdir(pool_dir)) if f.lower().endswith(exts)]
        files = files[: self.cfg.real_pool_max]

        imgs = []
        for fp in files:
            im = Image.open(fp).convert("L")
            if self.cfg.real_rotate_deg:
                im = im.rotate(self.cfg.real_rotate_deg, expand=True)
            im = im.resize((256, 256), resample=Image.BILINEAR)
            arr = np.array(im).astype(np.float32) / 255.0  # [0,1]
            t = torch.from_numpy(arr)[None, ...]  # 1,H,W
            imgs.append(t)

        if len(imgs) == 0:
            raise ValueError(f"No encontré imágenes en real_pool_dir={pool_dir}")

        pool = torch.stack(imgs, dim=0).to(device)  # M,1,H,W en [0,1]
        self._real_pool = pool
        self._real_pool_loaded = True

    def _get_val_dataloader(self, trainer):
        """
        Devuelve el val dataloader real (listo para iterar) de forma robusta.
        """
        try:
            # PL puede tener list/tuple
            v = trainer.val_dataloaders
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return v[0]
            if v is not None:
                return v
        except Exception:
            pass

        # Fallback: datamodule
        if getattr(trainer, "datamodule", None) is not None:
            vdl = trainer.datamodule.val_dataloader()
            if isinstance(vdl, (list, tuple)) and len(vdl) > 0:
                return vdl[0]
            return vdl

        raise RuntimeError("No pude obtener val_dataloader para sampling condicionado.")

    @torch.no_grad()
    def _sample_images_conditional(
        self,
        trainer,
        pl_module,
        n: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sampling CONDICIONADO: iteramos val loader (que trae mask),
        llamamos pl_module.log_images(batch, ...) y sacamos samples.
        Devuelve:
          gen_cpu: [n,1,H,W] en [0,1] CPU
          cond_cpu: [n,1,H,W] en [0,1] CPU (o None si no hay mask)
        """
        vdl = self._get_val_dataloader(trainer)

        gen_list: List[torch.Tensor] = []
        cond_list: List[torch.Tensor] = []

        # Iteramos batches de val hasta tener n
        for batch in vdl:
            if len(gen_list) >= n:
                break

            batch = _batch_to_device(batch, device)

            # Generación usando el path estándar del modelo (DDIM)
            # IMPORTANTE: esto suele respetar conditioning_key/cond_stage_key internamente
            images = pl_module.log_images(
                batch,
                split="best_eval",
                ddim_steps=self.cfg.ddim_steps,
                ddim_eta=self.cfg.ddim_eta,
            )

            # Busca tensor 4D
            key = None
            for kk in ["samples", "sample", "x_samples", "pred", "reconstructions"]:
                if kk in images:
                    key = kk
                    break
            if key is None:
                for kk, vv in images.items():
                    if torch.is_tensor(vv) and vv.dim() == 4:
                        key = kk
                        break
            if key is None:
                raise RuntimeError(f"log_images no devolvió tensor 4D. Keys={list(images.keys())}")

            x = images[key]  # [B,C,H,W], típico [-1,1]

            # a [0,1]
            if x.min() < -0.1:
                x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)

            # 1 canal
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)

            # máscara del batch (si existe) -> [B,1,H,W] [0,1]
            m = None
            if isinstance(batch, dict) and ("mask" in batch):
                m = _mask_to_1ch_01(batch["mask"])

            # acumula
            B = x.shape[0]
            for i in range(B):
                if len(gen_list) >= n:
                    break
                gen_list.append(x[i:i+1].detach().cpu())
                if m is not None:
                    cond_list.append(m[i:i+1].detach().cpu())

            del x, images
            torch.cuda.empty_cache()

        if len(gen_list) == 0:
            raise RuntimeError("No se generó ninguna muestra en _sample_images_conditional().")

        gen_cpu = torch.cat(gen_list, dim=0)[:n]  # [n,1,H,W]
        cond_cpu = torch.cat(cond_list, dim=0)[:n] if len(cond_list) else None
        return gen_cpu, cond_cpu

    @torch.no_grad()
    def evaluate_metrics(self, trainer, pl_module, device: torch.device) -> Dict[str, float]:
        self._lazy_init_metrics(device)
        self._load_real_pool(device)

        n = int(self.cfg.num_samples_metric)

        # Sampling condicionado (val batches)
        gen_cpu, _ = self._sample_images_conditional(trainer, pl_module, n=n, device=device)  # CPU [n,1,H,W]

        # NN matching por LPIPS contra real_pool en chunks
        pool = self._real_pool  # [M,1,H,W] [0,1]
        M = pool.shape[0]
        chunk = 32

        ssim_vals = []
        lpips_vals = []

        for i in range(n):
            g = gen_cpu[i:i+1].to(device, non_blocking=True)  # 1,1,H,W
            best_lp = None
            best_j = None

            for j0 in range(0, M, chunk):
                r = pool[j0:j0+chunk]  # c,1,H,W
                g3 = g.repeat(1, 3, 1, 1)
                r3 = r.repeat(1, 3, 1, 1)
                d = self._lpips(g3.expand_as(r3), r3).view(-1)  # [c]
                v, idx = torch.min(d, dim=0)
                if best_lp is None or v.item() < best_lp:
                    best_lp = v.item()
                    best_j = j0 + idx.item()

            nn = pool[best_j:best_j+1]  # 1,1,H,W

            # SSIM
            if self._ssim == "skimage":
                g_np = g[0, 0].detach().cpu().numpy()
                n_np = nn[0, 0].detach().cpu().numpy()
                ssim_vals.append(float(sk_ssim(g_np, n_np, data_range=1.0)))
            else:
                ssim_vals.append(self._ssim(g, nn).item())

            # LPIPS con su NN
            g3 = g.repeat(1, 3, 1, 1)
            n3 = nn.repeat(1, 3, 1, 1)
            lpips_vals.append(self._lpips(g3, n3).view(-1).mean().item())

        ssim_nn = float(np.mean(ssim_vals))
        lpips_nn = float(np.mean(lpips_vals))
        score = ssim_nn - self.cfg.w_lpips * lpips_nn

        return {
            "eval/ssim_nn": ssim_nn,
            "eval/lpips_nn": lpips_nn,
            "eval/score": score,
        }

    def _read_best(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.best_path):
            return None
        with open(self.best_path, "r") as f:
            return json.load(f)

    @rank_zero_only
    def save_best_checkpoint(self, trainer, pl_module, metrics: Dict[str, float], improved: bool):
        if not improved:
            return

        step = int(pl_module.global_step)
        _ensure_dir(self.best_dir)

        # 1) checkpoint overwrite (copiamos last.ckpt)
        last_ckpt = os.path.join(trainer.logdir, "checkpoints", "last.ckpt")
        if not os.path.exists(last_ckpt):
            trainer.save_checkpoint(last_ckpt)

        best_ckpt = os.path.join(self.best_dir, "best.ckpt")
        _safe_copy(last_ckpt, best_ckpt)

        # 2) best.json
        best_obj = {
            "step": step,
            "score": metrics["eval/score"],
            "ssim_nn": metrics["eval/ssim_nn"],
            "lpips_nn": metrics["eval/lpips_nn"],
            "best_ckpt": best_ckpt,
        }
        _atomic_write_json(self.best_path, best_obj)

        # 3) overwrite best_eval/
        _rm_and_mkdir(self.best_eval_dir)
        _rm_and_mkdir(self.best_eval_samples)
        _rm_and_mkdir(self.best_eval_conditions)

        device = pl_module.device
        self._lazy_init_metrics(device)

        n_save = int(self.cfg.num_samples_save)

        # Sampling condicionado, pero lo hacemos por “trozos” para no OOM:
        chunk = int(getattr(self.cfg, "sample_chunk_size", 8))
        chunk = max(1, min(chunk, n_save))

        grid_keep = int(self.cfg.grid_n)
        grid_buf = []

        idx = 0
        for j0 in range(0, n_save, chunk):
            bs = min(chunk, n_save - j0)

            gen_chunk, cond_chunk = self._sample_images_conditional(trainer, pl_module, n=bs, device=device)
            # gen_chunk: [bs,1,H,W] CPU [0,1]
            # cond_chunk: [bs,1,H,W] CPU [0,1] o None

            for b in range(gen_chunk.shape[0]):
                arr = _to_uint8_img(gen_chunk[b])
                Image.fromarray(arr).save(os.path.join(self.best_eval_samples, f"sample_{idx:05d}.png"))

                if cond_chunk is not None:
                    carr = _to_uint8_img(cond_chunk[b])
                    Image.fromarray(carr).save(os.path.join(self.best_eval_conditions, f"mask_{idx:05d}.png"))

                idx += 1

            # grid buffer (solo primeras grid_n)
            if len(grid_buf) < grid_keep:
                take = min(grid_keep - len(grid_buf), gen_chunk.shape[0])
                if take > 0:
                    grid_buf.append(gen_chunk[:take])

            del gen_chunk, cond_chunk
            torch.cuda.empty_cache()

        # grid.png
        if len(grid_buf) > 0:
            grid_tensor = torch.cat(grid_buf, dim=0)  # CPU [k,1,H,W]
            k = grid_tensor.shape[0]
            nrow = int(np.ceil(np.sqrt(k)))
            nrow = max(1, nrow)
            grid = torchvision.utils.make_grid(grid_tensor, nrow=nrow)  # CPU C,H,W
            grid_arr = _to_uint8_img(grid)
            Image.fromarray(grid_arr).save(os.path.join(self.best_eval_dir, "grid.png"))

        # metrics.json
        _atomic_write_json(os.path.join(self.best_eval_dir, "metrics.json"), best_obj)

        # log a W&B “best/*”
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                "best/step": step,
                "best/score": best_obj["score"],
                "best/ssim": best_obj["ssim_nn"],
                "best/lpips": best_obj["lpips_nn"],
            }, step=step)

    def _is_improved(self, new_score: float) -> bool:
        old = self._read_best()
        if old is None:
            return True
        return new_score > float(old.get("score", -1e9))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        step = int(pl_module.global_step)
        if step <= 0:
            return
        if step % self.cfg.eval_every_steps != 0:
            return
        if trainer.global_rank != 0:
            return

        was_training = pl_module.training
        pl_module.eval()

        ctx = getattr(pl_module, "ema_scope", None)
        if callable(ctx):
            ema_ctx = pl_module.ema_scope()
        else:
            class _Null:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ema_ctx = _Null()

        device = pl_module.device
        with torch.no_grad(), ema_ctx:
            metrics = self.evaluate_metrics(trainer, pl_module, device=device)

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)

        if step % self.cfg.save_every_steps == 0:
            improved = self._is_improved(metrics["eval/score"])
            self.save_best_checkpoint(trainer, pl_module, metrics, improved=improved)

        if was_training:
            pl_module.train()

    def on_fit_start(self, trainer, pl_module, *args, **kwargs):
        # si outdir es None o "__trainer_logdir__", úsalo del trainer
        if self.outdir in [None, "", "__trainer_logdir__"]:
            self.outdir = trainer.logdir
            self.best_path = os.path.join(self.outdir, "best.json")
            self.best_dir = os.path.join(self.outdir, "best")
            self.best_eval_dir = os.path.join(self.outdir, "best_eval")
            self.best_eval_samples = os.path.join(self.best_eval_dir, "samples")
            self.best_eval_conditions = os.path.join(self.best_eval_dir, "conditions")
