import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class AdaptZ(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        # inisialisasi pada weights awal akan menjadi 0, sehingga output akan menjadi 0 jika belum ada update pada weights
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, hisgrad: torch.Tensor,) -> torch.Tensor:
        return self.net(torch.cat([z, hisgrad], dim=-1)
        )

def _find_last_linear(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m

    if last is None:
        raise ValueError("Tidak ada layer Linear dalam module")

    return last

# Algoritma 1: Penyesuaian Automatic Delta melalui Tracking di Z-Space
class PersistentDeltaAdjuster(nn.Module):
    """
    encoder = f, x_t -> z_t
    head = g, z -> y_hat
    adapter = h, (z_t, hisgrad_{t-k}) -> delta_t (shape delta=shape_z)
    horizon_k = prediksi horizon k, misal k=1,2,3,4,5
    batch_size_b = jumlah langkah historis untuk menghitung hisgrad & loss
    head_last_layer = layer terakhir g yang ikut diupdate (default: nn.Linear terakhir  di g)
    konveksi indeks (1-based):
        - prediksi ke-s (y_hat_s) punya target y_s
        - y_s baru diketahui pada waktu t = s + k - 1 (yaitu y_{t-k+1} diobservasi pada waktu t)

    CATATAN: encoder f dan seluruh parameter g dibekukan, kecuali last layer g.
    """ 
    def __init__(
            self,
            encoder: nn.Module,
            head: nn.Module,
            adapter: nn.Module,
            horizon_k: int,
            batch_size_b: int,
            head_last_layer: Optional[nn.Module] = None,
            lr: float = 1e-3,
            grad_reduce: str = "last", # "last": grad di z_{t-k+1} | "mean": rata-rata window
            device: Optional[str] = None,
    ):
        assert horizon_k >=1 and batch_size_b >= 1
        assert grad_reduce in ("last", "mean")
        self.f, self.g, self.A = encoder, head, adapter
        self.k, self.b = horizon_k, batch_size_b
        self.grad_reduce = grad_reduce
        self.device = device or next(adapter.parameters()).device 
        for m in (self.m, self.g, self.A): 
            m.to(self.device)

        # beukukan f dan g, lalu buka hanya last layer dari g
        for p in self.f.paramters():
            p.requires_grad = False
        for p in self.g.paramters():
            p.requires_grad = False
        self.g_last = head_last_layer if head_last_layer is not None else _find_last_linear(self.g)
        for p in self.g_last.parameters():
            p.requires_grad = True

        params = list(self.A.parameters()) + list(self.g_last.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        
        
        self.reset()

    # state
    def reset(self):
        self_t = 0 # waktu saat ini (1-based)
        self_z: Dict[int, torch.Tensor] = {} #z_s
        self.hg_in: Dict[int, torch.Tensor] = {} #hisgrad_{s-k} yang dipakai saat prediksi s
        self.y: Dict[int, torch.Tensor] = {} # y_s (target)
        self.hisgrad: Dict[int, torch.Tensor] = {} # hisgrad_s (hisgrad_1 = 0 --> default zeros)
        self.last_lost: Dict[int, torch.Tensor] = None
    def _get_hisgrad(self, idx: int, like: torch.Tensor) -> torch.Tensor:
        return self.hisgrad.get(idx, torch.zeros_like(like))

    # step
    def step(
            self,
            x_t: torch.Tensor,
            y_new: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        satu iterasi 'for each time stape t'.
        x_t : input pada waktu t
        y_new   : y_{t-k+1} yang baru diobservasi di waktu t (None bila belum ada, ex. t<k)
        return: y_hat_t
        """
        self.t += 1
        t, k, b = self.t, self.k, self.b

        # 3-5 ekstrack fitur, hitung delta, prediksi
        with torch.no_grad():
            z_t = self.f(x_t.to(self.device))
            hg_prev = self._get_hisgrad(t-k, z_t)
            delta_t = self.A(z_t, hg_prev)
            y_hat_t = self.g(z_t, delta_t)
            

    
        

    
