"""Ledger label: satu-satunya jalan learner membaca data masa lalu pada protokol online.

Pada waktu `now`, label untuk indeks < now sudah terlihat; indeks >= now belum.
Fitur kalender (jam, hari, dst.) bersifat deterministik dan boleh dibaca ke depan.
"""
from __future__ import annotations

import numpy as np


class LeakageError(IndexError):
    """Learner mencoba membaca label yang belum tersedia pada waktu saat ini."""


def _readonly(a: np.ndarray) -> np.ndarray:
    v = np.asarray(a).view()
    v.flags.writeable = False
    return v


class Ledger:
    def __init__(self, values: np.ndarray, time_feats: np.ndarray):
        self._values = _readonly(values)
        self._tf = _readonly(time_feats)
        if len(self._values) != len(self._tf):
            raise ValueError("values dan time_feats harus sama panjang.")
        self.now = 0
        self.n_label_reads = 0
        self.max_label_read = -1  # indeks label terjauh yang pernah dibaca (untuk audit)

    @property
    def n(self) -> int:
        return len(self._values)

    @property
    def n_channels(self) -> int:
        return self._values.shape[1]

    def advance(self, t: int) -> None:
        """Majukan waktu ke t: label untuk indeks < t menjadi terlihat. Waktu tidak boleh mundur."""
        if t < self.now:
            raise ValueError(f"Waktu tidak boleh mundur: {t} < {self.now}.")
        if t > self.n:
            raise ValueError(f"Waktu {t} melewati panjang data {self.n}.")
        self.now = int(t)

    # ---- label ----------------------------------------------------------
    def labels(self, start: int, stop: int) -> np.ndarray:
        """Label indeks [start, stop). stop harus <= now, jika tidak LeakageError."""
        if start < 0 or stop < start:
            raise IndexError(f"Rentang tidak valid: [{start}, {stop}).")
        if stop > self.now:
            raise LeakageError(
                f"Label [{start}, {stop}) belum tersedia pada waktu {self.now} "
                f"(hanya indeks < {self.now} yang terlihat)."
            )
        self.n_label_reads += 1
        if stop > start:
            self.max_label_read = max(self.max_label_read, stop - 1)
        return self._values[start:stop]

    def past_window(self, t: int, length: int) -> np.ndarray:
        """Riwayat `length` langkah sebelum t (indeks t-length .. t-1)."""
        return self.labels(t - length, t)

    def completed_origins(self, horizon: int, past_length: int, count: int = 1) -> list[int]:
        """Origin s terbaru yang seluruh horizonnya sudah berlabel (s + horizon <= now).

        Terbaru lebih dulu. Hanya origin dengan riwayat lengkap (s - past_length >= 0).
        """
        out: list[int] = []
        s = self.now - horizon
        while len(out) < count and s - past_length >= 0:
            out.append(s)
            s -= 1
        return out

    def window(self, s: int, past_length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """(riwayat, masa depan) untuk origin s; masa depan harus sudah terlihat."""
        return self.labels(s - past_length, s), self.labels(s, s + horizon)

    # ---- kalender -------------------------------------------------------
    def time_features(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop < start or stop > self.n:
            raise IndexError(f"Rentang fitur waktu tidak valid: [{start}, {stop}).")
        return self._tf[start:stop]