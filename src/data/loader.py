from __future__ import annotations

import polars as pl

from src.data.data_supabase import load_dataset


def load_ett(name: str) -> pl.DataFrame:
    pdf = load_dataset(name)  
    return pl.from_pandas(pdf).sort("date")

