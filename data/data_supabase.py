import pandas as pd
import os
import pyscopg2 as pg
import dotenv

dotenv.load_dotenv()

# Connect to the Supabase database
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
conn = pg.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
    )



def load_dataset(dataset="ETTh1"):


    tables = {
        "ETTh1": "ett_h1",
        "ETTh2": "ett_h2",
        "ETTm1": "ett_m1",
        "ETTm2": "ett_m2",
    }

    if dataset not in tables:
        raise ValueError(
            f"Dataset tidak dikenal: {dataset}"
        )

    table = tables[dataset]

    query = f"""
        SELECT
            date,
            hufl,
            hull,
            mufl,
            mull,
            lufl,
            lull,
            ot
        FROM public.{table}
        ORDER BY date;
    """

    df = pd.read_sql(
        query,
        conn
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df