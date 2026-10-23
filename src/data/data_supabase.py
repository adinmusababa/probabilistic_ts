from sqlalchemy import create_engine
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

# format URI: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def load_dataset(dataset="ETTh1"):
    tables = {
        "ETTh1": "ett_h1",
        "ETTh2": "ett_h2",
        "ETTm1": "ett_m1",
        "ETTm2": "ett_m2",
    }
    if dataset not in tables:
        raise ValueError(f"Dataset tidak dikenal: {dataset}")
    table = tables[dataset]
    query = f"""
        SELECT date, hufl, hull, mufl, mull, lufl, lull, ot
        FROM public.{table}
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df
