"""
Database utility for managing pitch data in SQLite.
"""
import sqlite3
import pandas as pd
from src.constants import DATABASE_PATH

def get_db_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    return conn

def save_pitches_to_db(df: pd.DataFrame, table_name: str = "pitches"):
    """
    Saves a DataFrame of pitches to the SQLite database.
    Appends if the table already exists.
    """
    if df.empty:
        return
    
    conn = get_db_connection()
    try:
        # We use if_exists='append' to allow incremental updates
        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"  Successfully saved {len(df)} rows to database table '{table_name}'.")
    finally:
        conn.close()

def query_all_pitches(table_name: str = "pitches") -> pd.DataFrame:
    """Retrieves all pitches from the database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return df
    finally:
        conn.close()

def clear_table(table_name: str = "pitches"):
    """Deletes all data from the specified table."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"Table '{table_name}' cleared.")
    finally:
        conn.close()
