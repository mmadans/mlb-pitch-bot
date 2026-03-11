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

def delete_games_from_db(game_pks: list[int], table_name: str = "pitches"):
    """
    Deletes all pitches associated with the given game_pks from the database.
    Useful for ensuring we don't duplicate data when re-scraping a day.
    """
    if not game_pks:
        return
        
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Check if table exists first
        cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone()[0] == 1:
            # Create a parameterized query for the IN clause
            placeholders = ','.join(['?'] * len(game_pks))
            query = f"DELETE FROM {table_name} WHERE game_pk IN ({placeholders})"
            cursor.execute(query, game_pks)
            deleted_count = cursor.rowcount
            conn.commit()
            if deleted_count > 0:
                print(f"  Removed {deleted_count} existing rows for {len(game_pks)} games to prevent duplication.")
    except Exception as e:
        print(f"  Warning: Could not delete old games: {e}")
    finally:
        conn.close()

def save_pitches_to_db(df: pd.DataFrame, table_name: str = "pitches"):
    """
    Saves a DataFrame of pitches to the SQLite database.
    Appends if the table already exists.
    """
    if df.empty:
        return
        
    # Before appending, remove any existing data for these games to prevent dupes
    if 'game_pk' in df.columns:
        unique_games = df['game_pk'].unique().tolist()
        delete_games_from_db(unique_games, table_name)
    
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
    except sqlite3.OperationalError:
        # Table might not exist yet
        return pd.DataFrame()
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
