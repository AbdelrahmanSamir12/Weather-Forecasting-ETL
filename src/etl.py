from src.motherduck_utils import get_connection
from src.feature_engineer import feature_engineer_shiffted, feature_engineer


def preprocess_and_store():
    conn = get_connection()

    # Read raw data
    df_raw = conn.execute("SELECT * FROM raw_weather_data").fetchdf()

    # Datetime Features: Extract Temporal Patterns

    df = feature_engineer_shiffted(df_raw)


    
    conn.register("processed_df", df)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS processed_weather_data AS 
        SELECT * FROM processed_df WHERE FALSE;
    """)

    # Save to processed table
    
    conn.execute("DELETE FROM processed_weather_data")  # clear old
    conn.execute("""
        INSERT INTO processed_weather_data 
        SELECT * FROM processed_df
    """)

    conn.close()

if __name__ == "__main__":
    preprocess_and_store()