import os
import pyodbc
import pandas as pd
import numpy as np
import argparse

def main(args):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=tcp:{args.sql_server},1433;"
        f"DATABASE={args.sql_database};"
        f"UID={args.sql_username};"
        f"PWD={args.sql_password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    query = f"SELECT * FROM {args.sql_table}"
    df = pd.read_sql(query, conn)

    df["ACC_Mag"] = np.sqrt(df["ACC_X"]**2 + df["ACC_Y"]**2 + df["ACC_Z"]**2)

    os.makedirs(args.output_data, exist_ok=True)
    output_file = os.path.join(args.output_data, "preprocessed.csv")
    df.to_csv(output_file, index=False)
    print(f"[SAVE] Preprocessed data -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_server", type=str, required=True)
    parser.add_argument("--sql_database", type=str, required=True)
    parser.add_argument("--sql_username", type=str, required=True)
    parser.add_argument("--sql_password", type=str, required=True)
    parser.add_argument("--sql_table", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()
    main(args)