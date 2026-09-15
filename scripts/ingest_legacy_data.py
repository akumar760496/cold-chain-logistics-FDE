import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"
)
TABLE_NAME = "TBL_SC_FLEET_HIST_RAW"

LEGACY_MAPPING = {
    "timestamp": "TS_UTC",
    "vehicle_gps_latitude": "V_LAT",
    "vehicle_gps_longitude": "V_LON",
    "iot_temperature": "IOT_TEMP_VAL_C",
    "cargo_condition_status": "CGO_COND_CD",
    "risk_classification": "RISK_CLS_TXT",
    "delay_probability": "DELAY_PROB_DEC",
    "port_congestion_level": "PRT_CNG_LVL",
    "route_risk_level": "RT_RSK_IDX",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the logistics CSV into SQL Server."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--if-exists",
        choices=("fail", "replace", "append"),
        default="replace",
        help="How to handle an existing table (default: replace).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200,
        help="Rows inserted per batch (default: 200).",
    )
    return parser.parse_args()


def build_engine():
    load_dotenv(PROJECT_ROOT / ".env")
    user_key = "SQL_ADMIN_USER"
    secret_key = "SQL_ADMIN_PASSWORD"
    required = (user_key, secret_key)
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    host = os.getenv("SQL_SERVER_HOST", "localhost")
    port = os.getenv("SQL_SERVER_PORT", "1433")
    connect_timeout = int(os.getenv("SQL_SERVER_CONNECT_TIMEOUT", "10"))
    driver = os.getenv("SQL_SERVER_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
    user = os.environ[user_key]
    secret = os.environ[secret_key]
    connection_url = URL.create(
        "mssql+pyodbc",
        user,
        secret,
        host=host,
        port=int(port),
        database="master",
        query={
            "driver": driver,
            "Encrypt": "yes",
            "TrustServerCertificate": "yes",
        },
    )
    return create_engine(
        connection_url,
        connect_args={"timeout": connect_timeout},
        fast_executemany=True,
        pool_pre_ping=True,
    )


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than zero")
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    print(f"Loading CSV from {args.csv}...")
    df = pd.read_csv(args.csv)
    missing_columns = sorted(set(LEGACY_MAPPING) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing_columns)}"
        )

    df_legacy = df[list(LEGACY_MAPPING)].rename(columns=LEGACY_MAPPING)
    df_legacy["SYS_INGEST_FLAG"] = "Y"

    print(f"Connecting to SQL Server and ingesting into dbo.{TABLE_NAME}...")
    engine = build_engine()
    try:
        try:
            df_legacy.to_sql(
                TABLE_NAME,
                engine,
                if_exists=args.if_exists,
                index=False,
                schema="dbo",
                chunksize=args.chunksize,
            )
        except OperationalError as exc:
            host = os.getenv("SQL_SERVER_HOST", "localhost")
            port = os.getenv("SQL_SERVER_PORT", "1433")
            raise RuntimeError(
                f"Could not connect to SQL Server at {host}:{port}. "
                "Check that SQL Server is running and listening on that port, "
                "and that the EC2 security group, network ACL, and host firewall "
                "allow access from this machine."
            ) from exc
    finally:
        engine.dispose()
    print(f"Legacy data ingestion complete: {len(df_legacy)} rows.")


if __name__ == "__main__":
    main()
