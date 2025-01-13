import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Date, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert, select
from config.settings import DATABASE_URI
from config.logging_config import logger

class DatabaseManager:
    def __init__(self, db_uri: str = DATABASE_URI):
        self.engine = create_engine(db_uri, echo=False)
        self.metadata = MetaData()
        self._create_tables()

    def _create_tables(self):
        if not inspect(self.engine).has_table("historical_data"):
            Table(
                "historical_data", self.metadata,
                Column("ticker", String, primary_key=True),
                Column("date", Date, primary_key=True),
                Column("open", Float),
                Column("high", Float),
                Column("low", Float),
                Column("close", Float),
                Column("adj_close", Float),
                Column("volume", Float),
            )
            self.metadata.create_all(self.engine)
            logger.info("Created table 'historical_data'.")
        else:
            logger.info("Table 'historical_data' already exists.")

    def insert_historical_data(self, ticker: str, data: pd.DataFrame) -> None:
        try:
            data = data.copy()
            data["ticker"] = ticker
            # rename as needed to match DB columns
            data.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                    "Date": "date",
                },
                inplace=True
            )
            data = data[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
            data.to_sql("historical_data", self.engine, if_exists="append", index=False)
            logger.info(f"Inserted {len(data)} rows for {ticker}.")
        except SQLAlchemyError as e:
            logger.error(f"Insert error for {ticker}: {e}")

    def fetch_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        query = f"""
            SELECT * FROM historical_data
            WHERE ticker='{ticker}' AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC
        """
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Fetched data for {ticker}: {len(df)} rows.")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Fetch error for {ticker}: {e}")
            return pd.DataFrame()

    def load_all_historical_data(self) -> pd.DataFrame:
        try:
            df = pd.read_sql_table("historical_data", self.engine)
            logger.info("Loaded all historical data.")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Error loading all data: {e}")
            return pd.DataFrame()
