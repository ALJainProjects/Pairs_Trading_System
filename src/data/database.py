import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, inspect
from sqlalchemy.exc import SQLAlchemyError
from config.settings import DATABASE_URI
from config.logging_config import logger
from datetime import datetime, timedelta
import numpy as np


class DatabaseManager:
    """
    Database Management System
    """
    def __init__(self, db_uri: str = DATABASE_URI):
        """Initialize database connection and metadata."""
        self.engine = create_engine(db_uri, echo=False)
        self.metadata = MetaData()
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables, dropping existing ones if needed."""
        try:
            self.metadata.clear()

            if inspect(self.engine).has_table("historical_data"):
                inspector = inspect(self.engine)
                if "historical_data" in inspector.get_table_names():
                    historical_data = Table("historical_data", self.metadata, extend_existing=True)
                    historical_data.drop(self.engine)
                    logger.info("Dropped existing historical_data table.")

            Table(
                "historical_data",
                self.metadata,
                Column("symbol", String, primary_key=True),
                Column("date", String, primary_key=True),
                Column("open", Float),
                Column("high", Float),
                Column("low", Float),
                Column("close", Float),
                Column("adj_close", Float),
                Column("volume", Float),
                extend_existing=True
            )

            self.metadata.create_all(self.engine)
            logger.info("Successfully created historical_data table.")

        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            raise

    def insert_historical_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Insert historical data for a given ticker."""
        try:
            data = data.copy()

            data.columns = [col.lower() for col in data.columns]

            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

            data['symbol'] = ticker

            column_mapping = {
                'adj close': 'adj_close',
                'adjusted close': 'adj_close',
                'adjusted_close': 'adj_close',
                'volume': 'volume'
            }
            data.rename(columns=column_mapping, inplace=True)

            required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    if col in ['open', 'high', 'low', 'close']:
                        data[col] = data.get('adj_close', data.get('close', 0))
                    elif col == 'volume':
                        data[col] = 0
                    elif col == 'adj_close':
                        data[col] = data.get('close', 0)

            data = data[required_columns]

            data.to_sql('historical_data', self.engine, if_exists='append', index=False)
            logger.info(f"Successfully inserted {len(data)} rows for {ticker}")

        except SQLAlchemyError as e:
            logger.error(f"Database error inserting data for {ticker}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting data for {ticker}: {str(e)}")
            raise

    def fetch_historical_data(self, ticker: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch historical data with optional filtering."""
        try:
            query = "SELECT * FROM historical_data"
            conditions = []

            if ticker:
                conditions.append(f"symbol = '{ticker}'")

            if start_date:
                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                conditions.append(f"date >= '{start_date}'")

            if end_date:
                end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
                conditions.append(f"date <= '{end_date}'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY symbol, date"

            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Successfully fetched {len(df)} records")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching data: {str(e)}")
            return pd.DataFrame()

    def load_all_historical_data(self) -> pd.DataFrame:
        """Load all historical data from the database."""
        return self.fetch_historical_data()


def main():
    """
    Main function to demonstrate DatabaseManager functionality.
    Creates sample data and performs basic database operations.
    """
    logger.info("Initializing DatabaseManager...")
    db_manager = DatabaseManager()

    logger.info("Creating sample data...")

    end_date = datetime.now().date()
    dates = [(end_date - timedelta(days=x)) for x in range(30)]

    tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in tickers:
        base_price = np.random.uniform(100, 1000)
        daily_returns = np.random.normal(0, 0.02, len(dates))

        prices = [base_price]
        for ret in daily_returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)

        df = pd.DataFrame({
            'date': dates,
            'symbol': ticker,
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'adj_close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        })

        logger.info(f"Inserting sample data for {ticker}...")
        db_manager.insert_historical_data(ticker, df)

    logger.info("Verifying data insertion...")

    for ticker in tickers:
        df = db_manager.fetch_historical_data(
            ticker=ticker,
            start_date=(end_date - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        logger.info(f"\nData for {ticker}:")
        logger.info(f"Number of rows: {len(df)}")
        if not df.empty:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Average price: ${df['adj_close'].mean():.2f}")

    logger.info("\nLoading all historical data...")
    all_data = db_manager.load_all_historical_data()
    logger.info(f"Total number of records: {len(all_data)}")
    logger.info(f"Number of unique symbols: {len(all_data['symbol'].unique())}")


if __name__ == "__main__":
    try:
        main()
        logger.info("Database initialization completed successfully!")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise