import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Date, DateTime, inspect
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import SQLAlchemyError
from config.settings import DATABASE_URL, REALTIME_DATABASE_URL # Import both URLs
from config.logging_config import logger
from datetime import datetime, timedelta
import numpy as np


class HistoricalDatabaseManager: # Renamed from DatabaseManager
    """
    Database Management System for historical financial data.
    Manages connections, table creation, and data operations (insert, upsert, fetch).
    """

    def __init__(self, db_uri: str = DATABASE_URL): # Use DATABASE_URL
        """
        Initialize database connection and metadata.
        Args:
            db_uri (str): The database connection URI.
        """
        self.engine = create_engine(db_uri, echo=False)
        self.metadata = MetaData()
        self._initialize_tables()
        logger.info(f"HistoricalDatabaseManager initialized for URI: {db_uri.split('://')[0]}://...")


    def _initialize_tables(self):
        """
        Initialize historical_data table.
        The 'historical_data' table is defined with 'symbol' and 'date' as primary keys
        to support upsert operations.
        """
        try:
            self.metadata.clear() # Ensure fresh load

            Table(
                "historical_data",
                self.metadata,
                Column("symbol", String(50), primary_key=True),
                Column("date", Date, primary_key=True),
                Column("open", Float),
                Column("high", Float),
                Column("low", Float),
                Column("close", Float),
                Column("adj_close", Float),
                Column("volume", Float),
                extend_existing=True
            )

            self.metadata.create_all(self.engine)
            logger.info("Ensured 'historical_data' table exists with correct schema.")

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during historical table initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing historical tables: {e}")
            raise

    def insert_historical_data(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Insert historical data for a given ticker.
        WARNING: This method is prone to primary key violations if called for existing data
        and the table is not dropped or if using a database without ON CONFLICT.
        It's recommended to use `upsert_historical_data` for robustness.

        Args:
            ticker (str): The stock ticker symbol.
            data (pd.DataFrame): DataFrame containing historical data.
                                 Expected columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
        """
        logger.warning(
            "Using 'insert_historical_data'. For robust data management "
            "that handles existing entries, prefer 'upsert_historical_data'."
        )
        try:
            df_to_insert = data.copy()

            df_to_insert.columns = [col.lower() for col in df_to_insert.columns]

            column_mapping = {
                'adj close': 'adj_close',
                'adjusted close': 'adj_close',
                'adjusted_close': 'adj_close',
                'close': 'adj_close',
            }
            df_to_insert.rename(columns=column_mapping, inplace=True)

            df_to_insert['date'] = pd.to_datetime(df_to_insert['date']).dt.date
            df_to_insert['symbol'] = ticker

            required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in required_columns:
                if col not in df_to_insert.columns:
                    if col in ['open', 'high', 'low', 'close', 'adj_close']:
                        df_to_insert[col] = np.nan
                    elif col == 'volume':
                        df_to_insert[col] = 0

            df_to_insert = df_to_insert[required_columns]

            df_to_insert.to_sql('historical_data', self.engine, if_exists='append', index=False)
            logger.info(f"Successfully inserted {len(df_to_insert)} rows for {ticker}")

        except SQLAlchemyError as e:
            logger.error(f"Database error inserting data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting data for {ticker}: {e}")
            raise

    def upsert_historical_data(self, df: pd.DataFrame) -> None:
        """
        Inserts new historical data or updates existing data if a conflict on (symbol, date) occurs.
        Expects a standardized DataFrame with 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'.

        Args:
            df (pd.DataFrame): DataFrame containing data to upsert.
        """
        if df.empty:
            logger.debug("No data to upsert; DataFrame is empty.")
            return

        df_to_upsert = df.copy()

        df_to_upsert.rename(columns={
            'Date': 'date', 'Symbol': 'symbol', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Adj_Close': 'adj_close', 'Volume': 'volume'
        }, inplace=True)

        df_to_upsert['date'] = pd.to_datetime(df_to_upsert['date']).dt.date

        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in required_columns:
            if col not in df_to_upsert.columns:
                if col in ['open', 'high', 'low', 'close', 'adj_close']:
                    df_to_upsert[col] = np.nan
                elif col == 'volume':
                    df_to_upsert[col] = 0

        df_to_upsert = df_to_upsert[required_columns]

        with self.engine.connect() as conn:
            if self.engine.dialect.name == 'sqlite':
                historical_data_table = Table("historical_data", self.metadata, autoload_with=self.engine)
                data_records = df_to_upsert.to_dict(orient='records')

                stmt = sqlite.insert(historical_data_table).values(data_records)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'adj_close': stmt.excluded.adj_close,
                        'volume': stmt.excluded.volume
                    }
                )
                try:
                    conn.execute(on_conflict_stmt)
                    conn.commit()
                    logger.info(f"Successfully upserted {len(df_to_upsert)} rows into 'historical_data'.")
                except SQLAlchemyError as e:
                    conn.rollback()
                    logger.error(f"Database error during SQLite upsert for historical data: {e}")
                    raise
            else:
                logger.warning(
                    f"Upsert operation not explicitly optimized for {self.engine.dialect.name} for historical data. "
                    "Performing append. This may lead to primary key errors or duplicates "
                    "if not handled by database constraints/triggers (e.g., unique indices)."
                )
                try:
                    df_to_upsert.to_sql('historical_data', conn, if_exists='append', index=False)
                    conn.commit()
                    logger.info(
                        f"Successfully appended {len(df_to_upsert)} rows to 'historical_data' (non-SQLite upsert fallback).")
                except SQLAlchemyError as e:
                    conn.rollback()
                    logger.error(f"Database error during append for historical data (non-SQLite upsert fallback): {e}")
                    raise

    def fetch_historical_data(self, ticker: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical data from the database with optional filtering by ticker and date range.

        Args:
            ticker (str, optional): The stock ticker symbol to fetch data for.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing the fetched historical data.
        """
        try:
            query = "SELECT * FROM historical_data"
            conditions = []

            if ticker:
                conditions.append(f"symbol = '{ticker}'")

            if start_date:
                start_date_obj = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                conditions.append(f"date >= '{start_date_obj}'")

            if end_date:
                end_date_obj = pd.to_datetime(end_date).strftime('%Y-%m-%d')
                conditions.append(f"date <= '{end_date_obj}'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY symbol, date"

            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Successfully fetched {len(df)} records from 'historical_data'.")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching historical data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}")
            return pd.DataFrame()

    def load_all_historical_data(self) -> pd.DataFrame:
        """
        Load all historical data from the database.

        Returns:
            pd.DataFrame: DataFrame containing all historical data.
        """
        logger.info("Loading all historical data from the database.")
        return self.fetch_historical_data()


class RealtimeDatabaseManager: # New class for real-time data
    """
    Database Management System for real-time financial data.
    """

    def __init__(self, db_uri: str = REALTIME_DATABASE_URL): # Use REALTIME_DATABASE_URL
        self.engine = create_engine(db_uri, echo=False)
        self.metadata = MetaData()
        self._initialize_tables()
        logger.info(f"RealtimeDatabaseManager initialized for URI: {db_uri.split('://')[0]}://...")

    def _initialize_tables(self):
        """
        Initialize realtime_bars table.
        The 'realtime_bars' table uses (symbol, datetime) as primary keys.
        """
        try:
            self.metadata.clear() # Ensure fresh load

            Table(
                "realtime_bars",
                self.metadata,
                Column("symbol", String(50), primary_key=True),
                Column("datetime", DateTime, primary_key=True), # Use DateTime for granular data
                Column("open", Float),
                Column("high", Float),
                Column("low", Float),
                Column("close", Float),
                Column("volume", Float),
                Column("fetch_time", DateTime), # Keep track of when data was fetched
                extend_existing=True
            )

            self.metadata.create_all(self.engine)
            logger.info("Ensured 'realtime_bars' table exists with correct schema.")

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during real-time table initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing real-time tables: {e}")
            raise

    def upsert_realtime_data(self, df: pd.DataFrame) -> None:
        """
        Inserts new real-time data or updates existing data if a conflict on (symbol, datetime) occurs.
        Expects a DataFrame with 'Symbol', 'Date' (which will be mapped to 'datetime'), 'Open', 'High', 'Low', 'Close', 'Volume', 'fetch_time'.

        Args:
            df (pd.DataFrame): DataFrame containing data to upsert.
        """
        if df.empty:
            logger.debug("No real-time data to upsert; DataFrame is empty.")
            return

        df_to_upsert = df.copy()

        df_to_upsert.rename(columns={
            'Date': 'datetime', 'Symbol': 'symbol', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'fetch_time': 'fetch_time'
        }, inplace=True)

        # Ensure 'datetime' and 'fetch_time' are proper datetime objects
        df_to_upsert['datetime'] = pd.to_datetime(df_to_upsert['datetime'])
        df_to_upsert['fetch_time'] = pd.to_datetime(df_to_upsert['fetch_time'])


        required_columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'fetch_time']
        for col in required_columns:
            if col not in df_to_upsert.columns:
                if col in ['open', 'high', 'low', 'close']:
                    df_to_upsert[col] = np.nan
                elif col == 'volume':
                    df_to_upsert[col] = 0
                elif col == 'fetch_time':
                    df_to_upsert[col] = datetime.now() # Fallback for fetch_time

        df_to_upsert = df_to_upsert[required_columns]


        with self.engine.connect() as conn:
            if self.engine.dialect.name == 'sqlite':
                realtime_bars_table = Table("realtime_bars", self.metadata, autoload_with=self.engine)
                data_records = df_to_upsert.to_dict(orient='records')

                stmt = sqlite.insert(realtime_bars_table).values(data_records)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'datetime'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume,
                        'fetch_time': stmt.excluded.fetch_time
                    }
                )
                try:
                    conn.execute(on_conflict_stmt)
                    conn.commit()
                    logger.info(f"Successfully upserted {len(df_to_upsert)} rows into 'realtime_bars'.")
                except SQLAlchemyError as e:
                    conn.rollback()
                    logger.error(f"Database error during SQLite upsert for real-time data: {e}")
                    raise
            else:
                logger.warning(
                    f"Upsert operation not explicitly optimized for {self.engine.dialect.name} for real-time data. "
                    "Performing append. This may lead to primary key errors or duplicates "
                    "if not handled by database constraints/triggers (e.g., unique indices)."
                )
                try:
                    df_to_upsert.to_sql('realtime_bars', conn, if_exists='append', index=False)
                    conn.commit()
                    logger.info(
                        f"Successfully appended {len(df_to_upsert)} rows to 'realtime_bars' (non-SQLite upsert fallback).")
                except SQLAlchemyError as e:
                    conn.rollback()
                    logger.error(f"Database error during append for real-time data (non-SQLite upsert fallback): {e}")
                    raise

    def fetch_realtime_data(self, ticker: str = None, start_datetime: str = None, end_datetime: str = None, limit: int = None) -> pd.DataFrame:
        """
        Fetch real-time data from the database with optional filtering.

        Args:
            ticker (str, optional): The stock ticker symbol.
            start_datetime (str, optional): Start datetime in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str, optional): End datetime in 'YYYY-MM-DD HH:MM:SS' format.
            limit (int, optional): Maximum number of records to return.

        Returns:
            pd.DataFrame: DataFrame containing the fetched real-time data.
        """
        try:
            query = "SELECT * FROM realtime_bars"
            conditions = []

            if ticker:
                conditions.append(f"symbol = '{ticker}'")

            if start_datetime:
                start_dt_obj = pd.to_datetime(start_datetime).strftime('%Y-%m-%d %H:%M:%S')
                conditions.append(f"datetime >= '{start_dt_obj}'")

            if end_datetime:
                end_dt_obj = pd.to_datetime(end_datetime).strftime('%Y-%m-%d %H:%M:%S')
                conditions.append(f"datetime <= '{end_dt_obj}'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY symbol, datetime"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['fetch_time'] = pd.to_datetime(df['fetch_time'])

            logger.info(f"Successfully fetched {len(df)} records from 'realtime_bars'.")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching real-time data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching real-time data: {e}")
            return pd.DataFrame()


# Example usage for HistoricalDatabaseManager
def main_historical_db_test():
    """
    Main function to demonstrate HistoricalDatabaseManager functionality using upsert.
    Creates sample data and performs basic database operations.
    """
    logger.info("Initializing HistoricalDatabaseManager...")
    db_manager = HistoricalDatabaseManager()

    logger.info("Creating sample historical data...")

    end_date = datetime.now().date()
    dates = [(end_date - timedelta(days=x)) for x in range(30)]

    tickers = ['AAPL', 'MSFT', 'GOOGL']

    all_sample_data = pd.DataFrame()
    for ticker in tickers:
        np.random.seed(hash(ticker) % (2**32 - 1))
        base_price = np.random.uniform(100, 1000)
        daily_returns = np.random.normal(0, 0.02, len(dates))

        prices = [base_price]
        for ret in daily_returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)

        df_ticker = pd.DataFrame({
            'Date': dates,
            'Symbol': ticker,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
            'Low': prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
            'Close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'Adj_Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        })
        all_sample_data = pd.concat([all_sample_data, df_ticker], ignore_index=True)

    logger.info(f"Upserting {len(all_sample_data)} rows of sample historical data for initial load...")
    db_manager.upsert_historical_data(all_sample_data)

    logger.info("\nModifying some sample historical data for demonstration of upsert update...")
    aapl_data_to_update = all_sample_data[all_sample_data['Symbol'] == 'AAPL'].copy().head(5)
    aapl_data_to_update['Close'] = aapl_data_to_update['Close'] * 1.05
    aapl_data_to_update['Volume'] = aapl_data_to_update['Volume'] + 10000
    db_manager.upsert_historical_data(aapl_data_to_update)
    logger.info("Modified historical data upserted.")

    logger.info("\nVerifying historical data insertion/update...")

    for ticker in tickers:
        df = db_manager.fetch_historical_data(
            ticker=ticker,
            start_date=(end_date - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        logger.info(f"\nData for {ticker}:")
        logger.info(f"Number of rows: {len(df)}")
        if not df.empty:
            logger.info(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            logger.info(f"Average price: ${df['adj_close'].mean():.2f}")
            if ticker == 'AAPL':
                logger.info(f"AAPL first 5 close prices after update:\n{df.head(5)[['date', 'close']]}")

    logger.info("\nLoading all historical data...")
    all_data = db_manager.load_all_historical_data()
    logger.info(f"Total number of historical records: {len(all_data)}")
    logger.info(f"Number of unique symbols in historical data: {len(all_data['symbol'].unique())}")


# Example usage for RealtimeDatabaseManager
def main_realtime_db_test():
    logger.info("Initializing RealtimeDatabaseManager...")
    realtime_db_manager = RealtimeDatabaseManager()

    logger.info("Creating sample real-time data...")
    tickers = ['MSFT', 'GOOGL']
    now = datetime.now()

    realtime_sample_data = pd.DataFrame()
    for ticker in tickers:
        for i in range(5): # 5 minute bars
            timestamp = now - timedelta(minutes=i)
            # Ensure 'Date' column for consistency with LiveDataHandler output
            df_rt_ticker = pd.DataFrame([{
                'Date': timestamp, # This will be mapped to 'datetime'
                'Symbol': ticker,
                'Open': 150.0 + i,
                'High': 150.5 + i,
                'Low': 149.5 + i,
                'Close': 150.2 + i,
                'Volume': 1000 + i*100,
                'fetch_time': now # All fetched at the same time for this example batch
            }])
            realtime_sample_data = pd.concat([realtime_sample_data, df_rt_ticker], ignore_index=True)

    logger.info(f"Upserting {len(realtime_sample_data)} rows of sample real-time data...")
    realtime_db_manager.upsert_realtime_data(realtime_sample_data)

    logger.info("\nFetching real-time data for MSFT...")
    msft_rt_data = realtime_db_manager.fetch_realtime_data(ticker='MSFT')
    logger.info(f"\nReal-time data for MSFT:\n{msft_rt_data}")

    logger.info("\nFetching latest real-time data for GOOGL (last 2 entries)...")
    googl_rt_data = realtime_db_manager.fetch_realtime_data(ticker='GOOGL', limit=2)
    logger.info(f"\nLatest real-time data for GOOGL:\n{googl_rt_data}")

    logger.info("\nDemonstrating update for real-time data (adjusting last GOOGL bar)...")
    last_googl_bar = googl_rt_data.iloc[-1:].copy()
    if not last_googl_bar.empty:
        last_googl_bar['Close'] = last_googl_bar['Close'] * 1.01
        last_googl_bar['Volume'] = last_googl_bar['Volume'] + 50
        # Restore original column names before passing to upsert_realtime_data
        last_googl_bar.rename(columns={
            'datetime': 'Date', 'symbol': 'Symbol', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'fetch_time': 'fetch_time'
        }, inplace=True)
        realtime_db_manager.upsert_realtime_data(last_googl_bar)
        logger.info("Updated real-time bar for GOOGL.")
        updated_googl_rt_data = realtime_db_manager.fetch_realtime_data(ticker='GOOGL', limit=1)
        logger.info(f"GOOGL's latest bar after update:\n{updated_googl_rt_data}")


if __name__ == "__main__":
    try:
        main_historical_db_test()
        main_realtime_db_test()
        logger.info("All database operations completed successfully!")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise