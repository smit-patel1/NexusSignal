# NexusSignal

Multi-engine financial forecasting and signal system.

## Current Phase: Phase 1 - Data Ingestion

The data ingestion layer is now implemented with historical and incremental price data fetching, plus news ingestion from multiple sources.

## Project Structure

```
nexussignal/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py          # FastAPI application with startup/shutdown events
│       ├── config.py         # Configuration management with Pydantic
│       ├── db.py             # SQLAlchemy async database connector
│       ├── cache.py          # Redis/Valkey async client
│       └── routers/
│           ├── __init__.py
│           └── health.py     # Health check endpoint
├── data/
│   ├── raw/
│   │   ├── prices/          # Parquet files: {TICKER}_{TIMEFRAME}.parquet
│   │   └── news/            # JSONL files: YYYY-MM-DD.jsonl
│   └── processed/
│       └── features/        # Processed features (future)
├── models/
│   ├── numeric/             # Numeric models (future)
│   ├── nlp/                 # NLP models (future)
│   └── meta/                # Meta-learning models (future)
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Data ingestion scripts
│   ├── fetch_tiingo_history.py   # Historical price data fetcher
│   ├── update_tiingo_intraday.py # Incremental intraday updater
│   ├── fetch_news.py             # News ingestion from Finnhub/NewsAPI
│   └── utils/
│       └── parquet_utils.py      # Parquet file utilities
├── dashboard/
│   └── src/                 # Dashboard frontend (future)
├── tests/                   # Test files
├── .env.example             # Environment variables template
├── pyproject.toml           # Project dependencies and metadata
└── README.md                # This file
```

## Setup

### 1. Install Dependencies

Using pip:

```bash
pip install -e .
```

Or for development with testing tools:

```bash
pip install -e ".[dev]"
```

### 2. Configure Environment

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:
- Database URL (PostgreSQL with asyncpg)
- Redis/Valkey URL
- API keys (Tiingo, Finnhub, NewsAPI)
- Environment settings

### 3. Run the Application

Start the FastAPI server:

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:
- Main API: http://localhost:8000
- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

## Configuration

The application uses Pydantic BaseSettings for configuration management. All settings can be configured via environment variables or the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://localhost/nexussignal` |
| `REDIS_URL` | Redis/Valkey connection URL | `redis://localhost:6379/0` |
| `ENVIRONMENT` | Application environment (`dev` or `prod`) | `dev` |
| `LOG_LEVEL` | Logging level | `info` |
| `TIINGO_API_KEY` | Tiingo API key | (empty) |
| `FINNHUB_API_KEY` | Finnhub API key | (empty) |
| `NEWSAPI_API_KEY` | NewsAPI key | (empty) |

## API Endpoints

### Health Check
- `GET /health` - Returns `{"status": "ok"}`

### Root
- `GET /` - Returns API information and version

## Phase 1: Data Ingestion

Phase 1 implements comprehensive data ingestion for historical and real-time price data, plus news from multiple sources.

### Database Tables

**prices_live**: Stores latest price bars for real-time access
- ticker, timestamp, open, high, low, close, volume
- Indexed by (ticker, timestamp)

**news_live**: Stores recent news headlines
- ticker, timestamp, headline, sentiment_score, source, url
- Indexed by (ticker, timestamp)
- URL is unique to prevent duplicates

### Data Ingestion Scripts

#### 1. Historical Price Data Fetcher

Fetches 10+ years of daily data and 3-5 years of 1H intraday data from Tiingo.

```bash
python scripts/fetch_tiingo_history.py
```

**Features:**
- Fetches data for all configured tickers (25 stocks)
- Saves to Parquet: `data/raw/prices/{TICKER}_{TIMEFRAME}.parquet`
- Handles rate limiting and API errors gracefully
- Avoids duplicates via timestamp checking
- Resumable: checks for existing data and fetches only new bars

**Timeframes:**
- `daily`: 10+ years of daily OHLCV
- `1h`: 3-5 years of 1-hour OHLCV

#### 2. Incremental Intraday Updater

Updates intraday data incrementally (run via cron every 30 minutes).

```bash
python scripts/update_tiingo_intraday.py
```

**Features:**
- Loads existing Parquet files
- Fetches only new candles since last timestamp
- Appends to Parquet files
- Writes latest bar to `prices_live` table in Postgres
- Caches last 24 hours in Redis/Valkey: `prices:{ticker}:1h`
- Fully async for concurrent ticker updates

**Cache Format:**
```json
[
  {"ticker": "AAPL", "timestamp": "2024-01-15T14:00:00", "open": 185.5, ...},
  ...
]
```

#### 3. News Ingestion

Fetches news from Finnhub (company news) and NewsAPI (general market news).

```bash
python scripts/fetch_news.py
```

**Features:**
- Fetches company-specific news for each ticker (Finnhub)
- Fetches general market news (NewsAPI)
- Saves raw data to JSONL: `data/raw/news/YYYY-MM-DD.jsonl`
- Inserts recent headlines (last 24h) into `news_live` table
- Deduplicates by URL
- No sentiment analysis yet (Phase 3)

**News Sources:**
- Finnhub: Company-specific news tagged by ticker
- NewsAPI: General market news (S&P 500, NASDAQ, etc.)

### Parquet Utilities

Helper functions in `scripts/utils/parquet_utils.py`:

- `read_parquet()`: Read Parquet files
- `write_parquet()`: Write with compression
- `append_to_parquet()`: Append with deduplication
- `get_latest_timestamp()`: Get max timestamp from file
- `deduplicate_parquet()`: Remove duplicate timestamps
- `get_row_count()`: Count rows without loading

### Running the Full Pipeline

```bash
# 1. Fetch historical data (one-time or when adding new tickers)
python scripts/fetch_tiingo_history.py

# 2. Set up cron job for incremental updates (every 30 minutes)
# */30 * * * * cd /path/to/nexussignal && python scripts/update_tiingo_intraday.py

# 3. Fetch news (run daily or hourly)
python scripts/fetch_news.py
```

### Configured Tickers

The system tracks 25 major stocks configured in [backend/app/config.py](backend/app/config.py):

```python
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "MA",
    "PG", "CVX", "LLY", "HD", "MRK",
    "ABBV", "KO", "PEP", "AVGO", "COST"
]
```

## Phase 2: Feature Engineering

Phase 2 implements a complete feature engineering system for transforming raw price data into ML-ready features.

### Feature Engineering Components

#### 1. Technical Indicators ([scripts/features/indicators.py](scripts/features/indicators.py))

Vectorized technical indicators:
- **Moving Averages**: SMA, EMA (10, 20, 50, 200 periods)
- **RSI**: Relative Strength Index (14-period)
- **MACD**: MACD line, signal line, histogram
- **Bollinger Bands**: Upper, middle, lower bands, width
- **ATR**: Average True Range (14-period)
- **Stochastic Oscillator**: %K and %D
- **Price Features**: Range, change, change percentage
- **Volume Features**: Volume SMA, volume ratio

#### 2. Feature Utilities ([scripts/features/utils.py](scripts/features/utils.py))

Helper functions for feature engineering:
- `ensure_datetime_index()`: Normalize datetime indices
- `add_lagged_returns()`: Past returns (1h, 2h, 4h, 24h)
- `add_forward_returns()`: Future returns (targets for prediction)
- `add_volatility()`: Rolling volatility features
- `add_rolling_stats()`: Rolling mean, std, min, max
- `add_time_features()`: Hour, day of week, month (cyclical encoding)
- `merge_timeframes()`: Merge daily features into hourly data
- `clean_dataframe()`: Remove NaNs, infinities, duplicates
- `zscore()`: Z-score normalization

#### 3. Feature Builder ([scripts/features/build_features.py](scripts/features/build_features.py))

Main feature building pipeline:

```python
def build_features_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Builds complete feature set for a ticker:
    1. Load daily and hourly raw data
    2. Compute technical indicators
    3. Add returns, volatility, rolling stats
    4. Merge multi-timeframe features
    5. Add target variables (forward returns)
    6. Add time-based features
    7. Clean and save to Parquet
    """
```

**Output**: `data/processed/features/{TICKER}_1h.parquet`

#### 4. Auto-Runner ([scripts/features/run_feature_build.py](scripts/features/run_feature_build.py))

Automated feature generation for all tickers:

```bash
python scripts/features/run_feature_build.py
```

**Features:**
- Processes all tickers from `backend/app/config.py`
- Progress tracking with detailed logs
- Error handling for missing data
- Summary statistics per ticker
- Automatic output to `data/processed/features/`

### Feature Set Overview

Each processed dataset contains:

**Technical Indicators** (~25 features):
- Moving averages (SMA/EMA at multiple windows)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)

**Returns & Volatility** (~15 features):
- Lagged returns (1h, 2h, 4h, 24h)
- Rolling volatility (24h, 72h, 168h)
- Realized volatility

**Rolling Statistics** (~20 features):
- Rolling mean, std, min, max
- Percentage from rolling max (drawdown)

**Multi-Timeframe** (~10 features):
- Daily indicators merged into hourly data
- Long-term trends (50-day, 200-day SMAs)

**Time Features** (~10 features):
- Hour, day of week, month, quarter
- Cyclical encodings (sin/cos)

**Targets** (~3 features):
- Forward returns: 1h, 4h, 24h
- Used for supervised learning

**Total**: ~80-100 features per ticker

### Running Feature Engineering

**Prerequisites:**
```bash
# Must have raw price data first
python scripts/fetch_tiingo_history.py
```

**Build features for all tickers:**
```bash
python scripts/features/run_feature_build.py
```

**Build features for a specific ticker:**
```python
from scripts.features.build_features import build_features_for_ticker

df = build_features_for_ticker("AAPL")
```

### Output Structure

```
data/processed/features/
├── AAPL_1h.parquet
├── MSFT_1h.parquet
├── NVDA_1h.parquet
└── ... (all configured tickers)
```

Each file contains:
- Hourly granularity
- ~80-100 engineered features
- Clean data (no NaNs, infinities)
- Datetime index
- Target variables for prediction

## Development Status

**Current Phase:** Phase 2 - Feature Engineering ✅

### Phase 0 - Completed
- Project directory structure
- FastAPI application with startup/shutdown events
- Configuration system with Pydantic
- Async database connector (SQLAlchemy)
- Async cache connector (Redis/Valkey)
- Health check endpoint
- Environment configuration template

### Phase 1 - Completed
- SQLAlchemy models for `prices_live` and `news_live`
- Parquet utility functions for data management
- Historical price data fetcher (10+ years daily, 3-5 years 1H)
- Incremental intraday updater with Postgres and Redis caching
- News ingestion from Finnhub and NewsAPI
- JSONL storage for raw news data
- Deduplication and rate limiting

### Phase 2 - Completed
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic)
- Feature utilities (lagged returns, volatility, rolling stats, time features)
- Multi-timeframe feature merging (daily + hourly)
- Target variable generation (forward returns)
- Automated feature builder for all tickers
- Clean, normalized, ML-ready datasets

### Future Phases
- **Phase 3**: Sentiment analysis on news data
- **Phase 4**: ML model training (numeric, NLP, meta-learning)
- **Phase 5**: Signal generation and backtesting
- **Phase 6**: Dashboard frontend
- **Phase 7**: Advanced API endpoints and real-time streaming

## Technology Stack

- **Backend Framework:** FastAPI
- **Database:** PostgreSQL with asyncpg (SQLAlchemy ORM)
- **Cache:** Redis/Valkey
- **Configuration:** Pydantic Settings
- **ASGI Server:** Uvicorn
- **Data Storage:** Parquet (via pandas + pyarrow)
- **HTTP Client:** httpx (async)
- **Data APIs:** Tiingo (prices), Finnhub (news), NewsAPI (news)

## License

TBD
