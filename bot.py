import os
import json
import time
import math
import requests
import warnings
from io import StringIO
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# ABD BORSA PRO V9.4
# PRE-BREAKOUT HUNTER
# ============================================================

VERSION = "V9.4 PRE-BREAKOUT HUNTER"

MIN_PRICE = 0.10
MAX_PRICE = 15.00

MIN_EARLY_SCORE = 62
MIN_BREAKOUT_SCORE = 80

MAX_UNIVERSE = 3000
MAX_DEEP_CANDIDATES = 180

MAX_SIGNALS_PER_SCAN = 3

EARLY_COOLDOWN_MIN = 180
BREAKOUT_COOLDOWN_MIN = 30

SIGNALS_FILE = "signals.json"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

NASDAQ_LIST_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/nasdaqlisted.txt"
)

OTHER_LIST_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/otherlisted.txt"
)

# ============================================================
# LOG
# ============================================================

def log(msg):
    print(
        f"[{VERSION}] {msg}",
        flush=True
    )


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value):
            return default

        return value

    except Exception:
        return default


def clamp(value, low, high):
    try:
        return max(
            low,
            min(high, value)
        )
    except Exception:
        return low


# ============================================================
# HISTORY
# ============================================================

def load_history():

    if not os.path.exists(
        SIGNALS_FILE
    ):
        return {}

    try:
        with open(
            SIGNALS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as e:
        log(
            f"History okunamadı: {e}"
        )

    return {}


def save_history(history):

    try:
        with open(
            SIGNALS_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                history,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:
        log(
            f"History yazılamadı: {e}"
        )


def can_send(
    history,
    symbol,
    signal_type
):

    key = (
        f"{symbol}:{signal_type}"
    )

    item = history.get(key)

    if not item:
        return True

    try:
        last_time = float(
            item.get(
                "time",
                0
            )
        )
    except Exception:
        return True

    age = (
        time.time()
        - last_time
    ) / 60

    if signal_type == "BREAKOUT":
        cooldown = (
            BREAKOUT_COOLDOWN_MIN
        )
    else:
        cooldown = (
            EARLY_COOLDOWN_MIN
        )

    return age >= cooldown


def mark_sent(
    history,
    symbol,
    signal_type
):

    key = (
        f"{symbol}:{signal_type}"
    )

    history[key] = {
        "time": time.time(),
        "symbol": symbol,
        "type": signal_type
    }


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    if not BOT_TOKEN:
        log(
            "BOT_TOKEN eksik."
        )
        return False

    if not CHAT_ID:
        log(
            "CHAT_ID eksik."
        )
        return False

    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "disable_web_page_preview": True
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        log(
            f"[Telegram] HTTP "
            f"{response.status_code}"
        )

        if response.status_code != 200:
            log(
                response.text[:500]
            )
            return False

        return True

    except Exception as e:

        log(
            f"Telegram hatası: {e}"
        )

        return False


# ============================================================
# MARKET STATE
# ============================================================

def market_state():

    try:

        now = datetime.now(
            timezone.utc
        )

        minutes = (
            now.hour * 60
            + now.minute
        )

        pre_start = (
            8 * 60
        )

        market_open = (
            13 * 60 + 30
        )

        market_close = (
            20 * 60
        )

        if (
            pre_start
            <= minutes
            < market_open
        ):
            return "PREMARKET"

        if (
            market_open
            <= minutes
            < market_close
        ):
            return "OPEN"

        if (
            market_close
            <= minutes
        ):
            return "AFTER_HOURS"

        return "CLOSED"

    except Exception:

        return "UNKNOWN"


# ============================================================
# SYMBOL CLEANING
# ============================================================

def clean_symbol(symbol):

    if not symbol:
        return None

    symbol = (
        str(symbol)
        .strip()
        .upper()
    )

    if symbol in (
        "",
        "N/A",
        "NONE"
    ):
        return None

    # Yahoo'da sorun çıkaran sınıfları
    # mümkün olduğunca ele.
    if any(
        x in symbol
        for x in [
            ".A",
            ".B",
            ".V",
            "^",
            "=",
            "$",
            " "
        ]
    ):
        return None

    return symbol


# ============================================================
# OFFICIAL SYMBOL LIST
# ============================================================

def download_text(url):

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            },
            timeout=20
        )

        if response.status_code != 200:
            log(
                f"Sembol listesi HTTP "
                f"{response.status_code}"
            )
            return ""

        return response.text

    except Exception as e:

        log(
            f"Sembol listesi hatası: {e}"
        )

        return ""


def get_nasdaq_symbols():

    text = download_text(
        NASDAQ_LIST_URL
    )

    if not text:
        return []

    try:

        df = pd.read_csv(
            StringIO(text),
            sep="|",
            dtype=str
        )

        df.columns = [
            str(x).strip()
            for x in df.columns
        ]

        if "Symbol" not in df.columns:
            return []

        df = df[
            df["Symbol"]
            != "[File Creation Time]"
        ]

        if "Test Issue" in df.columns:

            df = df[
                df["Test Issue"]
                .fillna("N")
                == "N"
            ]

        if "ETF" in df.columns:

            df = df[
                df["ETF"]
                .fillna("N")
                == "N"
            ]

        result = []

        for symbol in df[
            "Symbol"
        ].tolist():

            symbol = clean_symbol(
                symbol
            )

            if symbol:
                result.append(
                    symbol
                )

        return result

    except Exception as e:

        log(
            f"NASDAQ parse hatası: {e}"
        )

        return []


def get_other_symbols():

    text = download_text(
        OTHER_LIST_URL
    )

    if not text:
        return []

    try:

        df = pd.read_csv(
            StringIO(text),
            sep="|",
            dtype=str
        )

        df.columns = [
            str(x).strip()
            for x in df.columns
        ]

        if "ACT Symbol" not in df.columns:
            return []

        df = df[
            df["ACT Symbol"]
            != "[File Creation Time]"
        ]

        if "Test Issue" in df.columns:

            df = df[
                df["Test Issue"]
                .fillna("N")
                == "N"
            ]

        if "ETF" in df.columns:

            df = df[
                df["ETF"]
                .fillna("N")
                == "N"
            ]

        if "Security Name" in df.columns:

            names = (
                df["Security Name"]
                .fillna("")
                .str.upper()
            )

            excluded = (
                "WARRANT",
                "RIGHT",
                "UNIT",
                "PREFERRED",
                "DEBENTURE",
                "BOND",
                "NOTE",
                "ETF",
                "FUND"
            )

            mask = ~names.apply(
                lambda x:
                any(
                    word in x
                    for word in excluded
                )
            )

            df = df[mask]

        result = []

        for symbol in df[
            "ACT Symbol"
        ].tolist():

            symbol = clean_symbol(
                symbol
            )

            if symbol:
                result.append(
                    symbol
                )

        return result

    except Exception as e:

        log(
            f"Other exchange parse "
            f"hatası: {e}"
        )

        return []


# ============================================================
# YAHOO SCREENER
# ============================================================

YAHOO_SCREENERS = [
    "most_actives",
    "day_gainers",
    "day_losers",
    "small_cap_gainers",
    "aggressive_small_caps",
    "growth_technology_stocks",
    "undervalued_growth_stocks"
]


def get_yahoo_symbols():

    result = set()

    for screen in YAHOO_SCREENERS:

        try:

            data = yf.screen(
                screen,
                count=250
            )

            quotes = []

            if isinstance(
                data,
                dict
            ):
                quotes = data.get(
                    "quotes",
                    []
                )

            count = 0

            for item in quotes:

                symbol = clean_symbol(
                    item.get(
                        "symbol"
                    )
                )

                if symbol:

                    result.add(
                        symbol
                    )

                    count += 1

            log(
                f"{screen}: {count}"
            )

        except Exception as e:

            log(
                f"Screener {screen} "
                f"hatası: {e}"
            )

    return result


# ============================================================
# FALLBACK
# ============================================================

FALLBACK_SYMBOLS = [
    "AEMD",
    "WHLR",
    "TNON",
    "EROC",
    "VACI",
    "LUXE",
    "CPIX",
    "ATRA",
    "PLCE",
    "EAF",
    "EMPD",
    "BRCC",
    "UEIC",
    "IIIV",
    "AIFU",
    "GLMD",
    "SPWR",
    "SURG",
    "CRDF",
    "FRTT",
    "CEPU",
    "SUZ",
    "SKDD",
    "SBGI",
    "SHLS",
    "SES",
    "NXDR",
    "ABR",
    "CLF",
    "HIVE",
    "UWMC",
    "LI",
    "RXRX",
    "CHPT",
    "BSBR",
    "ATAI"
]


# ============================================================
# UNIVERSE
# ============================================================

def build_universe():

    log(
        "PRE-BREAKOUT geniş evren "
        "oluşturuluyor..."
    )

    nasdaq = get_nasdaq_symbols()

    other = get_other_symbols()

    yahoo = get_yahoo_symbols()

    official = set(
        nasdaq + other
    )

    log(
        f"NASDAQ sembolü: "
        f"{len(nasdaq)}"
    )

    log(
        f"NYSE/Other sembolü: "
        f"{len(other)}"
    )

    log(
        f"Resmi toplam: "
        f"{len(official)}"
    )

    log(
        f"Yahoo ek sembolü: "
        f"{len(yahoo)}"
    )

    official_list = sorted(
        official
    )

    # Her 5 dakikada başka dilim.
    if len(official_list) > MAX_UNIVERSE:

        slot = int(
            time.time() // 300
        )

        start = (
            slot * MAX_UNIVERSE
        ) % len(official_list)

        end = (
            start + MAX_UNIVERSE
        )

        if end <= len(
            official_list
        ):

            rotating = (
                official_list[
                    start:end
                ]
            )

        else:

            rotating = (
                official_list[start:]
                +
                official_list[
                    :end
                    - len(official_list)
                ]
            )

        log(
            f"Rotasyon: {start} "
            f"-> {len(rotating)}"
        )

    else:

        rotating = official_list

    universe = set(
        rotating
    )

    universe.update(
        yahoo
    )

    universe.update(
        FALLBACK_SYMBOLS
    )

    # Son temizlik
    universe = [
        clean_symbol(x)
        for x in universe
    ]

    universe = [
        x for x in universe
        if x
    ]

    universe = list(
        dict.fromkeys(
            universe
        )
    )

    log(
        f"Toplam taranacak evren: "
        f"{len(universe)}"
    )

    return universe


# ============================================================
# BATCH DATA
# ============================================================

def extract_symbol_frame(
    data,
    symbol
):

    try:

        if (
            data is None
            or data.empty
        ):
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = list(
                data.columns
                .get_level_values(0)
            )

            level1 = list(
                data.columns
                .get_level_values(1)
            )

            if symbol in level0:

                return data[
                    symbol
                ].copy()

            if symbol in level1:

                return data.xs(
                    symbol,
                    axis=1,
                    level=1
                ).copy()

        return data.copy()

    except Exception:

        return None


# ============================================================
# DAILY PRE-FILTER
# ============================================================

def daily_prefilter(
    universe
):

    log(
        "Toplu günlük ön tarama..."
    )

    results = []

    symbols = list(
        dict.fromkeys(
            universe
        )
    )

    chunk_size = 150

    for start in range(
        0,
        len(symbols),
        chunk_size
    ):

        chunk = symbols[
            start:
            start + chunk_size
        ]

        try:

            data = yf.download(
                chunk,
                period="1mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True
            )

        except Exception as e:

            log(
                f"Batch hata: {e}"
            )

            continue

        for symbol in chunk:

            try:

                df = extract_symbol_frame(
                    data,
                    symbol
                )

                if (
                    df is None
                    or df.empty
                ):
                    continue

                if (
                    "Close" not in df.columns
                    or
                    "Volume" not in df.columns
                ):
                    continue

                df = df.dropna(
                    subset=["Close"]
                )

                if df.empty:
                    continue

                close = (
                    df["Close"]
                    .astype(float)
                )

                price = safe_float(
                    close.iloc[-1]
                )

                if not (
                    MIN_PRICE
                    <= price
                    <= MAX_PRICE
                ):
                    continue

                volume = (
                    df["Volume"]
                    .fillna(0)
                    .astype(float)
                )

                recent_volume = (
                    safe_float(
                        volume.iloc[-1],
                        0
                    )
                )

                avg_volume = (
                    safe_float(
                        volume.iloc[:-1]
                        .tail(10)
                        .mean(),
                        0
                    )
                )

                if avg_volume > 0:

                    volume_ratio = (
                        recent_volume
                        / avg_volume
                    )

                else:

                    volume_ratio = 0

                if len(close) >= 2:

                    day_change = (
                        close.iloc[-1]
                        /
                        close.iloc[-2]
                        - 1
                    ) * 100

                else:

                    day_change = 0

                # Önceki direnç
                previous = df.iloc[:-1]

                if previous.empty:
                    previous = df

                highs = (
                    previous["High"]
                    .astype(float)
                )

                if len(highs) >= 20:

                    high20 = safe_float(
                        highs.tail(20).max()
                    )

                else:

                    high20 = safe_float(
                        highs.max()
                    )

                if (
                    np.isfinite(high20)
                    and high20 > 0
                ):

                    distance = (
                        (
                            high20
                            - price
                        )
                        / high20
                    ) * 100

                else:

                    distance = 999

                # PRE-BREAKOUT ön puanı
                rank = 0

                # Hacim
                rank += clamp(
                    volume_ratio * 10,
                    0,
                    30
                )

                # Günlük momentum
                rank += clamp(
                    max(
                        day_change,
                        0
                    ) * 2.5,
                    0,
                    20
                )

                # Dirence yakınlık
                if 0 <= distance <= 15:

                    rank += (
                        15
                        - distance
                    )

                # Düşük fiyat
                if price <= 5:
                    rank += 8

                elif price <= 10:
                    rank += 5

                # Küçük fiyat hareketi
                # zaten hareket etmiş hisseleri
                # aşırı öne çıkarmamak için
                if day_change > 30:
                    rank -= 10

                results.append({
                    "symbol": symbol,
                    "price": price,
                    "rank": rank
                })

            except Exception:
                continue

        processed = min(
            start + chunk_size,
            len(symbols)
        )

        log(
            f"Günlük tarama: "
            f"{processed}/"
            f"{len(symbols)}"
        )

    results.sort(
        key=lambda x:
        x["rank"],
        reverse=True
    )

    log(
        f"Günlük ön tarama sonucu: "
        f"{len(results)} uygun"
    )

    selected = results[
        :MAX_DEEP_CANDIDATES
    ]

    log(
        f"Derin analiz: "
        f"{len(selected)} hisse"
    )

    return selected


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14
):

    try:

        delta = series.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.rolling(
            period
        ).mean()

        avg_loss = loss.rolling(
            period
        ).mean()

        rs = (
            avg_gain
            /
            avg_loss.replace(
                0,
                np.nan
            )
        )

        return (
            100
            -
            (
                100
                /
                (1 + rs)
            )
        )

    except Exception:

        return pd.Series(
            index=series.index,
            dtype=float
        )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    try:

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low

        tr2 = (
            high
            - close.shift(1)
        ).abs()

        tr3 = (
            low
            - close.shift(1)
        ).abs()

        tr = pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        ).max(axis=1)

        return tr.rolling(
            period
        ).mean()

    except Exception:

        return pd.Series(
            index=df.index,
            dtype=float
        )


# ============================================================
# VWAP
# ============================================================

def calculate_vwap(df):

    try:

        typical = (
            df["High"]
            +
            df["Low"]
            +
            df["Close"]
        ) / 3

        volume = (
            df["Volume"]
            .fillna(0)
        )

        cumulative_value = (
            typical * volume
        ).cumsum()

        cumulative_volume = (
            volume.cumsum()
        )

        return (
            cumulative_value
            /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

    except Exception:

        return pd.Series(
            index=df.index,
            dtype=float
        )


# ============================================================
# PREMARKET RVOL
# ============================================================

def premarket_rvol(df):

    try:

        if df.empty:
            return 0

        temp = df.copy()

        if temp.index.tz is not None:

            temp.index = (
                temp.index.tz_convert(
                    "America/New_York"
                )
            )

        pre = temp.between_time(
            "04:00",
            "09:29"
        )

        if pre.empty:
            return 0

        pre["DATE"] = (
            pre.index.date
        )

        dates = sorted(
            pre["DATE"].unique()
        )

        if not dates:
            return 0

        today = dates[-1]

        today_volume = safe_float(
            pre[
                pre["DATE"] == today
            ]["Volume"].sum(),
            0
        )

        previous = []

        for date in dates[:-1][-5:]:

            volume = safe_float(
                pre[
                    pre["DATE"] == date
                ]["Volume"].sum(),
                0
            )

            if volume > 0:
                previous.append(
                    volume
                )

        if not previous:
            return 0

        average = np.mean(
            previous
        )

        if average <= 0:
            return 0

        return (
            today_volume
            / average
        )

    except Exception:

        return 0


# ============================================================
# FLOAT / CATALYST
# ============================================================

def company_info(ticker):

    try:

        info = ticker.info

        market_cap = safe_float(
            info.get(
                "marketCap"
            )
        )

        float_shares = safe_float(
            info.get(
                "floatShares"
            )
        )

        shares = safe_float(
            info.get(
                "sharesOutstanding"
            )
        )

        if not np.isfinite(
            float_shares
        ):

            float_shares = shares

        return (
            market_cap,
            float_shares
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


def catalyst_exists(ticker):

    try:

        news = ticker.news

        return bool(
            news
        )

    except Exception:

        return False


# ============================================================
# DEEP ANALYSIS
# ============================================================

def analyze_symbol(symbol):

    try:

        ticker = yf.Ticker(
            symbol
        )

        intraday = ticker.history(
            period="5d",
            interval="5m",
            prepost=True,
            auto_adjust=False
        )

        if (
            intraday is None
            or intraday.empty
        ):
            return None

        daily = ticker.history(
            period="3mo",
            interval="1d",
            auto_adjust=False
        )

        if (
            daily is None
            or daily.empty
        ):
            return None

        intraday = intraday.dropna(
            subset=["Close"]
        )

        daily = daily.dropna(
            subset=["Close"]
        )

        if len(intraday) < 20:
            return None

        close = (
            intraday["Close"]
            .astype(float)
        )

        volume = (
            intraday["Volume"]
            .fillna(0)
            .astype(float)
        )

        price = safe_float(
            close.iloc[-1]
        )

        if not (
            MIN_PRICE
            <= price
            <= MAX_PRICE
        ):
            return None

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        ema9 = close.ewm(
            span=9,
            adjust=False
        ).mean()

        ema20 = close.ewm(
            span=20,
            adjust=False
        ).mean()

        ema9_last = safe_float(
            ema9.iloc[-1]
        )

        ema20_last = safe_float(
            ema20.iloc[-1]
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi = safe_float(
            calculate_rsi(
                close
            ).iloc[-1]
        )

        # ----------------------------------------------------
        # VWAP
        # ----------------------------------------------------

        vwap = safe_float(
            calculate_vwap(
                intraday
            ).iloc[-1]
        )

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = safe_float(
            calculate_atr(
                intraday
            ).iloc[-1]
        )

        if (
            not np.isfinite(atr)
            or atr <= 0
        ):
            atr = price * 0.03

        # ----------------------------------------------------
        # 5M MOMENTUM
        # ----------------------------------------------------

        if len(close) >= 4:

            momentum_5m = (
                close.iloc[-1]
                /
                close.iloc[-4]
                - 1
            ) * 100

        else:

            momentum_5m = 0

        # ----------------------------------------------------
        # 15M MOMENTUM
        # ----------------------------------------------------

        if len(close) >= 8:

            momentum_15m = (
                close.iloc[-1]
                /
                close.iloc[-8]
                - 1
            ) * 100

        else:

            momentum_15m = 0

        # ----------------------------------------------------
        # 5M VOLUME
        # ----------------------------------------------------

        last_volume = safe_float(
            volume.iloc[-1],
            0
        )

        average_volume = safe_float(
            volume.iloc[:-1]
            .tail(30)
            .mean(),
            0
        )

        if average_volume > 0:

            volume_ratio = (
                last_volume
                /
                average_volume
            )

        else:

            volume_ratio = 0

        # ----------------------------------------------------
        # TODAY VOLUME
        # ----------------------------------------------------

        try:

            idx = intraday.index

            if idx.tz is not None:

                local_index = (
                    idx.tz_convert(
                        "America/New_York"
                    )
                )

            else:

                local_index = idx

            mask = (
                local_index.date
                ==
                local_index[-1].date()
            )

            today_volume = safe_float(
                volume[mask].sum(),
                0
            )

        except Exception:

            today_volume = safe_float(
                volume.tail(78).sum(),
                0
            )

        # ----------------------------------------------------
        # DAILY RVOL
        # ----------------------------------------------------

        daily_volume = (
            daily["Volume"]
            .fillna(0)
            .astype(float)
        )

        average_daily_volume = safe_float(
            daily_volume.iloc[:-1]
            .tail(20)
            .mean(),
            0
        )

        if average_daily_volume > 0:

            rvol = (
                today_volume
                /
                average_daily_volume
            )

        else:

            rvol = 0

        # ----------------------------------------------------
        # PREMARKET RVOL
        # ----------------------------------------------------

        pm_rvol = (
            premarket_rvol(
                intraday
            )
        )

        # ----------------------------------------------------
        # MULTI-DAY RESISTANCE
        # ----------------------------------------------------

        previous_daily = (
            daily.iloc[:-1]
        )

        if previous_daily.empty:
            previous_daily = daily

        resistance_candidates = []

        for period in [
            3,
            5,
            10,
            20,
            30
        ]:

            if len(
                previous_daily
            ) >= period:

                resistance_candidates.append(
                    safe_float(
                        previous_daily[
                            "High"
                        ]
                        .astype(float)
                        .tail(period)
                        .max()
                    )
                )

        # Intraday resistance
        previous_intraday = (
            intraday.iloc[:-1]
        )

        if not previous_intraday.empty:

            intraday_resistance = safe_float(
                previous_intraday[
                    "High"
                ]
                .astype(float)
                .tail(78)
                .max()
            )

            if np.isfinite(
                intraday_resistance
            ):
                resistance_candidates.append(
                    intraday_resistance
                )

        resistance_candidates = [
            x for x
            in resistance_candidates
            if np.isfinite(x)
            and x > 0
        ]

        if not resistance_candidates:
            return None

        # En yakın üst direnç
        above_price = [
            x for x
            in resistance_candidates
            if x > price
        ]

        if above_price:

            resistance = min(
                above_price
            )

        else:

            # Yeni zirve yapmışsa
            # son geçmiş direnç
            resistance = max(
                resistance_candidates
            )

        # ----------------------------------------------------
        # RESISTANCE DISTANCE
        # ----------------------------------------------------

        if resistance > 0:

            resistance_distance = (
                (
                    resistance
                    - price
                )
                /
                resistance
            ) * 100

        else:

            resistance_distance = 999

        # ----------------------------------------------------
        # GAP
        # ----------------------------------------------------

        gap = 0

        if len(daily) >= 2:

            previous_close = safe_float(
                daily["Close"]
                .astype(float)
                .iloc[-2]
            )

            if (
                np.isfinite(
                    previous_close
                )
                and previous_close > 0
            ):

                gap = (
                    (
                        price
                        /
                        previous_close
                    )
                    - 1
                ) * 100

        # ----------------------------------------------------
        # FLOAT
        # ----------------------------------------------------

        market_cap, float_shares = (
            company_info(
                ticker
            )
        )

        if np.isfinite(
            float_shares
        ):

            float_m = (
                float_shares
                / 1_000_000
            )

        else:

            float_m = np.nan

        # ----------------------------------------------------
        # CATALYST
        # ----------------------------------------------------

        catalyst = (
            catalyst_exists(
                ticker
            )
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = 0

        factors = []

        # LOW FLOAT
        if np.isfinite(
            float_m
        ):

            if float_m <= 5:

                score += 15
                factors.append(
                    "LOW FLOAT"
                )

            elif float_m <= 10:

                score += 12
                factors.append(
                    "LOW FLOAT"
                )

            elif float_m <= 20:

                score += 6
                factors.append(
                    "FLOAT"
                )

        # GAP
        if gap >= 15:

            score += 15
            factors.append(
                "GAP"
            )

        elif gap >= 8:

            score += 12
            factors.append(
                "GAP"
            )

        elif gap >= 4:

            score += 9
            factors.append(
                "GAP"
            )

        elif gap >= 2:

            score += 5

        # PREMARKET RVOL
        if pm_rvol >= 8:

            score += 15
            factors.append(
                "PREMARKET RVOL"
            )

        elif pm_rvol >= 5:

            score += 12
            factors.append(
                "PREMARKET RVOL"
            )

        elif pm_rvol >= 3:

            score += 9
            factors.append(
                "PREMARKET HACİM"
            )

        elif pm_rvol >= 2:

            score += 5

        # NORMAL RVOL
        if rvol >= 8:

            score += 15
            factors.append(
                "RVOL"
            )

        elif rvol >= 5:

            score += 12
            factors.append(
                "RVOL"
            )

        elif rvol >= 3:

            score += 9
            factors.append(
                "RVOL"
            )

        elif rvol >= 2:

            score += 5

        # CATALYST
        if catalyst:

            score += 10
            factors.append(
                "CATALYST"
            )

        # VWAP
        if (
            np.isfinite(vwap)
            and price > vwap
        ):

            score += 8
            factors.append(
                "VWAP"
            )

        # 5M MOMENTUM
        if momentum_5m >= 5:

            score += 10
            factors.append(
                "5M MOMENTUM"
            )

        elif momentum_5m >= 3:

            score += 8
            factors.append(
                "5M MOMENTUM"
            )

        elif momentum_5m >= 1.5:

            score += 5

        # 15M MOMENTUM
        if momentum_15m >= 3:

            score += 5
            factors.append(
                "15M MOMENTUM"
            )

        # EMA
        if (
            np.isfinite(ema9_last)
            and
            np.isfinite(ema20_last)
            and
            ema9_last > ema20_last
        ):

            score += 5
            factors.append(
                "EMA"
            )

        # RSI
        if np.isfinite(rsi):

            if (
                50
                <= rsi
                <= 72
            ):

                score += 5

            elif rsi > 80:

                score -= 8

        # ----------------------------------------------------
        # PRE-BREAKOUT MESAFE
        # ----------------------------------------------------

        if (
            0
            <= resistance_distance
            <= 1.5
        ):

            score += 10
            factors.append(
                "KIRILIMA ÇOK YAKIN"
            )

        elif (
            1.5
            < resistance_distance
            <= 3
        ):

            score += 9
            factors.append(
                "KIRILIMA YAKIN"
            )

        elif (
            3
            < resistance_distance
            <= 5
        ):

            score += 7
            factors.append(
                "DİRENÇ YAKIN"
            )

        elif (
            5
            < resistance_distance
            <= 8
        ):

            score += 4

        # ----------------------------------------------------
        # AŞIRI YÜKSELENİ CEZALANDIR
        # ----------------------------------------------------

        if gap > 30:
            score -= 8

        if momentum_5m > 15:
            score -= 8

        score = int(
            clamp(
                score,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

        state = market_state()

        breakout_percent = 0

        if resistance > 0:

            breakout_percent = (
                (
                    price
                    - resistance
                )
                /
                resistance
            ) * 100

        real_breakout = False

        if state == "OPEN":

            price_break = (
                price
                > resistance * 1.005
            )

            candle_break = (
                safe_float(
                    close.iloc[-1]
                )
                > resistance
            )

            volume_confirmation = (
                volume_ratio >= 1.5
                or rvol >= 2
            )

            momentum_confirmation = (
                momentum_5m > 0
            )

            rsi_valid = (
                np.isfinite(rsi)
            )

            if (
                price_break
                and candle_break
                and volume_confirmation
                and momentum_confirmation
                and rsi_valid
                and score
                >= MIN_BREAKOUT_SCORE
            ):

                real_breakout = True

        # ----------------------------------------------------
        # EARLY
        # ----------------------------------------------------

        wanted_factors = {
            "LOW FLOAT",
            "GAP",
            "PREMARKET RVOL",
            "PREMARKET HACİM",
            "RVOL",
            "CATALYST",
            "VWAP",
            "5M MOMENTUM",
            "15M MOMENTUM",
            "KIRILIMA ÇOK YAKIN",
            "KIRILIMA YAKIN",
            "DİRENÇ YAKIN"
        }

        strong_count = len(
            set(factors)
            & wanted_factors
        )

        early = (
            score >= MIN_EARLY_SCORE
            and strong_count >= 3
            and resistance_distance <= 10
        )

        if real_breakout:

            signal_type = "BREAKOUT"

        elif early:

            signal_type = "EARLY"

        else:

            return None

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        entry = price

        if signal_type == "BREAKOUT":

            risk_distance = max(
                atr * 1.5,
                price * 0.04
            )

        else:

            risk_distance = max(
                atr * 1.5,
                price * 0.05
            )

            risk_distance = min(
                risk_distance,
                price * 0.10
            )

        stop = (
            entry
            - risk_distance
        )

        if stop <= 0:

            stop = (
                entry * 0.90
            )

        risk = (
            entry - stop
        )

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        target1 = (
            entry
            + risk * 1.5
        )

        target2 = (
            entry
            + risk * 2.5
        )

        target3 = (
            entry
            + risk * 4.0
        )

        return {
            "symbol": symbol,
            "price": price,
            "score": score,
            "signal_type": signal_type,

            "gap": gap,

            "rvol": rvol,
            "pm_rvol": pm_rvol,

            "volume_ratio":
                volume_ratio,

            "momentum_5m":
                momentum_5m,

            "momentum_15m":
                momentum_15m,

            "rsi": rsi,

            "vwap": vwap,

            "float": float_m,

            "resistance":
                resistance,

            "resistance_distance":
                resistance_distance,

            "breakout_percent":
                breakout_percent,

            "stop": stop,

            "target1":
                target1,

            "target2":
                target2,

            "target3":
                target3,

            "factors":
                factors
        }

    except Exception as e:

        log(
            f"{symbol} analiz hatası: "
            f"{e}"
        )

        return None


# ============================================================
# PARALLEL DEEP ANALYSIS
# ============================================================

def deep_analysis(
    candidates
):

    results = []

    symbols = [
        item["symbol"]
        for item in candidates
    ]

    log(
        f"Derin analiz başlıyor: "
        f"{len(symbols)} hisse"
    )

    with ThreadPoolExecutor(
        max_workers=12
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:

                result = (
                    future.result()
                )

                if result:
                    results.append(
                        result
                    )

            except Exception:
                continue

    results.sort(
        key=lambda x: (
            x["score"],
            x["rvol"],
            x["volume_ratio"],
            -x[
                "resistance_distance"
            ]
        ),
        reverse=True
    )

    log(
        f"Uygun sinyal: "
        f"{len(results)}"
    )

    return results


# ============================================================
# FORMAT
# ============================================================

def price_text(value):

    try:
        return (
            f"${float(value):.2f}"
        )
    except Exception:
        return "$-"


def float_text(value):

    if not np.isfinite(value):
        return "N/A"

    if value >= 100:
        return (
            f"{value:.0f}M"
        )

    return (
        f"{value:.1f}M"
    )


# ============================================================
# EARLY MESSAGE
# ============================================================

def early_message(s):

    return (
        f"🚨 #{s['symbol']}\n"
        f"🔥 ERKEN HAREKET ADAYI\n"
        f"━━━━━━━━━━━━━━\n"

        f"💰 Fiyat: "
        f"{price_text(s['price'])}\n\n"

        f"📊 Kırılım: "
        f"{price_text(s['resistance'])}\n"

        f"📏 Dirence: "
        f"%{s['resistance_distance']:.2f}\n\n"

        f"🎯 Hedef 1: "
        f"{price_text(s['target1'])}\n"

        f"🎯 Hedef 2: "
        f"{price_text(s['target2'])}\n"

        f"🎯 Hedef 3: "
        f"{price_text(s['target3'])}\n"

        f"🛑 Stop: "
        f"{price_text(s['stop'])}\n\n"

        f"━━━━━━━━━━━━━━\n"

        f"⭐ Skor: "
        f"{s['score']}/100\n"

        f"🔥 Hacim: "
        f"{s['volume_ratio']:.1f}x\n"

        f"📊 RVOL: "
        f"{s['rvol']:.1f}x\n"

        f"🔥 PM RVOL: "
        f"{s['pm_rvol']:.1f}x\n"

        f"📈 5M Momentum: "
        f"{s['momentum_5m']:+.2f}%\n"

        f"📊 Float: "
        f"{float_text(s['float'])}\n"

        f"📈 Gap: "
        f"{s['gap']:+.2f}%\n\n"

        f"⚡ Durum: "
        f"KIRILIMA YAKIN\n\n"

        f"💵 borsaanaliz.brs"
    )


# ============================================================
# BREAKOUT MESSAGE
# ============================================================

def breakout_message(s):

    return (
        f"🚨 #{s['symbol']}\n"
        f"🔥🔥 GERÇEK KIRILIM ONAYLANDI\n"
        f"━━━━━━━━━━━━━━\n"

        f"📊 Kırılım: "
        f"{price_text(s['resistance'])}\n"

        f"💰 Anlık: "
        f"{price_text(s['price'])}\n\n"

        f"🎯 Hedef 1: "
        f"{price_text(s['target1'])} ⏳\n"

        f"🎯 Hedef 2: "
        f"{price_text(s['target2'])} ⏳\n"

        f"🎯 Hedef 3: "
        f"{price_text(s['target3'])} ⏳\n"

        f"🛑 Stop: "
        f"{price_text(s['stop'])}\n\n"

        f"━━━━━━━━━━━━━━\n"

        f"📈 Kırılım: "
        f"{s['breakout_percent']:+.2f}%\n"

        f"🔥 Hacim: "
        f"{s['volume_ratio']:.1f}x\n"

        f"📊 RVOL: "
        f"{s['rvol']:.1f}x\n"

        f"⭐ Skor: "
        f"{s['score']}/100\n"

        f"✅ BREAKOUT ONAYLI\n\n"

        f"💵 borsaanaliz.brs"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================================="
    )

    print(
        f"[{VERSION}]"
    )

    print(
        "======================================================="
    )

    if not BOT_TOKEN:
        log(
            "BOT_TOKEN bulunamadı."
        )
        return

    if not CHAT_ID:
        log(
            "CHAT_ID bulunamadı."
        )
        return

    state = market_state()

    log(
        f"Piyasa: {state}"
    )

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    universe = build_universe()

    if not universe:

        log(
            "Evren oluşturulamadı."
        )

        return

    # --------------------------------------------------------
    # DAILY PREFILTER
    # --------------------------------------------------------

    candidates = (
        daily_prefilter(
            universe
        )
    )

    if not candidates:

        log(
            "Aday bulunamadı."
        )

        return

    # --------------------------------------------------------
    # DEEP ANALYSIS
    # --------------------------------------------------------

    signals = (
        deep_analysis(
            candidates
        )
    )

    if not signals:

        log(
            "Uygun sinyal yok."
        )

        return

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = load_history()

    sent = 0

    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    for signal in signals:

        if (
            sent
            >= MAX_SIGNALS_PER_SCAN
        ):
            break

        symbol = signal[
            "symbol"
        ]

        signal_type = signal[
            "signal_type"
        ]

        if not can_send(
            history,
            symbol,
            signal_type
        ):

            log(
                f"Cooldown: "
                f"#{symbol} "
                f"{signal_type}"
            )

            continue

        if signal_type == "BREAKOUT":

            message = (
                breakout_message(
                    signal
                )
            )

        else:

            message = (
                early_message(
                    signal
                )
            )

        log(
            f"[SİNYAL] "
            f"#{symbol} "
            f"{signal_type} "
            f"Score="
            f"{signal['score']}"
        )

        if telegram_send(
            message
        ):

            mark_sent(
                history,
                symbol,
                signal_type
            )

            sent += 1

    save_history(
        history
    )

    log(
        f"Gönderilen: {sent}"
    )

    log(
        f"{VERSION} tamamlandı."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
