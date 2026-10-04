"""
statarb.py - data loading, pair screening, signals, backtest and walk-forward
for crypto mean-reversion pairs and trend following. No Streamlit code here,
so the same functions can be imported by a scheduled trading script.
"""
import time
from itertools import product

import numpy as np
import pandas as pd

ANN = 365  # crypto trades every day

# symbol -> Yahoo Finance ticker (checked against Yahoo's crypto screener, Oct 2026).
# Yahoo appends an ID to some tickers; the plain ticker can be a different coin.
UNIVERSE = {
    # large caps
    "BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD", "XRP": "XRP-USD", "SOL": "SOL-USD",
    "TRX": "TRX-USD", "DOGE": "DOGE-USD", "ADA": "ADA-USD", "LINK": "LINK-USD", "XLM": "XLM-USD",
    "BCH": "BCH-USD", "LTC": "LTC-USD", "AVAX": "AVAX-USD", "HBAR": "HBAR-USD", "SHIB": "SHIB-USD",
    "DOT": "DOT-USD", "XMR": "XMR-USD", "ZEC": "ZEC-USD", "TON": "TON11419-USD", "SUI": "SUI20947-USD",
    # alts
    "HYPE": "HYPE32196-USD", "UNI": "UNI7083-USD", "NEAR": "NEAR-USD", "AAVE": "AAVE-USD",
    "TAO": "TAO22974-USD", "ONDO": "ONDO-USD", "ENA": "ENA-USD", "PEPE": "PEPE24478-USD",
    "ICP": "ICP-USD", "WLD": "WLD-USD", "ETC": "ETC-USD", "POL": "POL28321-USD",
    "ARB": "ARB11841-USD", "ALGO": "ALGO-USD", "KAS": "KAS-USD", "RENDER": "RENDER-USD",
    "ATOM": "ATOM-USD", "QNT": "QNT-USD", "CAKE": "CAKE-USD", "INJ": "INJ-USD", "FIL": "FIL-USD",
    "VET": "VET-USD", "DASH": "DASH-USD", "APT": "APT21794-USD", "STX": "STX4847-USD",
    "CRV": "CRV-USD", "PYTH": "PYTH-USD", "FET": "FET-USD", "SEI": "SEI-USD", "TIA": "TIA-USD",
    "BSV": "BSV-USD", "PENDLE": "PENDLE-USD", "XTZ": "XTZ-USD", "LDO": "LDO-USD", "BONK": "BONK-USD",
    "IMX": "IMX10603-USD", "OP": "OP-USD", "AR": "AR-USD", "STRK": "STRK22691-USD", "ENS": "ENS-USD",
    "GRT": "GRT6719-USD", "FLOKI": "FLOKI-USD", "JTO": "JTO-USD", "CRO": "CRO-USD", "RAY": "RAY-USD",
    "XDC": "XDC-USD", "DCR": "DCR-USD", "GNO": "GNO-USD", "CFX": "CFX-USD", "NEXO": "NEXO-USD",
    "MNT": "MNT27075-USD",
}
LARGE_CAPS = list(UNIVERSE)[:20]
ALL_COINS = list(UNIVERSE)


# ============================================================ data
def clean(close, dvol=None, min_rows=250):
    """Drop today's unfinished bar, bad prints and coins with too little history."""
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    close = close[close.index < today]
    close = close.where(close > 0).ffill(limit=3)
    keep = close.columns[close.notna().sum() >= min_rows]
    close = close[keep].dropna(how="all")
    if dvol is not None:
        dvol.index = pd.to_datetime(dvol.index).tz_localize(None).normalize()
        dvol = dvol[~dvol.index.duplicated(keep="last")].reindex(close.index)[keep]
    return close, dvol


def fetch_yahoo(symbols, start):
    """Daily closes and USD volume from Yahoo Finance. Returns (close, dollar_volume, failed)."""
    import yfinance as yf

    tick = {UNIVERSE.get(s, f"{s}-USD"): s for s in symbols}
    names = list(tick)
    closes, vols = [], []
    for i in range(0, len(names), 20):
        chunk = names[i:i + 20]
        df = yf.download(chunk, start=str(start), interval="1d", auto_adjust=False,
                         progress=False, threads=True, group_by="column")
        if df is None or df.empty:
            continue
        c, v = df["Close"], df["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(chunk[0]), v.to_frame(chunk[0])
        closes.append(c)
        vols.append(v)
        time.sleep(0.5)
    if not closes:
        raise RuntimeError("Yahoo Finance returned no data (it rate-limits shared servers; "
                           "try again, or switch the source to Coinbase).")
    close = pd.concat(closes, axis=1).rename(columns=tick).dropna(axis=1, how="all")
    dvol = pd.concat(vols, axis=1).rename(columns=tick)[close.columns]   # Yahoo crypto volume is in USD
    close, dvol = clean(close, dvol)
    return close, dvol, [s for s in symbols if s not in close.columns]


def fetch_coinbase(symbols, start, progress=None):
    """Daily candles from Coinbase Exchange's public API (no key). Returns (close, dollar_volume, failed)."""
    import requests

    start = pd.Timestamp(start)
    last = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    sess = requests.Session()
    sess.headers["User-Agent"] = "statarb-research"
    closes, vols = {}, {}
    for k, sym in enumerate(symbols):
        url = f"https://api.exchange.coinbase.com/products/{sym}-USD/candles"
        rows, end = [], last
        while end > start:
            s = max(start, end - pd.Timedelta(days=299))      # API returns at most 300 candles
            for attempt in range(4):
                r = sess.get(url, params={"granularity": 86400, "start": s.isoformat(),
                                          "end": end.isoformat()}, timeout=20)
                if r.status_code != 429:
                    break
                time.sleep(1 + attempt)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            rows += data
            end = s - pd.Timedelta(days=1)
            time.sleep(0.12)
        if rows:
            a = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "vol"])
            a.index = pd.to_datetime(a["t"], unit="s")
            a = a[~a.index.duplicated()].sort_index()
            closes[sym] = a["close"]
            vols[sym] = a["close"] * a["vol"]                 # volume is in coins
        if progress:
            progress((k + 1) / len(symbols), sym)
    if not closes:
        raise RuntimeError("Coinbase returned no data for the selected coins.")
    close, dvol = clean(pd.DataFrame(closes), pd.DataFrame(vols))
    return close, dvol, [s for s in symbols if s not in close.columns]


def synthetic(symbols, n=2200, seed=1):
    """Made-up prices with some cointegrated pairs, for checking the app offline. Not market data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=n, freq="D")
    drift = pd.Series(rng.normal(0, 0.004, n)).ewm(span=60).mean().values
    mkt = np.cumsum(drift + rng.normal(0, 0.03, n))
    logs, vols = {}, {}
    prev = mkt
    for k, s in enumerate(symbols):
        if k == 0:
            lp = mkt.copy()
        elif k % 3 == 1:                                       # cointegrated with the previous coin
            theta = rng.uniform(0.03, 0.12)
            ou = np.zeros(n)
            for i in range(1, n):
                ou[i] = ou[i - 1] * (1 - theta) + rng.normal(0, 0.025)
            lp = rng.uniform(0.8, 1.2) * prev + ou
        else:
            lp = rng.uniform(0.6, 1.3) * mkt + np.cumsum(rng.normal(0, 0.025, n))
        prev = lp
        lp = lp + rng.uniform(-2, 6)
        if k >= 10:
            lp[: min(n - 400, (k - 9) * 20)] = np.nan          # later listing dates
        logs[s] = lp
        vols[s] = np.exp(rng.normal(np.log(5e8) - 0.08 * k, 0.5, n))
    close = np.exp(pd.DataFrame(logs, index=idx))
    return close, pd.DataFrame(vols, index=idx).where(close.notna()), []


def eligible(px, dvol=None, min_dollar_vol=0.0):
    """Coins with a full price history in this window and enough median daily dollar volume."""
    ok = px.notna().all()
    if dvol is not None and min_dollar_vol > 0:
        ok &= dvol.reindex(px.index).median() >= min_dollar_vol
    return list(ok[ok].index)


# ============================================================ pair screening
def screen_pairs(px, min_corr=0.5, adf_max=-3.34, hl_range=(2, 60), top_k=20, max_per_coin=3):
    """Rank every pair in the window by an Engle-Granger style test on the log-price residual.

    adf_t is the Dickey-Fuller t-stat (no lags) of the residual; -3.34 is roughly the 5%
    critical value for two series. Testing thousands of pairs means many pass by chance,
    so the only fair test of this screen is trading its picks on later data (walk_forward_pairs).
    """
    cols = ["y", "x", "beta", "adf_t", "half_life", "corr"]
    px = px.dropna(axis=1)
    n = px.shape[1]
    if n < 2 or len(px) < 60:
        return pd.DataFrame(columns=cols)
    names = np.array(px.columns)
    L = np.log(px.values)
    corr = np.corrcoef(np.diff(L, axis=0).T)
    Lc = L - L.mean(0)
    cov = Lc.T @ Lc / len(L)
    var = np.diag(cov)
    iu, ju = np.triu_indices(n, 1)
    yi, xi = np.concatenate([iu, ju]), np.concatenate([ju, iu])      # both regression directions
    beta = cov[yi, xi] / var[xi]
    t = np.empty(len(yi))
    rho = np.empty(len(yi))
    for a in range(0, len(yi), 400):                                  # chunked to bound memory
        sl = slice(a, a + 400)
        E = Lc[:, yi[sl]] - Lc[:, xi[sl]] * beta[sl]
        e0, de = E[:-1], np.diff(E, axis=0)
        sxx = (e0 ** 2).sum(0)
        rh = (e0 * de).sum(0) / sxx
        s2 = ((de - e0 * rh) ** 2).sum(0) / (len(de) - 1)
        rho[sl], t[sl] = rh, rh / np.sqrt(s2 / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where(rho < 0, -np.log(2) / np.log1p(np.clip(rho, -0.999, -1e-12)), np.inf)
    df = pd.DataFrame({"y": names[yi], "x": names[xi], "beta": beta, "adf_t": t,
                       "half_life": hl, "corr": corr[yi, xi]})
    df = df[(df["corr"] >= min_corr) & (df["adf_t"] <= adf_max) & (df["beta"] > 0)
            & df["half_life"].between(*hl_range)].sort_values("adf_t")
    picked, used, seen = [], {}, set()
    for row in df.itertuples(index=False):
        key = frozenset((row.y, row.x))
        if key in seen or used.get(row.y, 0) >= max_per_coin or used.get(row.x, 0) >= max_per_coin:
            continue
        seen.add(key)
        used[row.y] = used.get(row.y, 0) + 1
        used[row.x] = used.get(row.x, 0) + 1
        picked.append(row)
        if len(picked) >= top_k:
            break
    return pd.DataFrame(picked, columns=cols)


# ============================================================ signals
def pair_zscore(py, px_, lookback=60):
    ly, lx = np.log(py), np.log(px_)
    beta = ly.rolling(lookback).cov(lx) / lx.rolling(lookback).var()
    alpha = ly.rolling(lookback).mean() - beta * lx.rolling(lookback).mean()
    resid = ly - (alpha + beta * lx)
    return resid / resid.rolling(lookback).std(), beta


def pair_legs(py, px_, lookback=60, entry=2.0, exit_=0.5, stop=4.0, max_hold=30, use_macd=False):
    """Z-score pairs trade. Returns dollar weights for (y, x) with gross exposure 1 when in a trade.

    Enter when |z| > entry (with use_macd, only once the MACD histogram of z has turned back
    toward zero). Exit when z is back inside exit_, when |z| > stop, or after max_hold bars.
    After a stop the pair waits until |z| is back inside entry before it can trade again.
    """
    z, beta = pair_zscore(py, px_, lookback)
    macd = z.ewm(span=12).mean() - z.ewm(span=26).mean()
    hist = (macd - macd.ewm(span=9).mean()).values
    zv = z.values
    state, held, armed = 0.0, 0, True
    pos = np.zeros(len(zv))
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            state = 0.0
            continue
        if state == 0:
            if abs(zi) < entry:
                armed = True
            elif armed and abs(zi) < stop:
                side = -np.sign(zi)                         # z high -> short the spread
                if not use_macd or hist[i] * side > 0:
                    state, held = side, 0
        else:
            held += 1
            if abs(zi) > stop:
                state, armed = 0.0, False
            elif state * zi > -exit_ or held >= max_hold:
                state = 0.0
        pos[i] = state
    s = pd.Series(pos, index=py.index)
    b = beta.clip(0.1, 5).fillna(0)
    gross = 1 + b
    return s / gross, -s * b / gross


def mr_weights(px, pairs, **kw):
    """Equal capital per pair. pairs is a list of (y, x) tuples."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for y, x in pairs:
        wy, wx = pair_legs(px[y], px[x], **kw)
        w[y] += wy / len(pairs)
        w[x] += wx / len(pairs)
    return w


def trend_weights(px, speeds=((8, 32), (16, 64), (32, 128)), vol_lb=30, target_vol=0.20,
                  cap=1.0, long_only=False):
    """Blend of EMA crossovers, scaled by volatility, equal risk per coin that has data."""
    r = px.pct_change(fill_method=None)
    dvol = r.ewm(span=vol_lb, min_periods=vol_lb).std()
    fc = 0
    for fast, slow in speeds:
        raw = (px.ewm(span=fast, min_periods=slow).mean()
               - px.ewm(span=slow, min_periods=slow).mean()) / (px * dvol)
        scaled = raw / raw.abs().expanding(min_periods=60).mean()        # uses past data only
        fc = fc + scaled.clip(-2, 2) / len(speeds)
    if long_only:
        fc = fc.clip(lower=0)
    live = fc.notna().sum(axis=1).clip(lower=1)
    w = (fc * target_vol / (dvol * np.sqrt(ANN))).div(live, axis=0)
    return w.clip(-cap, cap).fillna(0)


def vol_target(px, w, target=0.15, lb=60, max_lev=3.0):
    """Scale the whole book toward a target annualised vol, using only past realised vol."""
    pnl = (w.shift(1) * px.pct_change(fill_method=None)).sum(axis=1)
    realised = pnl.rolling(lb).std() * np.sqrt(ANN)
    lev = (target / realised.where(realised > 0)).shift(1).clip(upper=max_lev).fillna(0)
    return w.mul(lev, axis=0)


# ============================================================ backtest
def backtest(px, w, cost_bps=15, short_carry_bps=0, lag=1):
    """Daily results net of costs. Weights decided at close t earn returns from close t+lag-1 on.

    lag=1 assumes you trade at the same close the signal uses; lag=2 is the conservative check.
    cost_bps is charged on every unit of turnover; short_carry_bps is an annual cost on shorts
    (borrow, or funding/roll if you short through futures).
    """
    r = px.pct_change(fill_method=None).fillna(0)
    held = w.reindex(index=px.index, columns=px.columns).fillna(0).shift(lag).fillna(0)
    turnover = held.diff().abs().sum(axis=1).fillna(0)
    carry = held.clip(upper=0).abs().sum(axis=1) * short_carry_bps / 1e4 / ANN
    ret = (held * r).sum(axis=1) - turnover * cost_bps / 1e4 - carry
    return pd.DataFrame({"ret": ret, "gross": held.abs().sum(axis=1), "turnover": turnover})


def stats(bt):
    r = bt["ret"].dropna()
    if len(r) < 2 or r.std() == 0:
        return {"Sharpe": 0.0, "Ann. return": 0.0, "Vol": 0.0, "Max DD": 0.0,
                "Avg gross": 0.0, "Turnover/yr": 0.0, "Days": len(r)}
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(ANN)
    return {
        "Sharpe": float(r.mean() * ANN / vol),
        "Ann. return": float(eq.iloc[-1] ** (ANN / len(r)) - 1),
        "Vol": float(vol),
        "Max DD": float((eq / eq.cummax() - 1).min()),
        "Avg gross": float(bt["gross"].mean()),
        "Turnover/yr": float(bt["turnover"].mean() * ANN),
        "Days": len(r),
    }


def sharpe(bt):
    return stats(bt)["Sharpe"]


# ============================================================ walk-forward
def _windows(n, train, test):
    return range(train, n, test)


def walk_forward_pairs(px, dvol=None, train=365, test=90, min_dollar_vol=0.0, screen_kw=None,
                       base_kw=None, grid=None, bt_kw=None):
    """Each window: screen pairs and (optionally) pick parameters on the training data only,
    then trade those pairs on the following test window. Returns (stitched weights, log)."""
    screen_kw, base_kw, bt_kw = screen_kw or {}, base_kw or {}, bt_kw or {}
    combos = [dict(zip(grid, v)) for v in product(*grid.values())] if grid else [{}]
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    log = []
    for start in _windows(len(px), train, test):
        tr = px.iloc[start - train:start]
        coins = eligible(tr, dvol, min_dollar_vol)
        picks = screen_pairs(tr[coins], **screen_kw)
        row = {"test_start": px.index[start].date(), "coins": len(coins), "pairs": len(picks)}
        if len(picks):
            plist = list(zip(picks["y"], picks["x"]))
            best = max(combos, key=lambda p: sharpe(backtest(tr, mr_weights(tr, plist, **{**base_kw, **p}), **bt_kw)))
            seg = px.iloc[start - train:start + test]              # training part is warm-up only
            wseg = mr_weights(seg, plist, **{**base_kw, **best}).iloc[train:]
            W.loc[wseg.index] = wseg
            row.update(best)
            row["picked"] = ", ".join(f"{y}/{x}" for y, x in plist)
        log.append(row)
    return W, pd.DataFrame(log)


def walk_forward_trend(px, dvol=None, train=365, test=90, min_dollar_vol=0.0, speed_sets=None,
                       base_kw=None, bt_kw=None):
    """Each window: pick the EMA speed set with the best training Sharpe, trade it on the test window."""
    base_kw, bt_kw = base_kw or {}, bt_kw or {}
    speed_sets = speed_sets or [((8, 32), (16, 64), (32, 128))]
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    log = []
    for start in _windows(len(px), train, test):
        tr = px.iloc[start - train:start]
        coins = eligible(tr, dvol, min_dollar_vol)
        row = {"test_start": px.index[start].date(), "coins": len(coins)}
        if coins:
            best = max(speed_sets, key=lambda s: sharpe(backtest(tr[coins], trend_weights(tr[coins], speeds=s, **base_kw), **bt_kw)))
            seg = px.iloc[start - train:start + test][coins]
            wseg = trend_weights(seg, speeds=best, **base_kw).iloc[train:]
            W.loc[wseg.index, coins] = wseg
            row["speeds"] = str(best)
        log.append(row)
    return W, pd.DataFrame(log)
