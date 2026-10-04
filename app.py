"""
Crypto stat-arb research app: pair screening, in-sample backtest and walk-forward test.
Run locally with:  streamlit run app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import statarb as sa

st.set_page_config(page_title="Crypto stat-arb research", layout="wide")
st.title("Crypto stat-arb research")
st.caption("Mean-reversion pairs and trend following on daily spot prices. Research tool only: "
           "it places no orders, and backtest results are not a forecast.")

SPEED_SETS = {
    "Fast (8/32)": ((8, 32),),
    "Medium (16/64)": ((16, 64),),
    "Slow (32/128)": ((32, 128),),
    "Blend of all three": ((8, 32), (16, 64), (32, 128)),
}

# ------------------------------------------------------------ sidebar
with st.sidebar.form("settings"):
    st.header("Data")
    source = st.selectbox("Price source", ["Yahoo Finance", "Coinbase Exchange", "Upload CSV", "Synthetic demo"])
    preset = st.selectbox("Universe", ["Large caps + alts (71)", "Large caps (20)", "Custom"])
    custom = st.multiselect("Custom coins", sa.ALL_COINS, default=sa.LARGE_CAPS)
    start = st.date_input("History start", dt.date(2019, 1, 1), min_value=dt.date(2014, 1, 1))
    upload = st.file_uploader("CSV (date index, one close column per coin)", type="csv")
    min_dv = st.number_input("Min median daily volume, $m", 0.0, 5000.0, 10.0, 5.0)
    holdout = st.number_input("Holdout: hide the last N days", 0, 730, 0, 30)

    st.header("Pair screen")
    top_k = st.slider("Max pairs to trade", 1, 40, 15)
    max_per_coin = st.slider("Max pairs per coin", 1, 10, 3)
    min_corr = st.slider("Min return correlation", 0.0, 0.95, 0.5, 0.05)
    adf_max = st.slider("Max residual ADF t-stat", -6.0, -1.5, -3.34, 0.02)
    hl_range = st.slider("Half-life range, days", 1, 120, (2, 45))

    st.header("Pair rules")
    lookback = st.slider("Z-score lookback, days", 20, 180, 60, 5)
    entry = st.slider("Entry |z|", 1.0, 3.5, 2.0, 0.1)
    exit_ = st.slider("Exit |z|", 0.0, 1.5, 0.5, 0.1)
    stop = st.slider("Stop |z|", 2.5, 8.0, 4.0, 0.25)
    max_hold = st.slider("Max holding days", 5, 120, 30, 5)
    use_macd = st.checkbox("Require MACD turn on the z-score before entry", False)

    st.header("Trend rules")
    speed_name = st.selectbox("EMA speeds", list(SPEED_SETS), index=3)
    long_only = st.checkbox("Long or flat only (no shorts)", False)
    trend_vol = st.slider("Trend target vol", 0.05, 0.60, 0.20, 0.05)

    st.header("Portfolio and costs")
    mix = st.slider("Weight on pairs (rest on trend)", 0.0, 1.0, 0.5, 0.05)
    use_vt = st.checkbox("Scale combined book to a vol target", True)
    book_vol = st.slider("Combined target vol", 0.05, 0.60, 0.15, 0.05)
    cost_bps = st.slider("Cost per trade, bps of notional", 0, 100, 15)
    carry_bps = st.slider("Annual cost of shorts, bps", 0, 3000, 500, 50)
    lag = st.radio("Execution lag, days", [1, 2], horizontal=True)

    st.header("Walk-forward")
    train = st.slider("Training window, days", 250, 1000, 365, 5)
    test = st.slider("Test window, days", 30, 365, 90, 5)
    tune = st.checkbox("Also re-pick lookback and entry each window", True)
    st.form_submit_button("Run", type="primary")

symbols = {"Large caps + alts (71)": sa.ALL_COINS, "Large caps (20)": sa.LARGE_CAPS}.get(preset, custom)


# ------------------------------------------------------------ cached work
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices...")
def load(source, symbols, start, csv_bytes):
    if source == "Yahoo Finance":
        return sa.fetch_yahoo(list(symbols), start)
    if source == "Coinbase Exchange":
        return sa.fetch_coinbase(list(symbols), start)
    if source == "Upload CSV":
        import io
        raw = pd.read_csv(io.BytesIO(csv_bytes), index_col=0, parse_dates=True)
        close, _ = sa.clean(raw.apply(pd.to_numeric, errors="coerce"))
        return close[close.index >= pd.Timestamp(start)], None, []
    close, dvol, failed = sa.synthetic(list(symbols))
    keep = close.index >= pd.Timestamp(start)
    return close[keep], dvol[keep], failed


@st.cache_data(show_spinner="Running in-sample backtest...")
def run_in_sample(px, dvol, min_dv, screen_kw, pair_kw, trend_kw, bt_kw):
    coins = sa.eligible(px.dropna(axis=1), dvol, min_dv)          # full-history coins for pairs
    picks = sa.screen_pairs(px[coins], **screen_kw) if len(coins) > 1 else pd.DataFrame()
    plist = list(zip(picks["y"], picks["x"])) if len(picks) else []
    w_mr = sa.mr_weights(px, plist, **pair_kw) if plist else pd.DataFrame(0.0, index=px.index, columns=px.columns)
    tcoins = [c for c in px.columns if dvol is None or min_dv <= 0 or dvol[c].median() >= min_dv]
    w_tr = sa.trend_weights(px[tcoins], **trend_kw).reindex(columns=px.columns).fillna(0)
    return picks, w_mr, w_tr


@st.cache_data(show_spinner="Running walk-forward test (this is the slow one)...")
def run_walk_forward(px, dvol, min_dv, train, test, screen_kw, pair_kw, grid, trend_kw, speed_sets, bt_kw):
    w_mr, log_mr = sa.walk_forward_pairs(px, dvol, train, test, min_dv, screen_kw, pair_kw, grid, bt_kw)
    w_tr, log_tr = sa.walk_forward_trend(px, dvol, train, test, min_dv, speed_sets, trend_kw, bt_kw)
    return w_mr, log_mr, w_tr, log_tr


def combine(px, w_mr, w_tr):
    w = mix * w_mr + (1 - mix) * w_tr
    return sa.vol_target(px, w, book_vol) if use_vt else w


def report(px, books, since=None):
    """Stats table plus equity and drawdown charts for a dict of {name: weights}."""
    rows, eq = {}, {}
    for name, w in books.items():
        bt = sa.backtest(px, w, **bt_kw)
        if since is not None:
            bt = bt[bt.index >= since]
        rows[name] = sa.stats(bt)
        eq[name] = (1 + bt["ret"]).cumprod()
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format({"Sharpe": "{:.2f}", "Ann. return": "{:.1%}", "Vol": "{:.1%}",
                                     "Max DD": "{:.1%}", "Avg gross": "{:.2f}", "Turnover/yr": "{:.1f}",
                                     "Days": "{:.0f}"}))
    eq = pd.DataFrame(eq)
    st.markdown("**Growth of 1, net of costs**")
    st.line_chart(eq)
    st.markdown("**Drawdown**")
    st.line_chart(eq / eq.cummax() - 1)
    return table, eq


# ------------------------------------------------------------ load data
if source == "Upload CSV" and upload is None:
    st.info("Upload a CSV in the sidebar, then press Run.")
    st.stop()
if len(symbols) < 2 and source != "Upload CSV":
    st.error("Pick at least two coins.")
    st.stop()
try:
    px_all, dvol_all, failed = load(source, tuple(symbols), start, upload.getvalue() if upload else None)
except Exception as e:                                             # network, rate limit, bad file
    st.error(f"Could not load prices from {source}: {e}")
    st.stop()
if px_all.shape[1] < 2 or len(px_all) < train + test:
    st.error(f"Not enough data: {px_all.shape[1]} coins and {len(px_all)} days loaded; "
             f"the walk-forward needs at least {train + test} days.")
    st.stop()

px = px_all.iloc[:-holdout] if holdout else px_all
dvol = None if dvol_all is None else dvol_all.reindex(px.index)
min_dv_usd = min_dv * 1e6 if dvol is not None else 0.0

screen_kw = dict(min_corr=min_corr, adf_max=adf_max, hl_range=tuple(hl_range), top_k=top_k, max_per_coin=max_per_coin)
pair_kw = dict(lookback=lookback, entry=entry, exit_=exit_, stop=stop, max_hold=max_hold, use_macd=use_macd)
trend_kw = dict(target_vol=trend_vol, long_only=long_only)
bt_kw = dict(cost_bps=cost_bps, short_carry_bps=carry_bps, lag=lag)
grid = {"lookback": [30, 60, 90], "entry": [1.5, 2.0, 2.5]} if tune else None

if source == "Synthetic demo":
    st.warning("Synthetic demo data: made-up prices with cointegration built in. Results mean nothing "
               "about real markets; this source only shows that the app works.")

tab_data, tab_screen, tab_is, tab_wf = st.tabs(
    ["Data", "Pair screen", "Backtest (in-sample)", "Walk-forward (out-of-sample)"])

# ------------------------------------------------------------ data tab
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Coins loaded", px.shape[1])
    c2.metric("First day", str(px.index[0].date()))
    c3.metric("Last day used", str(px.index[-1].date()))
    if holdout:
        st.info(f"The last {holdout} days ({px_all.index[-holdout].date()} onward) are held out and not used "
                "anywhere. Set the holdout to 0 once your rules are final to see how they did.")
    if failed:
        st.warning("No usable data for: " + ", ".join(failed))
    cover = pd.DataFrame({"First day": px.apply(lambda s: s.first_valid_index().date()),
                          "Days": px.notna().sum(), "Last price": px.ffill().iloc[-1]})
    if dvol is not None:
        cover["Median daily volume, $m"] = (dvol.median() / 1e6).round(1)
    st.dataframe(cover)
    st.caption("The coin list is today's survivors. Coins that died or were delisted are missing, "
               "which flatters any backtest that reaches back several years.")

# ------------------------------------------------------------ in-sample
picks, w_mr, w_tr = run_in_sample(px, dvol, min_dv_usd, screen_kw, pair_kw,
                                  dict(trend_kw, speeds=SPEED_SETS[speed_name]), bt_kw)

with tab_screen:
    n_full = px.dropna(axis=1).shape[1]
    st.markdown(f"Screened **{n_full * (n_full - 1) // 2:,}** pairs among the {n_full} coins with a full "
                f"history; **{len(picks)}** passed. Ranked by the residual's ADF t-stat (more negative is stronger).")
    if len(picks):
        st.dataframe(picks.style.format({"beta": "{:.2f}", "adf_t": "{:.2f}", "half_life": "{:.1f}", "corr": "{:.2f}"}))
        labels = [f"{y}/{x}" for y, x in zip(picks["y"], picks["x"])]
        chosen = st.selectbox("Inspect a pair", labels)
        y, x = chosen.split("/")
        z, _ = sa.pair_zscore(px[y], px[x], lookback)
        st.markdown(f"**Rolling z-score of {chosen}**")
        st.line_chart(pd.DataFrame({"z": z, "entry": entry, "-entry": -entry}).dropna())
    else:
        st.info("No pairs passed. Loosen the screen (higher ADF t-stat, lower correlation) or add coins.")
    st.caption("This screen uses the whole sample, so these pairs were chosen with hindsight. "
               "The walk-forward tab re-runs the screen using only past data.")

with tab_is:
    st.warning("In-sample: pairs were selected and traded on the same data, and the settings are whatever "
               "you chose after looking at results. Treat these numbers as an upper bound.")
    is_table, _ = report(px, {"Pairs": w_mr, "Trend": w_tr, "Combined": combine(px, w_mr, w_tr)})

# ------------------------------------------------------------ walk-forward
wf_mr, log_mr, wf_tr, log_tr = run_walk_forward(
    px, dvol, min_dv_usd, train, test, screen_kw, pair_kw, grid, trend_kw,
    list(SPEED_SETS.values()), bt_kw)
wf_combo = combine(px, wf_mr, wf_tr)

with tab_wf:
    oos_start = px.index[train]
    st.markdown(f"Every **{test}** days the app re-screens pairs and re-picks parameters on the previous "
                f"**{train}** days, then trades them on the next {test}. Out-of-sample period starts "
                f"**{oos_start.date()}**; costs are charged when the pair set changes.")
    wf_table, wf_eq = report(px, {"Pairs": wf_mr, "Trend": wf_tr, "Combined": wf_combo}, since=oos_start)
    cmp = pd.DataFrame({"In-sample Sharpe": is_table["Sharpe"], "Walk-forward Sharpe": wf_table["Sharpe"]})
    st.markdown("**In-sample vs walk-forward**")
    st.dataframe(cmp.style.format("{:.2f}"))
    st.caption("A large drop from the first column to the second is the size of the overfitting.")

    st.markdown("**Current target weights** (what the walk-forward book holds after the last close)")
    last = pd.DataFrame({"Pairs": wf_mr.iloc[-1], "Trend": wf_tr.iloc[-1], "Combined": wf_combo.iloc[-1]})
    last = last[(last.abs() > 1e-4).any(axis=1)].sort_values("Combined")
    if len(last):
        st.dataframe(last.style.format("{:+.2%}"))
    else:
        st.write("Flat.")
    st.caption("To forward-test, record these weights each day and compare the paper P&L with this page later.")

    with st.expander("Pairs chosen in each window"):
        st.dataframe(log_mr)
    with st.expander("Trend speeds chosen in each window"):
        st.dataframe(log_tr)
    out = pd.DataFrame({k: sa.backtest(px, w, **bt_kw)["ret"] for k, w in
                        {"pairs": wf_mr, "trend": wf_tr, "combined": wf_combo}.items()})
    st.download_button("Download daily out-of-sample returns (CSV)",
                       out[out.index >= oos_start].to_csv().encode(), "walk_forward_returns.csv", "text/csv")
    st.download_button("Download current target weights (CSV)", last.to_csv().encode(),
                       "target_weights.csv", "text/csv")
