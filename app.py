import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import pytz
from scipy.optimize import brentq
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2 import service_account
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Arrakis Portfolio", layout="wide")

# -------------------------------------------------------------------
# ARRAKIS CUSTOM CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-image: radial-gradient(circle at top right, #2B1F17, #1A1512); }
    [data-testid="stSidebar"] { background-color: #1A1512 !important; border-right: 1px solid #FF6B00; }
    [data-testid="stMetricValue"] { color: #FF6B00 !important; text-shadow: 0 0 5px rgba(255, 107, 0, 0.4); font-family: 'Georgia', serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #2B231D; border-radius: 4px 4px 0px 0px; border: 1px solid #3D3026; color: #EBD5B3; }
    .stTabs [aria-selected="true"] { background-color: #FF6B00 !important; color: #1A1512 !important; font-weight: bold; }
    [data-testid="stDataFrame"] { border: 1px solid #FF6B00; border-radius: 5px; box-shadow: 0 0 10px rgba(255, 107, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

st.title("🏜️ Arrakis Portfolio Terminal")
st.caption("Spice · VaR · Beta · Volatility · Cost Basis · Lot-level P&L")

# -----------------------------
# GOOGLE SHEETS CONNECTION
# -----------------------------
SPREADSHEET_ID = "1Y5tdp37t-p-E6ixLIVQddRMVO-WFdCdCK9L2ed3bjvw"

@st.cache_resource
def get_gspread_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner="Loading portfolio...")
def load_data_from_sheet(sheet_id):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(sheet_id)
        df_port = pd.DataFrame(sheet.sheet1.get_all_records())
        try: df_div = pd.DataFrame(sheet.worksheet("Dividends").get_all_records())
        except: df_div = pd.DataFrame(columns=["Date", "Symbol", "Dividend per Share", "Total Cash", "Currency"])
        try: df_watch = pd.DataFrame(sheet.worksheet("Watchlist").get_all_records())
        except: df_watch = pd.DataFrame(columns=["Symbol"])
        return df_port, df_div, df_watch
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df, df_div, df_watch = load_data_from_sheet(SPREADSHEET_ID)

if df.empty:
    st.warning("No portfolio data found.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
df["Trade Date"] = pd.to_datetime(df["Trade Date"].astype(str), format="%Y%m%d", errors="coerce")
for c in ["Current Price", "Purchase Price", "Quantity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Symbol", "Current Price", "Purchase Price", "Quantity"])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers):
    live_data = {}
    for t in tickers:
        try: live_data[t] = yf.Ticker(t).fast_info['last_price']
        except: live_data[t] = np.nan
    return live_data

def get_market_status():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        return True, now.strftime("%H:%M:%S EST")
    return False, now.strftime("%H:%M:%S EST")

is_market_open, current_time = get_market_status()

with st.sidebar:
    st.header("Terminal Settings")
    if is_market_open:
        st.success(f"🟢 Market Open ({current_time})")
        live_mode = st.toggle("Live Auto-Refresh (60s)", value=True)
    else:
        st.error(f"🔴 Market Closed ({current_time})")
        live_mode = False
        
    benchmark = st.text_input("Benchmark ticker", "^GSPC")
    lookback_years = st.selectbox("Return lookback (years)", [1, 2, 3], index=1)
    conf_levels = st.multiselect("VaR confidence levels", [0.95, 0.99], default=[0.95, 0.99])
    horizon_days = st.number_input("VaR horizon (trading days)", 1, 30, 1)
    fx_ticker = st.text_input("CAD→USD FX ticker", "CADUSD=X")
    st.markdown("---")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if live_mode:
    st_autorefresh(interval=60000, key="live_tracker_refresh")

df["Currency"] = np.where(df["Symbol"].str.endswith(".TO"), "CAD", "USD")

@st.cache_data(ttl=300, show_spinner=False)
def get_fx(ticker):
    try: return yf.Ticker(ticker).fast_info['last_price']
    except: return 0.6974

cad_usd = get_fx(fx_ticker)
st.sidebar.metric("CAD→USD used", f"{cad_usd:.4f}")

tickers = sorted(df["Symbol"].unique().tolist())
live_prices = fetch_live_prices(tuple(tickers))

def get_best_price(row):
    live_p = live_prices.get(row["Symbol"])
    return live_p if pd.notna(live_p) and live_p > 0 else row["Current Price"]

df["Current Price Live"] = df.apply(get_best_price, axis=1)
df["Current Price USD"] = np.where(df["Currency"] == "CAD", df["Current Price Live"] * cad_usd, df["Current Price Live"])
df["Purchase Price USD"] = np.where(df["Currency"] == "CAD", df["Purchase Price"] * cad_usd, df["Purchase Price"])

@st.cache_data(ttl=3600, show_spinner="Checking for new dividends...")
def check_and_update_dividends(sheet_id, tickers):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(sheet_id)
        try: sheet_div = sheet.worksheet("Dividends")
        except: return None 
        existing_divs = pd.DataFrame(sheet_div.get_all_records())
        if existing_divs.empty: existing_divs = pd.DataFrame(columns=["Date", "Symbol", "Dividend per Share", "Total Cash", "Currency"])
        new_rows = []
        for t in tickers:
            try:
                ticker = yf.Ticker(t)
                divs = ticker.dividends
                if divs.empty: continue
                last_div_date = divs.index[-1].strftime("%Y-%m-%d")
                last_div_amt = float(divs.iloc[-1])
                already_logged = existing_divs[(existing_divs["Symbol"] == t) & (existing_divs["Date"] == last_div_date)]
                if already_logged.empty:
                    days_ago = (datetime.now() - divs.index[-1].replace(tzinfo=None)).days
                    if 0 <= days_ago <= 30:
                        total_shares = df[df["Symbol"] == t]["Quantity"].sum()
                        if total_shares > 0:
                            new_rows.append({"Date": last_div_date, "Symbol": t, "Dividend per Share": last_div_amt, "Total Cash": total_shares * last_div_amt, "Currency": "CAD" if t.endswith(".TO") else "USD"})
            except: continue
        if new_rows:
            existing_df = pd.concat([existing_divs, pd.DataFrame(new_rows)], ignore_index=True)
            set_with_dataframe(sheet_div, existing_df)
            return True
        return False
    except: return False

dividends_updated = check_and_update_dividends(SPREADSHEET_ID, tuple(tickers))
if dividends_updated:
    st.toast("🎉 New dividends detected and logged to your sheet!")
    df_div = pd.DataFrame(get_gspread_client().open_by_key(SPREADSHEET_ID).worksheet("Dividends").get_all_records())

if not df_div.empty:
    df_div["Total Cash"] = pd.to_numeric(df_div["Total Cash"], errors="coerce").fillna(0)
    df_div["Total Cash USD"] = np.where(df_div["Currency"] == "CAD", df_div["Total Cash"] * cad_usd, df_div["Total Cash"])
    total_div_usd = df_div["Total Cash USD"].sum()
else:
    total_div_usd = 0

df["Market Value"] = df["Current Price USD"] * df["Quantity"]
df["Cost Basis"] = df["Purchase Price USD"] * df["Quantity"]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prev_closes(tickers):
    prev_closes = {}
    for t in tickers:
        try: prev_closes[t] = yf.Ticker(t).fast_info['previous_close']
        except: prev_closes[t] = np.nan
    return prev_closes

prev_closes = fetch_prev_closes(tuple(tickers))
df["Prev Close"] = df["Symbol"].map(prev_closes)
df["Prev Close USD"] = np.where(df["Currency"] == "CAD", df["Prev Close"] * cad_usd, df["Prev Close"])
df["Daily P&L"] = (df["Current Price USD"] - df["Prev Close USD"]) * df["Quantity"]
daily_pnl = df["Daily P&L"].sum()

# -----------------------------
# TIME PERIOD RETURNS
# -----------------------------
today = datetime.now()
start_of_month = today.replace(day=1)
end_of_last_month = start_of_month - timedelta(days=1)
start_of_quarter = today.replace(month=((today.month-1)//3)*3 + 1, day=1)

@st.cache_data(ttl=3600, show_spinner="Calculating time period returns...")
def fetch_period_returns(tickers, mom_date, qtd_date):
    data = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="3mo")
            if hist.empty:
                data.append({"Symbol": t, "Daily %": np.nan, "MTD %": np.nan, "QTD %": np.nan})
                continue
                
            hist.index = hist.index.tz_localize(None)
            curr_price = hist['Close'].iloc[-1]
            
            daily_ret = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) >= 2 else np.nan
            
            mom_data = hist[hist.index <= mom_date]
            mom_price = mom_data['Close'].iloc[-1] if not mom_data.empty else hist['Close'].iloc[0]
            mom_ret = ((curr_price / mom_price) - 1) * 100 if pd.notna(mom_price) and mom_price > 0 else np.nan
            
            qtd_data = hist[hist.index >= qtd_date]
            qtd_price = qtd_data['Close'].iloc[0] if not qtd_data.empty else hist['Close'].iloc[0]
            qtd_ret = ((curr_price / qtd_price) - 1) * 100 if pd.notna(qtd_price) and qtd_price > 0 else np.nan
            
            data.append({
                "Symbol": t,
                "Daily %": daily_ret,
                "MTD %": mom_ret,
                "QTD %": qtd_ret
            })
        except:
            data.append({"Symbol": t, "Daily %": np.nan, "MTD %": np.nan, "QTD %": np.nan})
            
    return pd.DataFrame(data)

df_period_returns = fetch_period_returns(tuple(tickers), end_of_last_month, start_of_quarter)

pos = df.groupby("Symbol").agg(
    Quantity=("Quantity", "sum"),
    Market_Value=("Market Value", "sum"),
    Cost_Basis=("Cost Basis", "sum"),
    Avg_Purchase_Price=("Purchase Price USD", lambda s: np.nan), 
).reset_index()

wavg = df.groupby("Symbol").apply(
    lambda g: (g["Purchase Price USD"] * g["Quantity"]).sum() / g["Quantity"].sum(),
    include_groups=False,
).rename("Avg_Purchase_Price")
pos = pos.drop(columns=["Avg_Purchase_Price"]).merge(wavg, on="Symbol")

pos["Weight_MV"] = pos["Market_Value"] / pos["Market_Value"].sum()
pos["Weight_Cost"] = pos["Cost_Basis"] / pos["Cost_Basis"].sum()
pos["Unrealized_PnL"] = pos["Market_Value"] - pos["Cost_Basis"]
pos["Return_%"] = np.where(pos["Cost_Basis"] > 0, pos["Unrealized_PnL"] / pos["Cost_Basis"] * 100, np.nan)

pos = pos.merge(df_period_returns, on="Symbol", how="left")

pos["Daily Contrib %"] = (pos["Daily %"] / 100) * pos["Weight_MV"] * 100
pos["MTD Contrib %"] = (pos["MTD %"] / 100) * pos["Weight_MV"] * 100
pos["QTD Contrib %"] = (pos["QTD %"] / 100) * pos["Weight_MV"] * 100

total_port_val = pos["Market_Value"].sum()
pos["Daily $ Impact"] = (pos["Daily Contrib %"] / 100) * total_port_val
pos["MTD $ Impact"] = (pos["MTD Contrib %"] / 100) * total_port_val
pos["QTD $ Impact"] = (pos["QTD Contrib %"] / 100) * total_port_val

port_daily_ret = pos["Daily Contrib %"].sum()
port_mtd_ret = pos["MTD Contrib %"].sum()
port_qtd_ret = pos["QTD Contrib %"].sum()

# -----------------------------
# RISK METRICS
# -----------------------------
end_date = datetime.now()
start_date = end_date - pd.DateOffset(days=int((lookback_years + 0.1) * 365))
bench = benchmark.strip()

@st.cache_data(ttl=3600, show_spinner="Fetching historical prices…")
def fetch_prices(tickers, bench, start, end):
    all_t = list(tickers) + [bench]
    raw = yf.download(all_t, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.Series): close = close.to_frame()
    return close.dropna(how="all")

prices = fetch_prices(tuple(tickers), bench, start_date, end_date)

if cad_usd:
    for t in tickers:
        if t.endswith(".TO") and t in prices.columns: prices[t] = prices[t] * cad_usd

min_obs = int(lookback_years * 200)
valid = [t for t in tickers if t in prices.columns and prices[t].notna().sum() >= min_obs]
missing_data = [t for t in tickers if t not in valid]

if not valid:
    st.error("No tickers have enough price history. Try a shorter lookback.")
    st.stop()

prices = prices[valid + ([bench] if bench in prices.columns else [])].dropna()
returns = prices.pct_change().dropna()

if bench not in returns.columns:
    st.error(f"Benchmark {bench} data missing.")
    st.stop()

w = pos.set_index("Symbol").loc[valid, "Market_Value"]
w = w / w.sum()
port_ret = (returns[valid] * w).sum(axis=1)
bench_ret = returns[bench]

cov = np.cov(port_ret, bench_ret)
beta_port = cov[0, 1] / cov[1, 1]

betas = {}
for t in valid:
    c = np.cov(returns[t], bench_ret)
    betas[t] = c[0, 1] / c[1, 1]
beta_df = pd.DataFrame({"Symbol": list(betas.keys()), "Beta": list(betas.values())})

ann_vol = port_ret.std() * np.sqrt(252)
ann_ret = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
sharpe = ann_ret / ann_vol if ann_vol else np.nan

downside_returns = port_ret[port_ret < 0]
downside_vol = downside_returns.std() * np.sqrt(252)
sortino = ann_ret / downside_vol if downside_vol else np.nan

cum = (1 + port_ret).cumprod()
mdd = (cum / cum.cummax() - 1).min()

bench_ann_ret = (1 + bench_ret).prod() ** (252 / len(bench_ret)) - 1
active_ret = port_ret - bench_ret
tracking_error = active_ret.std() * np.sqrt(252)
alpha = ann_ret - (0.04 + beta_port * (bench_ann_ret - 0.04))
info_ratio = active_ret.mean() * 252 / tracking_error if tracking_error else np.nan

current_year = end_date.year
ytd_prices = prices[prices.index.year == current_year]
if not ytd_prices.empty:
    ytd_returns_series = ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1
    port_ytd = (ytd_returns_series[valid] * w).sum()
    bench_ytd = ytd_returns_series[bench]
else:
    port_ytd = np.nan; bench_ytd = np.nan; ytd_returns_series = pd.Series(dtype=float)

def hist_var(series, conf, horizon):
    if horizon == 1: h = series
    else: h = series.rolling(horizon).apply(lambda x: (1 + x).prod() - 1, raw=True).dropna()
    return -np.percentile(h, (1 - conf) * 100)

var_rows = []
for conf in conf_levels:
    v = hist_var(port_ret, conf, horizon_days)
    var_rows.append({"Confidence": f"{int(conf*100)}%", "Horizon (days)": horizon_days, "VaR (% of portfolio)": v * 100, "VaR ($)": v * pos["Market_Value"].sum()})
var_df = pd.DataFrame(var_rows)

conf_comp = conf_levels[0] if conf_levels else 0.95
port_var_pct = hist_var(port_ret, conf_comp, horizon_days)
total_mv = pos["Market_Value"].sum()

comp_var_data = []
for t in valid:
    stock_var = hist_var(returns[t], conf_comp, horizon_days)
    corr_with_port = np.corrcoef(returns[t], port_ret)[0,1]
    comp_var_pct = stock_var * w[t] * corr_with_port
    comp_var_usd = comp_var_pct * total_mv
    comp_var_data.append({"Symbol": t, "Stock VaR (%)": stock_var * 100, "Weight (%)": w[t] * 100, "Component VaR ($)": comp_var_usd, "YTD Return (%)": ytd_returns_series.get(t, np.nan) * 100 if not ytd_returns_series.empty else np.nan})
comp_var_df = pd.DataFrame(comp_var_data).sort_values("Component VaR ($)", ascending=False)

def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for d, cf in cashflows)
def xirr(cashflows):
    try: return brentq(lambda r: xnpv(r, cashflows), -0.9999, 10)
    except: return np.nan

cf = [(d, -amt) for d, amt in zip(df["Trade Date"], df["Cost Basis"]) if pd.notna(d)]
cf.append((end_date, df["Market Value"].sum() + total_div_usd)) 
cf = sorted(cf, key=lambda x: x[0])
port_xirr = xirr(cf)

# -----------------------------
# DISPLAY
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Overview", "Risk & Benchmark", "Allocation", "Holdings", "Lot-level", "Trade", "Dividends", "Watchlist"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Market Value (USD)", f"${pos['Market_Value'].sum():,.0f}", delta=f"${daily_pnl:,.0f} Today")
    c2.metric("Total Cost Basis (USD)", f"${pos['Cost_Basis'].sum():,.0f}")
    c3.metric("Unrealized P&L", f"${pos['Unrealized_PnL'].sum():,.0f}", f"{pos['Unrealized_PnL'].sum()/pos['Cost_Basis'].sum()*100:.1f}%")
    c4.metric("Dividends Collected", f"${total_div_usd:,.0f}")
    c5.metric("Portfolio XIRR", f"{port_xirr*100:.1f}%" if not np.isnan(port_xirr) else "n/a")

    st.subheader("Returns & Contribution by Time Period")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Portfolio Daily Return", f"{port_daily_ret:+.2f}%")
    rc2.metric("Portfolio Month-to-Date", f"{port_mtd_ret:+.2f}%")
    rc3.metric("Portfolio Quarter-to-Date", f"{port_qtd_ret:+.2f}%")
    
    period_table = pos[["Symbol", "Weight_MV", "Daily %", "Daily Contrib %", "Daily $ Impact", "MTD %", "MTD Contrib %", "MTD $ Impact", "QTD %", "QTD Contrib %", "QTD $ Impact"]].sort_values("Weight_MV", ascending=False).copy()
    period_table = period_table.rename(columns={"Weight_MV": "Weight", "Daily %": "D %", "Daily Contrib %": "D Contrib", "Daily $ Impact": "D $ Impact", "MTD %": "MTD %", "MTD Contrib %": "MTD Contrib", "MTD $ Impact": "MTD $ Impact", "QTD %": "QTD %", "QTD Contrib %": "QTD Contrib", "QTD $ Impact": "QTD $ Impact"})
    period_style = period_table.style.format({
        "Weight": "{:.1%}", 
        "D %": "{:+.2f}%", "D Contrib": "{:+.2f}%", "D $ Impact": "${:+,.2f}",
        "MTD %": "{:+.2f}%", "MTD Contrib": "{:+.2f}%", "MTD $ Impact": "${:+,.2f}",
        "QTD %": "{:+.2f}%", "QTD Contrib": "{:+.2f}%", "QTD $ Impact": "${:+,.2f}"
    }).map(lambda x: 'color: #00FFAA' if x > 0 else 'color: #FF4B4B', subset=["D %", "D Contrib", "D $ Impact", "MTD %", "MTD Contrib", "MTD $ Impact", "QTD %", "QTD Contrib", "QTD $ Impact"])
    st.dataframe(period_style, use_container_width=True)

    st.subheader("Daily Contribution to Return ($)")
    waterfall_df = pos[["Symbol", "Daily $ Impact"]].sort_values("Daily $ Impact", ascending=False)
    fig_waterfall = go.Figure(go.Waterfall(
        name="Daily P&L", orientation="v",
        measure=["relative"] * len(waterfall_df) + ["total"],
        x=list(waterfall_df["Symbol"]) + ["Total Portfolio"],
        y=list(waterfall_df["Daily $ Impact"]) + [daily_pnl],
        text=[f"${v:,.0f}" for v in waterfall_df["Daily $ Impact"]] + [f"${daily_pnl:,.0f}"],
        textposition="outside",
        connector={"line": {"color": "#3D3026"}},
        decreasing={"marker": {"color": "#FF4B4B"}},
        increasing={"marker": {"color": "#00FFAA"}},
        totals={"marker": {"color": "#FF6B00"}}
    ))
    fig_waterfall.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#EBD5B3"), showlegend=False)
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    if missing_data: st.warning(f"No/insufficient price history (excluded from risk): {', '.join(missing_data)}")
    
    st.subheader("Performance & Risk Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio YTD", f"{port_ytd*100:.1f}%" if not np.isnan(port_ytd) else "n/a", delta=f"{bench_ytd*100:.1f}% (Bench)" if not np.isnan(bench_ytd) else None)
    c2.metric("Portfolio Beta", f"{beta_port:.2f}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")
    c4.metric("Sortino Ratio", f"{sortino:.2f}" if not np.isnan(sortino) else "n/a")
    c5.metric("Info Ratio", f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "n/a")

    st.subheader("Benchmark Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracking Error", f"{tracking_error*100:.1f}%")
    c2.metric("Portfolio Alpha", f"{alpha*100:.1f}%")
    c3.metric("Benchmark Ann. Return", f"{bench_ann_ret*100:.1f}%")

    st.subheader("Growth of $10,000 (Portfolio vs Benchmark)")
    cum_port = (1 + port_ret).cumprod() * 10000
    cum_bench = (1 + bench_ret).cumprod() * 10000
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", line=dict(color='#FF6B00', width=3)))
    fig_growth.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=bench, line=dict(color='#EBD5B3', width=2, dash='dash')))
    fig_growth.update_layout(xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#EBD5B3"))
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("Value at Risk (Historical)")
    st.dataframe(var_df.style.format({"VaR (% of portfolio)": "{:.2f}%", "VaR ($)": "${:,.0f}"}), use_container_width=True)

    st.subheader("Component VaR & YTD Breakdown")
    comp_style = comp_var_df.style.format({
        "Stock VaR (%)": "{:.2f}%", "Weight (%)": "{:.1f}%", "Component VaR ($)": "${:,.0f}", "YTD Return (%)": "{:.1f}%"
    }).map(lambda x: 'color: #00FFAA' if x > 0 else 'color: #FF4B4B', subset=["YTD Return (%)"])
    st.dataframe(comp_style, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Per-holding Beta")
        b = beta_df.merge(pos[["Symbol", "Weight_MV"]], on="Symbol").sort_values("Beta", ascending=False)
        st.plotly_chart(px.bar(b, x="Symbol", y="Beta", color="Weight_MV", color_continuous_scale="Oranges"), use_container_width=True)
    with col2:
        st.subheader("Portfolio return distribution")
        st.plotly_chart(px.histogram(port_ret * 100, nbins=80, labels={"value": "Daily return (%)"}, color_discrete_sequence=["#FF6B00"]), use_container_width=True)

    st.subheader("Correlation matrix (daily returns)")
    corr = returns[valid].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Oranges", zmin=-1, zmax=1), use_container_width=True)

with tab3:
    st.subheader("Portfolio Allocation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Donut Chart
        fig_pie = px.pie(pos, names="Symbol", values="Market_Value", hole=0.5, color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#EBD5B3"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Horizontal Bar Chart
        pos_sorted = pos.sort_values("Weight_MV", ascending=True)
        fig_bar = px.bar(pos_sorted, x="Weight_MV", y="Symbol", orientation='h', color="Weight_MV", color_continuous_scale="Oranges")
        fig_bar.update_layout(xaxis_title="Portfolio Weight", yaxis_title="", coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#EBD5B3"))
        fig_bar.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Target Allocation & Rebalancing")
    st.write("Type your Target % below. The app will calculate exactly how many shares you need to Buy or Sell to hit your goal.")
    
    # Target Rebalancing Logic
    # We use st.data_editor to let the user type in target percentages
    rebal_df = pos[["Symbol", "Quantity", "Current Price Live", "Market_Value", "Weight_MV"]].copy()
    rebal_df["Target %"] = 0.0 # Default to 0
    rebal_df = rebal_df.rename(columns={"Weight_MV": "Actual %"})
    
    edited_df = st.data_editor(
        rebal_df,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
            "Quantity": st.column_config.NumberColumn("Current Shares", disabled=True, format="%.2f"),
            "Current Price Live": st.column_config.NumberColumn("Price", disabled=True, format="$%.2f"),
            "Market_Value": st.column_config.NumberColumn("Current Value", disabled=True, format="$%.2f"),
            "Actual %": st.column_config.NumberColumn("Actual %", disabled=True, format="%.2f%%"),
            "Target %": st.column_config.NumberColumn("Target % (Type Here)", min_value=0.0, max_value=100.0, format="%.2f%%")
        },
        hide_index=True,
        use_container_width=True,
        key="rebalance_editor"
    )
    
    # Calculate shares to trade
    total_target = edited_df["Target %"].sum()
    if total_target > 100.0:
        st.warning(f"Your targets sum to {total_target:.2f}%. Please adjust so they equal 100% or less.")
    
    # Calculate dollar target and shares to buy/sell
    edited_df["Target $"] = (edited_df["Target %"] / 100) * total_port_val
    edited_df["$ To Trade"] = edited_df["Target $"] - edited_df["Market_Value"]
    edited_df["Shares To Trade"] = edited_df["$ To Trade"] / edited_df["Current Price Live"]
    
    # Display action items
    actions = edited_df[abs(edited_df["Shares To Trade"]) > 0.01].copy()
    if not actions.empty:
        actions["Action"] = np.where(actions["Shares To Trade"] > 0, "BUY", "SELL")
        actions["Shares To Trade"] = actions["Shares To Trade"].abs()
        
        st.write("### Action Items to Reach Target")
        
        action_style = actions[["Action", "Symbol", "Shares To Trade", "Current Price Live", "$ To Trade"]].style.format({
            "Shares To Trade": "{:.2f}",
            "Current Price Live": "${:.2f}",
            "$ To Trade": "${:+,.2f}"
        }).map(lambda x: 'color: #00FFAA' if x == "BUY" else 'color: #FF4B4B' if x == "SELL" else '', subset=["Action"])
        
        st.dataframe(action_style, use_container_width=True, hide_index=True)
    else:
        st.info("Enter Target % above to see your rebalancing trades.")

with tab4:
    show = pos.sort_values("Market_Value", ascending=False).copy()
    show_style = show.style.format({
        "Quantity": "{:.4f}", "Market_Value": "${:,.2f}", "Cost_Basis": "${:,.2f}",
        "Avg_Purchase_Price": "${:,.2f}", "Weight_MV": "{:.1%}", "Weight_Cost": "{:.1%}",
        "Unrealized_PnL": "${:,.2f}", "Return_%": "{:.1f}%",
    }).map(lambda x: 'color: #00FFAA' if x > 0 else 'color: #FF4B4B', subset=["Unrealized_PnL", "Return_%"])
    st.dataframe(show_style, use_container_width=True)

with tab5:
    lots = df.copy().sort_values(["Symbol", "Trade Date"])
    lots["Days Held"] = (end_date - lots["Trade Date"]).dt.days
    lots["Lot P&L"] = lots["Market Value"] - lots["Cost Basis"]
    lots["Lot Return %"] = np.where(lots["Cost Basis"] > 0, lots["Lot P&L"] / lots["Cost Basis"] * 100, np.nan)
    lots["Term"] = np.where(lots["Days Held"] >= 365, "Long", "Short")
    
    lots_style = lots[["Symbol", "Trade Date", "Quantity", "Purchase Price USD", "Current Price USD",
                       "Cost Basis", "Market Value", "Lot P&L", "Lot Return %", "Days Held", "Term"]].style.format({
                     "Trade Date": "{:%Y-%m-%d}", "Quantity": "{:.4f}",
                     "Purchase Price USD": "${:,.2f}", "Current Price USD": "${:,.2f}",
                     "Cost Basis": "${:,.2f}", "Market Value": "${:,.2f}",
                     "Lot P&L": "${:,.2f}", "Lot Return %": "{:.1f}%",
                 }).map(lambda x: 'color: #00FFAA' if x > 0 else 'color: #FF4B4B', subset=["Lot P&L", "Lot Return %"])
    st.dataframe(lots_style, use_container_width=True)
    st.download_button("Download lot-level CSV", lots.to_csv(index=False), "lots.csv")

with tab6:
    st.subheader("Add a New Trade")
    with st.form("add_trade_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_symbol = st.text_input("Symbol (e.g., AAPL or CSU.TO)").upper().strip()
            new_qty = st.number_input("Quantity", min_value=0.0001, step=1.0, format="%.4f")
        with col2:
            new_price = st.number_input("Price per Share", min_value=0.01, step=1.0, format="%.2f")
            new_date = st.date_input("Trade Date", datetime.now())
        submitted = st.form_submit_button("Submit Trade")
        
        if submitted:
            if not new_symbol: st.error("Please enter a symbol.")
            else:
                new_row = {
                    "Symbol": new_symbol, "Current Price": new_price, "Date": datetime.now().strftime("%Y/%m/%d"),
                    "Time": "16:00 EST", "Change": 0, "Open": 0, "High": 0, "Low": 0, "Volume": 0,
                    "Trade Date": new_date.strftime("%Y%m%d"), "Purchase Price": new_price, "Quantity": new_qty,
                    "Commission": 0, "High Limit": "", "Low Limit": "", "Comment": "Added via App", "Transaction Type": "BUY"
                }
                try:
                    client = get_gspread_client()
                    sheet = client.open_by_key(SPREADSHEET_ID).sheet1
                    existing_df = pd.DataFrame(sheet.get_all_records())
                    existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                    set_with_dataframe(sheet, existing_df)
                    st.success(f"Successfully added {new_qty} shares of {new_symbol}!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e: st.error(f"Failed to add trade: {e}")

with tab7:
    st.subheader("Dividend History & Breakdown")
    if df_div.empty: st.info("No dividends logged yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Dividends Collected (USD)", f"${total_div_usd:,.2f}")
        c2.metric("Total Payouts Logged", len(df_div))
        div_by_symbol = df_div.groupby("Symbol")["Total Cash USD"].sum().reset_index().sort_values("Total Cash USD", ascending=False)
        fig_div = px.bar(div_by_symbol, x="Symbol", y="Total Cash USD", text_auto='.2s', color="Symbol", color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig_div.update_layout(showlegend=False, yaxis_title="Cash Received (USD)")
        st.plotly_chart(fig_div, use_container_width=True)
        
        st.write("### Dividend Log")
        display_div = df_div.sort_values("Date", ascending=False).copy()
        st.dataframe(display_div.style.format({
            "Dividend per Share": "{:.4f}", "Total Cash": "${:,.2f}", "Total Cash USD": "${:,.2f}"
        }), use_container_width=True)

with tab8:
    st.subheader("Watchlist Tracker")
    with st.form("add_watchlist_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1: new_watch_symbol = st.text_input("Add Ticker to Watchlist").upper().strip()
        with col2:
            st.write(""); st.write("")
            submit_watch = st.form_submit_button("Add Ticker")
            
        if submit_watch and new_watch_symbol:
            try:
                client = get_gspread_client()
                sheet = client.open_by_key(SPREADSHEET_ID)
                try: sheet_watch = sheet.worksheet("Watchlist")
                except:
                    st.error("Could not find 'Watchlist' tab in your Google Sheet.")
                    st.stop()
                existing_watch = pd.DataFrame(sheet_watch.get_all_records())
                if new_watch_symbol not in existing_watch["Symbol"].values:
                    new_watch_row = pd.DataFrame([{"Symbol": new_watch_symbol}])
                    existing_watch = pd.concat([existing_watch, new_watch_row], ignore_index=True)
                    set_with_dataframe(sheet_watch, existing_watch)
                    st.success(f"Added {new_watch_symbol} to Watchlist!")
                    st.cache_data.clear()
                    st.rerun()
                else: st.warning(f"{new_watch_symbol} is already on the watchlist.")
            except Exception as e: st.error(f"Error updating watchlist: {e}")

    if df_watch.empty: st.info("Watchlist is empty. Add a ticker above!")
    else:
        watch_tickers = df_watch["Symbol"].tolist()
        
        @st.cache_data(ttl=300, show_spinner="Fetching watchlist data...")
        def fetch_watchlist_data(tickers):
            data = []
            for t in tickers:
                try:
                    ticker = yf.Ticker(t)
                    hist_1y = ticker.history(period="1y")
                    hist_ytd = ticker.history(period="ytd")
                    hist_5d = ticker.history(period="5d")
                    
                    if hist_1y.empty or hist_5d.empty: continue
                    curr_price = hist_5d['Close'].iloc[-1]
                    
                    prev_close = hist_5d['Close'].iloc[-2] if len(hist_5d) >= 2 else curr_price
                    daily_change = ((curr_price / prev_close) - 1) * 100
                    
                    ytd_ret = ((curr_price / hist_ytd['Close'].iloc[0]) - 1) * 100 if not hist_ytd.empty else np.nan
                    
                    high_52w = hist_1y['High'].max()
                    low_52w = hist_1y['Low'].min()
                    
                    curr = "CAD" if t.endswith(".TO") else "USD"
                    
                    data.append({
                        "Symbol": t, "Price": curr_price, "Daily Change (%)": daily_change,
                        "YTD Return (%)": ytd_ret, "52W High": high_52w, "52W Low": low_52w, "Currency": curr
                    })
                except: continue
            return pd.DataFrame(data)
            
        watch_data = fetch_watchlist_data(tuple(watch_tickers))
        
        if not watch_data.empty:
            watch_style = watch_data.style.format({
                "Price": "{:.2f}", "Daily Change (%)": "{:+.2f}%", "YTD Return (%)": "{:+.2f}%",
                "52W High": "{:.2f}", "52W Low": "{:.2f}"
            }).map(lambda x: 'color: #00FFAA' if x > 0 else 'color: #FF4B4B', subset=["Daily Change (%)", "YTD Return (%)"])
            st.dataframe(watch_style, use_container_width=True)
