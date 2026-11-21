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

st.set_page_config(page_title="Portfolio Risk Tracker", layout="wide")
st.title("📊 Portfolio Risk & Traits Tracker")
st.caption("VaR · Beta · Volatility · Correlation · Cost Basis · Lot-level P&L")

# -----------------------------
# GOOGLE SHEETS CONNECTION (READ/WRITE)
# -----------------------------
SPREADSHEET_ID = "1Y5tdp37t-p-E6ixLIVQddRMVO-WFdCdCK9L2ed3bjvw" # <--- MAKE SURE THIS IS HERE

@st.cache_resource
def get_gspread_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=300, show_spinner="Loading portfolio...")
def load_data_from_sheet(sheet_id):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(sheet_id)
        
        # Load Portfolio (Sheet1)
        sheet1 = sheet.sheet1
        df_port = pd.DataFrame(sheet1.get_all_records())
        
        # Load Dividends
        try:
            sheet_div = sheet.worksheet("Dividends")
            df_div = pd.DataFrame(sheet_div.get_all_records())
        except:
            df_div = pd.DataFrame(columns=["Date", "Symbol", "Dividend per Share", "Total Cash", "Currency"])
            
        # Load Watchlist
        try:
            sheet_watch = sheet.worksheet("Watchlist")
            df_watch = pd.DataFrame(sheet_watch.get_all_records())
        except:
            df_watch = pd.DataFrame(columns=["Symbol"])
            
        return df_port, df_div, df_watch
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df, df_div, df_watch = load_data_from_sheet(SPREADSHEET_ID)

if df.empty:
    st.warning("No portfolio data found.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
required = ["Symbol", "Current Price", "Trade Date", "Purchase Price", "Quantity"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in Google Sheet: {missing}")
    st.stop()

df["Trade Date"] = pd.to_datetime(df["Trade Date"].astype(str), format="%Y%m%d", errors="coerce")
for c in ["Current Price", "Purchase Price", "Quantity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Symbol", "Current Price", "Purchase Price", "Quantity"])

# -----------------------------
# MARKET STATUS & LIVE PRICES
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers):
    live_data = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            price = ticker.fast_info['last_price']
            live_data[t] = price
        except:
            live_data[t] = np.nan
    return live_data

def get_market_status():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        return True, now.strftime("%H:%M:%S EST")
    return False, now.strftime("%H:%M:%S EST")

is_market_open, current_time = get_market_status()

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
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

# -----------------------------
# Currency & Live Price Handling
# -----------------------------
df["Currency"] = np.where(df["Symbol"].str.endswith(".TO"), "CAD", "USD")

@st.cache_data(ttl=300, show_spinner=False)
def get_fx(ticker):
    try:
        fx = yf.Ticker(ticker)
        return fx.fast_info['last_price']
    except:
        return 0.6974

cad_usd = get_fx(fx_ticker)
st.sidebar.metric("CAD→USD used", f"{cad_usd:.4f}")

tickers = sorted(df["Symbol"].unique().tolist())
live_prices = fetch_live_prices(tuple(tickers))

def get_best_price(row):
    live_p = live_prices.get(row["Symbol"])
    if pd.notna(live_p) and live_p > 0:
        return live_p
    return row["Current Price"]

df["Current Price Live"] = df.apply(get_best_price, axis=1)
df["Current Price USD"] = np.where(df["Currency"] == "CAD", df["Current Price Live"] * cad_usd, df["Current Price Live"])
df["Purchase Price USD"] = np.where(df["Currency"] == "CAD", df["Purchase Price"] * cad_usd, df["Purchase Price"])

# -----------------------------
# DIVIDEND AUTO-TRACKER
# -----------------------------
@st.cache_data(ttl=3600, show_spinner="Checking for new dividends...")
def check_and_update_dividends(sheet_id, tickers):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(sheet_id)
        
        try:
            sheet_div = sheet.worksheet("Dividends")
        except:
            return None 
            
        existing_divs = pd.DataFrame(sheet_div.get_all_records())
        if existing_divs.empty:
            existing_divs = pd.DataFrame(columns=["Date", "Symbol", "Dividend per Share", "Total Cash", "Currency"])
            
        new_rows = []
        
        for t in tickers:
            try:
                ticker = yf.Ticker(t)
                divs = ticker.dividends
                if divs.empty:
                    continue
                
                last_div_date = divs.index[-1].strftime("%Y-%m-%d")
                last_div_amt = float(divs.iloc[-1])
                
                already_logged = existing_divs[
                    (existing_divs["Symbol"] == t) & 
                    (existing_divs["Date"] == last_div_date)
                ]
                
                if already_logged.empty:
                    days_ago = (datetime.now() - divs.index[-1].replace(tzinfo=None)).days
                    if 0 <= days_ago <= 30:
                        total_shares = df[df["Symbol"] == t]["Quantity"].sum()
                        if total_shares > 0:
                            new_rows.append({
                                "Date": last_div_date,
                                "Symbol": t,
                                "Dividend per Share": last_div_amt,
                                "Total Cash": total_shares * last_div_amt,
                                "Currency": "CAD" if t.endswith(".TO") else "USD"
                            })
            except:
                continue
                
        if new_rows:
            existing_df = pd.concat([existing_divs, pd.DataFrame(new_rows)], ignore_index=True)
            set_with_dataframe(sheet_div, existing_df)
            return True
        return False
    except Exception as e:
        st.sidebar.error(f"Dividend check failed: {e}")
        return False

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

# -----------------------------
# Position aggregation
# -----------------------------
df["Market Value"] = df["Current Price USD"] * df["Quantity"]
df["Cost Basis"] = df["Purchase Price USD"] * df["Quantity"]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prev_closes(tickers):
    prev_closes = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            prev_closes[t] = ticker.fast_info['previous_close']
        except:
            prev_closes[t] = np.nan
    return prev_closes

prev_closes = fetch_prev_closes(tuple(tickers))
df["Prev Close"] = df["Symbol"].map(prev_closes)
df["Prev Close USD"] = np.where(df["Currency"] == "CAD", df["Prev Close"] * cad_usd, df["Prev Close"])
df["Daily P&L"] = (df["Current Price USD"] - df["Prev Close USD"]) * df["Quantity"]

daily_pnl = df["Daily P&L"].sum()

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

# -----------------------------
# Price history for risk
# -----------------------------
end_date = datetime(2026, 11, 19)
start_date = end_date - pd.DateOffset(days=int((lookback_years + 0.1) * 365))
bench = benchmark.strip()

@st.cache_data(ttl=3600, show_spinner="Fetching historical prices…")
def fetch_prices(tickers, bench, start, end):
    all_t = list(tickers) + [bench]
    raw = yf.download(all_t, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close.dropna(how="all")

prices = fetch_prices(tuple(tickers), bench, start_date, end_date)

if cad_usd:
    for t in tickers:
        if t.endswith(".TO") and t in prices.columns:
            prices[t] = prices[t] * cad_usd

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

# -----------------------------
# Weights, Risk, & Metrics
# -----------------------------
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
    port_ytd = np.nan
    bench_ytd = np.nan
    ytd_returns_series = pd.Series(dtype=float)

def hist_var(series, conf, horizon):
    if horizon == 1:
        h = series
    else:
        h = series.rolling(horizon).apply(lambda x: (1 + x).prod() - 1, raw=True).dropna()
    return -np.percentile(h, (1 - conf) * 100)

var_rows = []
for conf in conf_levels:
    v = hist_var(port_ret, conf, horizon_days)
    var_rows.append({
        "Confidence": f"{int(conf*100)}%",
        "Horizon (days)": horizon_days,
        "VaR (% of portfolio)": v * 100,
        "VaR ($)": v * pos["Market_Value"].sum(),
    })
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
    
    comp_var_data.append({
        "Symbol": t,
        "Stock VaR (%)": stock_var * 100,
        "Weight (%)": w[t] * 100,
        "Component VaR ($)": comp_var_usd,
        "YTD Return (%)": ytd_returns_series.get(t, np.nan) * 100 if not ytd_returns_series.empty else np.nan
    })
comp_var_df = pd.DataFrame(comp_var_data).sort_values("Component VaR ($)", ascending=False)

def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for d, cf in cashflows)

def xirr(cashflows):
    try:
        return brentq(lambda r: xnpv(r, cashflows), -0.9999, 10)
    except Exception:
        return np.nan

cf = [(d, -amt) for d, amt in zip(df["Trade Date"], df["Cost Basis"]) if pd.notna(d)]
cf.append((end_date, df["Market Value"].sum() + total_div_usd)) 
cf = sorted(cf, key=lambda x: x[0])
port_xirr = xirr(cf)

# -----------------------------
# DISPLAY
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Risk & Benchmark Metrics", "Holdings", "Lot-level", "Add Trade", "Dividend History", "Watchlist"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Market Value (USD)", f"${pos['Market_Value'].sum():,.0f}", delta=f"${daily_pnl:,.0f} Today")
    c2.metric("Total Cost Basis (USD)", f"${pos['Cost_Basis'].sum():,.0f}")
    c3.metric("Unrealized P&L", f"${pos['Unrealized_PnL'].sum():,.0f}",
              f"{pos['Unrealized_PnL'].sum()/pos['Cost_Basis'].sum()*100:.1f}%")
    c4.metric("Dividends Collected", f"${total_div_usd:,.0f}")
    c5.metric("Portfolio XIRR", f"{port_xirr*100:.1f}%" if not np.isnan(port_xirr) else "n/a")

    st.subheader("Allocation by market value")
    fig = px.pie(pos, names="Symbol", values="Market_Value", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost basis vs market value")
    cmp = pos[["Symbol", "Cost_Basis", "Market_Value"]].melt(id_vars="Symbol",
            var_name="Type", value_name="USD")
    st.plotly_chart(px.bar(cmp, x="Symbol", y="USD", color="Type", barmode="group"),
                    use_container_width=True)

with tab2:
    if missing_data:
        st.warning(f"No/insufficient price history (excluded from risk): {', '.join(missing_data)}")
    
    st.subheader("Performance & Risk Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio YTD", f"{port_ytd*100:.1f}%" if not np.isnan(port_ytd) else "n/a", 
              delta=f"{bench_ytd*100:.1f}% (Bench)" if not np.isnan(bench_ytd) else None)
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
    fig_growth.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", line=dict(color='royalblue', width=2)))
    fig_growth.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=bench, line=dict(color='gray', width=2, dash='dash')))
    fig_growth.update_layout(xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified")
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("Value at Risk (Historical)")
    st.dataframe(var_df.style.format({"VaR (% of portfolio)": "{:.2f}%", "VaR ($)": "${:,.0f}"}),
                 use_container_width=True)

    st.subheader("Component VaR & YTD Breakdown")
    st.dataframe(comp_var_df.style.format({
        "Stock VaR (%)": "{:.2f}%",
        "Weight (%)": "{:.1f}%",
        "Component VaR ($)": "${:,.0f}",
        "YTD Return (%)": "{:.1f}%"
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Per-holding Beta")
        b = beta_df.merge(pos[["Symbol", "Weight_MV"]], on="Symbol").sort_values("Beta", ascending=False)
        st.plotly_chart(px.bar(b, x="Symbol", y="Beta", color="Weight_MV",
                               color_continuous_scale="Blues"), use_container_width=True)
    with col2:
        st.subheader("Portfolio return distribution")
        st.plotly_chart(px.histogram(port_ret * 100, nbins=80, labels={"value": "Daily return (%)"}),
                        use_container_width=True)

    st.subheader("Correlation matrix (daily returns)")
    corr = returns[valid].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1), use_container_width=True)

with tab3:
    show = pos.sort_values("Market_Value", ascending=False).copy()
    st.dataframe(show.style.format({
        "Quantity": "{:.4f}", "Market_Value": "${:,.2f}", "Cost_Basis": "${:,.2f}",
        "Avg_Purchase_Price": "${:,.2f}", "Weight_MV": "{:.1%}", "Weight_Cost": "{:.1%}",
        "Unrealized_PnL": "${:,.2f}", "Return_%": "{:.1f}%",
    }), use_container_width=True)

with tab4:
    lots = df.copy().sort_values(["Symbol", "Trade Date"])
    lots["Days Held"] = (end_date - lots["Trade Date"]).dt.days
    lots["Lot P&L"] = lots["Market Value"] - lots["Cost Basis"]
    lots["Lot Return %"] = np.where(lots["Cost Basis"] > 0, lots["Lot P&L"] / lots["Cost Basis"] * 100, np.nan)
    lots["Term"] = np.where(lots["Days Held"] >= 365, "Long", "Short")
    st.dataframe(lots[["Symbol", "Trade Date", "Quantity", "Purchase Price USD", "Current Price USD",
                       "Cost Basis", "Market Value", "Lot P&L", "Lot Return %", "Days Held", "Term"]]
                 .style.format({
                     "Trade Date": "{:%Y-%m-%d}", "Quantity": "{:.4f}",
                     "Purchase Price USD": "${:,.2f}", "Current Price USD": "${:,.2f}",
                     "Cost Basis": "${:,.2f}", "Market Value": "${:,.2f}",
                     "Lot P&L": "${:,.2f}", "Lot Return %": "{:.1f}%",
                 }), use_container_width=True)
    st.download_button("Download lot-level CSV", lots.to_csv(index=False), "lots.csv")

with tab5:
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
            if not new_symbol:
                st.error("Please enter a symbol.")
            else:
                new_row = {
                    "Symbol": new_symbol,
                    "Current Price": new_price,
                    "Date": datetime.now().strftime("%Y/%m/%d"),
                    "Time": "16:00 EST",
                    "Change": 0,
                    "Open": 0,
                    "High": 0,
                    "Low": 0,
                    "Volume": 0,
                    "Trade Date": new_date.strftime("%Y%m%d"),
                    "Purchase Price": new_price,
                    "Quantity": new_qty,
                    "Commission": 0,
                    "High Limit": "",
                    "Low Limit": "",
                    "Comment": "Added via App",
                    "Transaction Type": "BUY"
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
                except Exception as e:
                    st.error(f"Failed to add trade: {e}")

with tab6:
    st.subheader("Dividend History & Breakdown")
    
    if df_div.empty:
        st.info("No dividends logged yet. As your companies pay out (and you've owned them for the ex-dividend date), they will automatically appear here!")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Dividends Collected (USD)", f"${total_div_usd:,.2f}")
        c2.metric("Total Payouts Logged", len(df_div))
        
        st.write("### Total Cash by Security (USD)")
        div_by_symbol = df_div.groupby("Symbol")["Total Cash USD"].sum().reset_index().sort_values("Total Cash USD", ascending=False)
        fig_div = px.bar(div_by_symbol, x="Symbol", y="Total Cash USD", text_auto='.2s', color="Symbol")
        fig_div.update_layout(showlegend=False, yaxis_title="Cash Received (USD)")
        st.plotly_chart(fig_div, use_container_width=True)
        
        st.write("### Dividend Log")
        display_div = df_div.sort_values("Date", ascending=False).copy()
        st.dataframe(display_div.style.format({
            "Dividend per Share": "{:.4f}",
            "Total Cash": "${:,.2f}",
            "Total Cash USD": "${:,.2f}"
        }), use_container_width=True)

with tab7:
    st.subheader("Watchlist Tracker")
    
    # Form to add to watchlist
    with st.form("add_watchlist_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_watch_symbol = st.text_input("Add Ticker to Watchlist").upper().strip()
        with col2:
            st.write("")
            st.write("")
            submit_watch = st.form_submit_button("Add Ticker")
            
        if submit_watch and new_watch_symbol:
            try:
                client = get_gspread_client()
                sheet = client.open_by_key(SPREADSHEET_ID)
                try:
                    sheet_watch = sheet.worksheet("Watchlist")
                except:
                    st.error("Could not find 'Watchlist' tab in your Google Sheet. Please create it and add 'Symbol' to cell A1.")
                    st.stop()
                    
                existing_watch = pd.DataFrame(sheet_watch.get_all_records())
                if new_watch_symbol not in existing_watch["Symbol"].values:
                    new_watch_row = pd.DataFrame([{"Symbol": new_watch_symbol}])
                    existing_watch = pd.concat([existing_watch, new_watch_row], ignore_index=True)
                    set_with_dataframe(sheet_watch, existing_watch)
                    st.success(f"Added {new_watch_symbol} to Watchlist!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(f"{new_watch_symbol} is already on the watchlist.")
            except Exception as e:
                st.error(f"Error updating watchlist: {e}")

    if df_watch.empty:
        st.info("Watchlist is empty. Add a ticker above!")
    else:
        # Fetch data for all watchlist tickers
        watch_tickers = df_watch["Symbol"].tolist()
        
        @st.cache_data(ttl=300, show_spinner="Fetching watchlist data...")
        def fetch_watchlist_data(tickers):
            data = []
            for t in tickers:
                try:
                    ticker = yf.Ticker(t)
                    info = ticker.fast_info
                    
                    # YTD Return calculation
                    hist = ticker.history(period="ytd")
                    if not hist.empty:
                        ytd_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                    else:
                        ytd_ret = np.nan
                        
                    # Currency
                    curr = "CAD" if t.endswith(".TO") else "USD"
                    
                    data.append({
                        "Symbol": t,
                        "Price": info.get('last_price', np.nan),
                        "Daily Change (%)": (info.get('last_price', 0) / info.get('previous_close', 1) - 1) * 100,
                        "YTD Return (%)": ytd_ret,
                        "52W High": info.get('year_high', np.nan),
                        "52W Low": info.get('year_low', np.nan),
                        "Currency": curr
                    })
                except:
                    continue
            return pd.DataFrame(data)
            
        watch_data = fetch_watchlist_data(tuple(watch_tickers))
        
        if not watch_data.empty:
            st.dataframe(watch_data.style.format({
                "Price": "{:.2f}",
                "Daily Change (%)": "{:+.2f}%",
                "YTD Return (%)": "{:+.2f}%",
                "52W High": "{:.2f}",
                "52W Low": "{:.2f}"
            }), use_container_width=True)
