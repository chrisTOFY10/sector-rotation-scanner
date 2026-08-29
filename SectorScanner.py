import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Market Rotation Scanner", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Scanner Dashboard", "Market Heatmap 📊", "How It Works (Metrics)"])
st.sidebar.divider()

# --- AUTOMATED DATA FEED ---
@st.cache_data(ttl=3600) 
def load_google_sheet():
    # MAKE SURE YOUR GOOGLE SHEET EXPORT URL IS PASTED HERE
    sheet_url = "https://docs.google.com/spreadsheets/d/1Dgu6N8rwsa49JwBGNWBN5xN3-fzeEOnyYKoUXvsPNlg/export?format=csv"
    return pd.read_csv(sheet_url)

# ==========================================
# PAGE 1: THE SCANNER DASHBOARD
# ==========================================
if page == "Scanner Dashboard":
    st.title("🔄 Market Rotation & Relative Strength Scanner")
    st.markdown("Identify capital flows and find the strongest stocks within leading groups.")

    st.sidebar.header("1. Data Feed")
    st.sidebar.success("🟢 Connected to Live Google Sheet")

    st.sidebar.header("2. Scan Level")
    grouping_level = st.sidebar.radio("Group stocks by:", ["Sector", "Industry"])

    with st.spinner("Fetching latest market data from Google Sheets..."):
        try:
            df = load_google_sheet()
            df.columns = df.columns.str.strip()
            
            # --- DATA CLEANING (Added Open for Volume Flow calculation) ---
            cols_to_clean = ['Open', 'Close', 'Volume']
            for col in cols_to_clean:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            
            df['Date'] = pd.to_datetime(df['Date'])
            has_open = 'Open' in df.columns
            
            if 'Exchange' in df.columns:
                df['Exchange'] = df['Exchange'].astype(str).str.strip()
            if 'Sector' in df.columns:
                df['Sector'] = df['Sector'].astype(str).str.strip()
            if 'Industry' in df.columns:
                df['Industry'] = df['Industry'].astype(str).str.strip()
                
            if grouping_level not in df.columns:
                st.error(f"⚠️ The column '{grouping_level}' was not found in your data.")
                st.stop()
            
            if 'Exchange' in df.columns:
                available_exchanges = sorted(df[df['Exchange'] != 'nan']['Exchange'].unique().tolist())
                selected_exchanges = st.sidebar.multiselect("Select Exchange(s):", options=available_exchanges, default=available_exchanges)
                if not selected_exchanges:
                    st.warning("⚠️ Please select at least one exchange in the sidebar to run the scan.")
                    st.stop()
                df = df[df['Exchange'].isin(selected_exchanges)]
            
            df = df.sort_values(by=['Symbol', 'Date'])
            trading_days = len(df['Date'].unique())
            latest_date = df['Date'].max()
            
            if trading_days < 3:
                st.error(f"⚠️ **Error:** Your dataset only has {trading_days} day(s) of history. You need at least 3 days.")
                st.stop()
                
            st.sidebar.caption(f"Last updated: {latest_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"Failed to load data. Error: {e}")
            st.stop()

        st.sidebar.header("3. Analysis Parameters")
        max_periods = trading_days - 1
        
        long_window = st.sidebar.slider("Long-Term Momentum (Days)", min_value=2, max_value=max_periods, value=min(5, max_periods))
        short_window = st.sidebar.slider("Short-Term Momentum (Days)", min_value=1, max_value=long_window-1, value=min(1, long_window-1))
        vol_window = st.sidebar.slider("Baseline Volume Avg (Days)", min_value=2, max_value=trading_days, value=min(5, trading_days))
        rs_window = st.sidebar.slider("Relative Strength Trend (Days)", min_value=2, max_value=trading_days, value=min(3, trading_days))

        # --- PHASE 1: MACRO ROLL-UP ---
        st.header(f"Phase 1: {grouping_level} Rotation (Capital Flows)")
        
        group_df = df.groupby(['Date', grouping_level]).agg({'Close': 'mean', 'Volume': 'sum'}).reset_index()
        group_metrics = []
        groups = group_df[grouping_level].unique()
        
        for grp in groups:
            g_data = group_df[group_df[grouping_level] == grp].sort_values('Date').copy()
            g_data['Short_Return'] = g_data['Close'].pct_change(periods=short_window) * 100
            g_data['Long_Return'] = g_data['Close'].pct_change(periods=long_window) * 100
            g_data['Vol_Baseline'] = g_data['Volume'].rolling(window=vol_window).mean()
            
            latest_data = g_data.iloc[-1]
            acceleration = latest_data['Short_Return'] - latest_data['Long_Return']
            vol_surge_ratio = latest_data['Volume'] / latest_data['Vol_Baseline'] if latest_data['Vol_Baseline'] > 0 else 0
            
            group_metrics.append({
                grouping_level: grp,
                f'{short_window}-Day Return (%)': round(latest_data['Short_Return'], 2),
                f'{long_window}-Day Return (%)': round(latest_data['Long_Return'], 2),
                'Momentum Acceleration': round(acceleration, 2),
                f'Volume Surge (Today vs {vol_window}D Avg)': round(vol_surge_ratio, 2)
            })

        group_results = pd.DataFrame(group_metrics).dropna().sort_values(by='Momentum Acceleration', ascending=False)
        st.dataframe(group_results, use_container_width=True)
        st.divider()

        # --- PHASE 2 & 3: STOCK SCREENER WITH VOLUME FLOW ---
        st.header(f"Phase 2: Stock Screener (Relative Strength, Breakouts, & Accumulation)")
        selected_group = st.selectbox(f"Select a Leading {grouping_level} to Scan:", options=group_results[grouping_level].tolist())
        
        if selected_group:
            stock_df = df[df[grouping_level] == selected_group].copy()
            stock_metrics = []
            tickers = stock_df['Symbol'].unique()
            min_required = max(long_window + 1, vol_window, rs_window, 10) # Added 10 for volume flow
            
            with st.spinner(f"Scanning stocks in {selected_group}..."):
                for ticker in tickers:
                    t_data = stock_df[stock_df['Symbol'] == ticker].sort_values('Date').copy()
                    if len(t_data) < min_required: continue 
                    
                    t_data = pd.merge(t_data, group_df[group_df[grouping_level] == selected_group][['Date', 'Close']], on='Date', suffixes=('', '_Group'))
                    t_data['RS'] = t_data['Close'] / t_data['Close_Group']
                    t_data['RS_MA'] = t_data['RS'].rolling(window=rs_window).mean()
                    t_data['Vol_Baseline'] = t_data['Volume'].rolling(window=vol_window).mean()
                    
                    # --- PHASE 3: VOLUME-WEIGHTED TREND ---
                    if has_open:
                        t_data['Green_Vol'] = np.where(t_data['Close'] > t_data['Open'], t_data['Volume'], 0)
                        t_data['Red_Vol'] = np.where(t_data['Close'] < t_data['Open'], t_data['Volume'], 0)
                        t_data['Green_Vol_10D'] = t_data['Green_Vol'].rolling(window=10).sum()
                        t_data['Red_Vol_10D'] = t_data['Red_Vol'].rolling(window=10).sum()
                        t_data['Vol_Flow_Ratio'] = np.where(t_data['Red_Vol_10D'] > 0, t_data['Green_Vol_10D'] / t_data['Red_Vol_10D'], np.nan)
                    
                    latest_t = t_data.iloc[-1]
                    rs_trend = "Uptrend" if latest_t['RS'] > latest_t['RS_MA'] else "Downtrend"
                    vol_breakout = "Yes 🔥" if latest_t['Volume'] > (1.5 * latest_t['Vol_Baseline']) else "No"
                    
                    if has_open:
                        flow_ratio = latest_t['Vol_Flow_Ratio']
                        if pd.isna(flow_ratio):
                            flow_status = "N/A"
                        else:
                            flow_status = f"{round(flow_ratio, 2)}x (Buy > Sell)" if flow_ratio > 1.0 else f"{round(flow_ratio, 2)}x (Sell > Buy)"
                    else:
                        flow_status = "N/A (Missing Open data)"
                    
                    stock_metrics.append({
                        'Symbol': ticker,
                        'Exchange': latest_t['Exchange'] if 'Exchange' in latest_t else 'N/A',
                        'Latest Close': latest_t['Close'],
                        f'RS vs {grouping_level}': rs_trend,
                        'Volume Breakout (>150%)': vol_breakout,
                        '10-Day Vol Flow': flow_status
                    })
            
            if stock_metrics:
                final_stocks = pd.DataFrame(stock_metrics)
                # Filter for strongest candidates only
                strong_stocks = final_stocks[(final_stocks[f'RS vs {grouping_level}'] == 'Uptrend') & (final_stocks['Volume Breakout (>150%)'] == 'Yes 🔥')].copy()
                
                # If the filter removes everything, just show all uptrend stocks
                if strong_stocks.empty:
                    strong_stocks = final_stocks[(final_stocks[f'RS vs {grouping_level}'] == 'Uptrend')].copy()
                    st.write(f"No volume breakouts today. Showing all outperforming stocks in **{selected_group}**:")
                else:
                    st.write(f"🔥 Showing outperforming stocks in **{selected_group}** with Active Volume Breakouts:")
                    
                st.dataframe(strong_stocks, use_container_width=True)
            else:
                st.warning(f"No stocks in {selected_group} have enough historical data.")


# ==========================================
# PAGE 2: MARKET HEATMAP
# ==========================================
elif page == "Market Heatmap 📊":
    st.title("📊 Market Heatmap")
    st.markdown("Visualize capital flow and performance across the market simultaneously.")

    st.sidebar.header("1. Data Feed")
    st.sidebar.success("🟢 Connected to Live Google Sheet")

    st.sidebar.header("2. Map Level")
    grouping_level = st.sidebar.radio("Group heatmap by:", ["Sector", "Industry"])

    with st.spinner("Generating heatmap..."):
        try:
            df = load_google_sheet()
            df.columns = df.columns.str.strip()
            df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'])
            
            if grouping_level not in df.columns:
                st.error(f"⚠️ The column '{grouping_level}' was not found in your data.")
                st.stop()
                
            if 'Exchange' in df.columns:
                df['Exchange'] = df['Exchange'].astype(str).str.strip()
                available_exchanges = sorted(df[df['Exchange'] != 'nan']['Exchange'].unique().tolist())
                selected_exchanges = st.sidebar.multiselect("Select Exchange(s):", options=available_exchanges, default=available_exchanges)
                if not selected_exchanges:
                    st.warning("⚠️ Please select at least one exchange in the sidebar to generate the heatmap.")
                    st.stop()
                df = df[df['Exchange'].isin(selected_exchanges)]
            
            df = df.sort_values(by=['Symbol', 'Date'])
            trading_days = len(df['Date'].unique())
            if trading_days < 2:
                st.error("⚠️ Not enough data for the selected exchanges.")
                st.stop()
                
        except Exception as e:
            st.error(f"Failed to load data. Error: {e}")
            st.stop()

    st.sidebar.header("3. Heatmap Settings")
    max_map_periods = trading_days - 1
    map_window = st.sidebar.slider("Heatmap Timeframe (Days to measure Return)", min_value=1, max_value=max_map_periods, value=1)
    color_sensitivity = st.sidebar.slider("Color Sensitivity (Max Return %)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)

    df['Return (%)'] = df.groupby('Symbol')['Close'].pct_change(periods=map_window) * 100
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    
    latest_df = latest_df.replace([np.inf, -np.inf], np.nan)
    latest_df = latest_df.dropna(subset=['Return (%)', 'Volume', 'Symbol', grouping_level])
    latest_df = latest_df[latest_df['Volume'] > 0] 
    latest_df = latest_df.reset_index(drop=True)
    
    latest_df['Symbol'] = latest_df['Symbol'].astype(str).str.strip()
    latest_df = latest_df[latest_df['Symbol'] != '']
    latest_df = latest_df[latest_df['Symbol'].str.lower() != 'nan']
    
    latest_df[grouping_level] = latest_df[grouping_level].astype(str).str.strip()
    latest_df.loc[latest_df[grouping_level] == '', grouping_level] = 'Unclassified'
    latest_df.loc[latest_df[grouping_level].str.lower() == 'nan', grouping_level] = 'Unclassified'

    latest_df['Root'] = 'Overall Market'
    latest_df[grouping_level] = latest_df[grouping_level] + " "
    latest_df['Symbol'] = latest_df['Symbol'] + "  "
    latest_df = latest_df.drop_duplicates(subset=['Symbol'], keep='last')

    if not latest_df.empty:
        fig = px.treemap(
            latest_df,
            path=['Root', grouping_level, 'Symbol'],
            values='Volume',
            color='Return (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,  
            range_color=[-color_sensitivity, color_sensitivity], 
            hover_data={'Close': ':.2f', 'Volume': ':.0f', 'Return (%)': ':.2f'}
        )
        fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=700, coloraxis_colorbar=dict(title=f"{map_window}-Day Return (%)"))
        fig.update_traces(textposition='middle center', textinfo="label+text+value", hovertemplate="<b>%{label}</b><br>Return: %{color:.2f}%<br>Volume: %{value:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Viewing Date:** {latest_date.strftime('%Y-%m-%d')}")
    else:
        st.warning("No data available to plot.")

# ==========================================
# PAGE 3: HOW IT WORKS (METRICS GUIDE)
# ==========================================
elif page == "How It Works (Metrics)":
    st.title("📚 How It Works: A Trader's Guide to the Data")
    st.divider()

    st.header("1. Momentum Acceleration")
    st.markdown("**The Concept:** Measures whether a trend is speeding up or running out of gas.")
    st.latex(r"Acceleration = R_{short} - R_{long}")
    st.write("---")

    st.header("2. Macro Volume Surge")
    st.markdown("**The Concept:** Measures the influx of broad capital into an entire group relative to its normal baseline.")
    st.latex(r"\text{Macro Surge} = \frac{V_{current}}{\overline{V}_{baseline}}")
    st.write("---")

    st.header("3. RS vs Group (Relative Strength)")
    st.markdown("**The Concept:** Evaluates a stock's performance directly against its peers.")
    st.latex(r"RS = \frac{P_{stock}}{P_{group}}")
    st.write("---")
    
    st.header("4. Volume-Weighted Trend (10-Day Volume Flow)")
    st.markdown("**The Concept:** Directly pulled from your **Technical Analysis** file concepts, this metric compares total trading volume on up-days versus down-days to expose hidden institutional accumulation or distribution over the last 10 days.")
    st.latex(r"\text{Flow} = \frac{\sum V_{green\_days}}{\sum V_{red\_days}}")
