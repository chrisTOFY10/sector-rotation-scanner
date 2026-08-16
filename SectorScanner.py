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

    # --- THE NEW SCAN LEVEL TOGGLE ---
    st.sidebar.header("2. Scan Level")
    grouping_level = st.sidebar.radio("Group stocks by:", ["Sector", "Industry"])
    st.sidebar.caption(f"Currently scanning at the **{grouping_level}** level.")

    with st.spinner("Fetching latest market data from Google Sheets..."):
        try:
            df = load_google_sheet()
            
            # --- DATA CLEANING ---
            df.columns = df.columns.str.strip()
            df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Clean textual columns
            if 'Exchange' in df.columns:
                df['Exchange'] = df['Exchange'].astype(str).str.strip()
            if 'Sector' in df.columns:
                df['Sector'] = df['Sector'].astype(str).str.strip()
            if 'Industry' in df.columns:
                df['Industry'] = df['Industry'].astype(str).str.strip()
                
            # Verify the chosen grouping column actually exists in the data
            if grouping_level not in df.columns:
                st.error(f"⚠️ The column '{grouping_level}' was not found in your data. Please check your data source.")
                st.stop()
            
            # --- EXCHANGE FILTER ---
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
                st.error(f"⚠️ **Error:** Your dataset only has {trading_days} day(s) of history for the selected exchanges. You need at least 3 days to calculate momentum.")
                st.stop()
                
            st.sidebar.caption(f"Last updated: {latest_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"Failed to load data. Error: {e}")
            st.stop()

        # --- SIDEBAR: DYNAMIC TIMEFRAMES ---
        st.sidebar.header("3. Analysis Parameters")
        max_periods = trading_days - 1
        
        long_window = st.sidebar.slider("Long-Term Momentum (Days)", min_value=2, max_value=max_periods, value=min(5, max_periods))
        short_window = st.sidebar.slider("Short-Term Momentum (Days)", min_value=1, max_value=long_window-1, value=min(1, long_window-1))
        vol_window = st.sidebar.slider("Baseline Volume Avg (Days)", min_value=2, max_value=trading_days, value=min(5, trading_days))
        rs_window = st.sidebar.slider("Relative Strength Trend (Days)", min_value=2, max_value=trading_days, value=min(3, trading_days))

        # --- PHASE 1: MACRO ROLL-UP ---
        st.header(f"Phase 1: {grouping_level} Rotation (Capital Flows)")
        
        # Dynamically group by whatever the user selected (Sector or Industry)
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

        # --- PHASE 2: INDIVIDUAL STOCK SCREENER ---
        st.header(f"Phase 2: Stock Screener (Relative Strength & Breakouts)")
        selected_group = st.selectbox(f"Select a Leading {grouping_level} to Scan:", options=group_results[grouping_level].tolist())
        
        if selected_group:
            stock_df = df[df[grouping_level] == selected_group].copy()
            stock_metrics = []
            tickers = stock_df['Symbol'].unique()
            min_required = max(long_window + 1, vol_window, rs_window)
            
            with st.spinner(f"Scanning stocks in {selected_group}..."):
                for ticker in tickers:
                    t_data = stock_df[stock_df['Symbol'] == ticker].sort_values('Date').copy()
                    if len(t_data) < min_required: continue 
                    
                    # Merge with the dynamically generated group index to calculate RS
                    t_data = pd.merge(t_data, group_df[group_df[grouping_level] == selected_group][['Date', 'Close']], on='Date', suffixes=('', '_Group'))
                    t_data['RS'] = t_data['Close'] / t_data['Close_Group']
                    t_data['RS_MA'] = t_data['RS'].rolling(window=rs_window).mean()
                    t_data['Vol_Baseline'] = t_data['Volume'].rolling(window=vol_window).mean()
                    
                    latest_t = t_data.iloc[-1]
                    rs_trend = "Uptrend" if latest_t['RS'] > latest_t['RS_MA'] else "Downtrend"
                    vol_breakout = "Yes 🔥" if latest_t['Volume'] > (1.5 * latest_t['Vol_Baseline']) else "No"
                    
                    stock_metrics.append({
                        'Symbol': ticker,
                        'Exchange': latest_t['Exchange'] if 'Exchange' in latest_t else 'N/A',
                        'Latest Close': latest_t['Close'],
                        f'RS vs {grouping_level}': rs_trend,
                        f'Daily Vol vs {vol_window}D Avg': round(latest_t['Volume'] / latest_t['Vol_Baseline'], 2) if latest_t['Vol_Baseline'] > 0 else 0,
                        'Volume Breakout (>150%)': vol_breakout
                    })
            
            if stock_metrics:
                final_stocks = pd.DataFrame(stock_metrics)
                strong_stocks = final_stocks[(final_stocks[f'RS vs {grouping_level}'] == 'Uptrend')].sort_values(by=f'Daily Vol vs {vol_window}D Avg', ascending=False)
                
                cols = ['Symbol', 'Exchange', 'Latest Close', f'RS vs {grouping_level}', f'Daily Vol vs {vol_window}D Avg', 'Volume Breakout (>150%)']
                strong_stocks = strong_stocks[[c for c in cols if c in strong_stocks.columns]]
                
                st.dataframe(strong_stocks, use_container_width=True)
            else:
                st.warning(f"No stocks in {selected_group} have enough historical data.")


# ==========================================
# PAGE 2: MARKET HEATMAP
# ==========================================
elif page == "Market Heatmap 📊":
    st.title("📊 Market Heatmap")
    st.markdown("Visualize capital flow and performance across the market simultaneously. Block size represents **Volume**, while color represents **Percentage Return**.")

    st.sidebar.header("1. Data Feed")
    st.sidebar.success("🟢 Connected to Live Google Sheet")

    # --- HEATMAP SCAN LEVEL TOGGLE ---
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
                st.error("⚠️ Not enough data for the selected exchanges. You need at least 2 days to calculate percentage returns.")
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
    
    latest_df = latest_df.dropna(subset=['Return (%)'])
    latest_df = latest_df[latest_df['Volume'] > 0] 
    
    latest_df[grouping_level] = latest_df[grouping_level].astype(str).replace(['nan', 'None', ''], 'Unclassified')
    latest_df['Symbol'] = latest_df['Symbol'].astype(str).replace(['nan', 'None', ''], 'Unknown')
    latest_df = latest_df.drop_duplicates(subset=['Symbol'], keep='last')

    if not latest_df.empty:
        # Dynamically map the path based on user toggle
        fig = px.treemap(
            latest_df,
            path=[px.Constant("Overall Market"), grouping_level, 'Symbol'],
            values='Volume',
            color='Return (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,  
            range_color=[-color_sensitivity, color_sensitivity], 
            hover_data={'Close': ':.2f', 'Volume': ':.0f', 'Return (%)': ':.2f'}
        )

        fig.update_layout(
            margin=dict(t=30, l=10, r=10, b=10),
            height=700,
            coloraxis_colorbar=dict(title=f"{map_window}-Day Return (%)")
        )
        
        fig.update_traces(
            textposition='middle center',
            textinfo="label+text+value",
            hovertemplate="<b>%{label}</b><br>Return: %{color:.2f}%<br>Volume: %{value:,.0f}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Viewing Date:** {latest_date.strftime('%Y-%m-%d')} | **Comparing against:** {map_window} trading day(s) prior.")
    else:
        st.warning("No data available to plot for the selected timeframe and exchanges.")


# ==========================================
# PAGE 3: HOW IT WORKS (METRICS GUIDE)
# ==========================================
elif page == "How It Works (Metrics)":
    st.title("📚 How It Works: A Trader's Guide to the Data")
    st.markdown("Trading isn't about predicting the future; it's about following the footprints of big money. This guide breaks down the core calculations powering the scanner so you can read the market's story with confidence.")
    st.divider()

    st.header("1. Momentum Acceleration")
    st.markdown("**The Concept:** This measures whether a sector/industry's trend is speeding up or running out of gas. A group might be up overall, but if its recent gains are shrinking, the trend is dying.")
    st.markdown("**🚗 The Novice Analogy:** Imagine you are driving a car. Your *Long-Term Return* is your average speed on the highway (60 mph). Your *Short-Term Return* is pressing the gas pedal to pass someone (80 mph). This metric measures how hard you are pressing the gas right now.")
    st.latex(r"Acceleration = R_{short} - R_{long}")
    st.success("**💡 Real-World Insight:** If a group is up 5% over the long-term, but up 8% in the short-term, its acceleration is **+3.00**. This tells you new buyers are aggressively stepping in *right now*. You want to look for positive numbers at the top of the Phase 1 chart.")
    st.write("---")

    st.header("2. Volume Surge (Macro Level)")
    st.markdown("**The Concept:** Price moves can be faked by a lack of liquidity, but massive trading volume cannot be faked. This metric measures the influx of broad capital into an entire sector or industry relative to its normal baseline.")
    st.markdown("**👣 The Novice Analogy:** Think of volume like footprints in the snow. A few retail traders leave a light trail. But when hedge funds and mutual funds buy into a sector, they march an entire army through the snow. This metric looks for the army.")
    st.latex(r"\text{Macro Surge} = \frac{V_{current}}{\overline{V}_{baseline}}")
    st.success("**💡 Real-World Insight:** A ratio of **1.50** means today's macro volume is 50% higher than its recent average. If a group is accelerating in price *and* has a volume ratio above 1.0, the move is legitimate. Institutions are actively buying.")
    st.write("---")

    st.header("3. RS vs Group (Relative Strength)")
    st.markdown("**The Concept:** This evaluates a stock's performance directly against its own peers rather than the broad market.")
    st.markdown("**🏃‍♂️ The Novice Analogy:** Imagine a track team running a race with a heavy tailwind. The whole team will run faster times (the sector is going up). But Relative Strength asks: *Who is the fastest runner on the team?* You don't just want a stock that is being dragged up by the wind; you want the strongest athlete.")
    st.latex(r"RS = \frac{P_{stock}}{P_{group}}")
    st.success("**💡 Real-World Insight:** If the scanner shows **'Uptrend'**, it means this specific stock is gaining more on green days and losing less on red days compared to its peers. This is your prime target.")
    st.write("---")

    st.header("4. Daily Vol vs Baseline Avg (Stock Level)")
    st.markdown("**The Concept:** This looks at an individual stock's trading activity against its own normal behavior, isolating it from the broader market.")
    st.markdown("**🗣️ The Novice Analogy:** Imagine a normally quiet library where suddenly 50 people start having a loud conversation. You instantly know something specific is happening at that exact table.")
    st.latex(r"\text{Stock Volume Ratio} = \frac{V_{daily}}{\overline{V}_{average}}")
    st.success("**💡 Real-World Insight:** If a stock normally trades 1 million shares but today traded 3 million, something fundamental (news, earnings, a big fund buying) just attracted eyeballs. It confirms the stock has its own catalyst.")
    st.write("---")

    st.header("5. Volume Breakout (>150%)")
    st.markdown("**The Concept:** This is a strict, binary filter that flags stocks experiencing extreme, undeniable buying pressure.")
    st.markdown("**🚨 The Novice Analogy:** This is your smoke alarm. It only rings when the volume exceeds 1.5 times the normal limits.")
    st.latex(r"\text{Breakout Trigger} = V_{daily} > (1.5 \times \overline{V}_{average})")
    st.success("**💡 Real-World Insight:** If the tool flags **'Yes 🔥'**, it is highly probable that 'smart money' (institutional algorithms) is accumulating shares. These stocks are often the safest entries for swing trades because big money is actively supporting the price.")