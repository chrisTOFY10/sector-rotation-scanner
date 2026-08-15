import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Sector Rotation Scanner", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Scanner Dashboard", "Sector Heatmap 📊", "How It Works (Metrics)"])
st.sidebar.divider()

# ==========================================
# PAGE 1: THE SCANNER DASHBOARD
# ==========================================
if page == "Scanner Dashboard":
    st.title("🔄 Sector Rotation & Relative Strength Scanner")
    st.markdown("Identify capital flows and find the strongest stocks within leading sectors.")

    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your TSX_Export file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        with st.spinner("Loading and processing data..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df.columns = df.columns.str.strip()
                df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by=['Symbol', 'Date'])
                
                trading_days = len(df['Date'].unique())
                latest_date = df['Date'].max()
                
                if trading_days < 3:
                    st.error(f"⚠️ **Error:** Your dataset only has {trading_days} day(s) of history. You need at least 3 days to calculate momentum.")
                    st.stop()
                    
                st.sidebar.success(f"Data loaded: {trading_days} trading days up to {latest_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                st.error(f"Error loading file. Error: {e}")
                st.stop()

        st.sidebar.header("2. Analysis Parameters")
        max_periods = trading_days - 1
        
        long_window = st.sidebar.slider("Long-Term Momentum (Days)", min_value=2, max_value=max_periods, value=min(5, max_periods))
        short_window = st.sidebar.slider("Short-Term Momentum (Days)", min_value=1, max_value=long_window-1, value=min(1, long_window-1))
        vol_window = st.sidebar.slider("Baseline Volume Avg (Days)", min_value=2, max_value=trading_days, value=min(5, trading_days))
        rs_window = st.sidebar.slider("Relative Strength Trend (Days)", min_value=2, max_value=trading_days, value=min(3, trading_days))

        # --- PHASE 1: SECTOR ROLL-UP ---
        st.header("Phase 1: Sector Rotation (Capital Flows)")
        
        sector_df = df.groupby(['Date', 'Sector']).agg({'Close': 'mean', 'Volume': 'sum'}).reset_index()
        sector_metrics = []
        sectors = sector_df['Sector'].unique()
        
        for sector in sectors:
            s_data = sector_df[sector_df['Sector'] == sector].sort_values('Date').copy()
            s_data['Short_Return'] = s_data['Close'].pct_change(periods=short_window) * 100
            s_data['Long_Return'] = s_data['Close'].pct_change(periods=long_window) * 100
            s_data['Vol_Baseline'] = s_data['Volume'].rolling(window=vol_window).mean()
            
            latest_data = s_data.iloc[-1]
            acceleration = latest_data['Short_Return'] - latest_data['Long_Return']
            vol_surge_ratio = latest_data['Volume'] / latest_data['Vol_Baseline'] if latest_data['Vol_Baseline'] > 0 else 0
            
            sector_metrics.append({
                'Sector': sector,
                f'{short_window}-Day Return (%)': round(latest_data['Short_Return'], 2),
                f'{long_window}-Day Return (%)': round(latest_data['Long_Return'], 2),
                'Momentum Acceleration': round(acceleration, 2),
                f'Volume Surge (Today vs {vol_window}D Avg)': round(vol_surge_ratio, 2)
            })

        sector_results = pd.DataFrame(sector_metrics).dropna().sort_values(by='Momentum Acceleration', ascending=False)
        st.dataframe(sector_results, use_container_width=True)
        st.divider()

        # --- PHASE 2: INDIVIDUAL STOCK SCREENER ---
        st.header("Phase 2: Stock Screener (Relative Strength & Breakouts)")
        selected_sector = st.selectbox("Select a Leading Sector to Scan:", options=sector_results['Sector'].tolist())
        
        if selected_sector:
            stock_df = df[df['Sector'] == selected_sector].copy()
            stock_metrics = []
            tickers = stock_df['Symbol'].unique()
            min_required = max(long_window + 1, vol_window, rs_window)
            
            with st.spinner(f"Scanning stocks in {selected_sector}..."):
                for ticker in tickers:
                    t_data = stock_df[stock_df['Symbol'] == ticker].sort_values('Date').copy()
                    if len(t_data) < min_required: continue 
                    
                    t_data = pd.merge(t_data, sector_df[sector_df['Sector'] == selected_sector][['Date', 'Close']], on='Date', suffixes=('', '_Sector'))
                    t_data['RS'] = t_data['Close'] / t_data['Close_Sector']
                    t_data['RS_MA'] = t_data['RS'].rolling(window=rs_window).mean()
                    t_data['Vol_Baseline'] = t_data['Volume'].rolling(window=vol_window).mean()
                    
                    latest_t = t_data.iloc[-1]
                    rs_trend = "Uptrend" if latest_t['RS'] > latest_t['RS_MA'] else "Downtrend"
                    vol_breakout = "Yes 🔥" if latest_t['Volume'] > (1.5 * latest_t['Vol_Baseline']) else "No"
                    
                    stock_metrics.append({
                        'Symbol': ticker,
                        'Latest Close': latest_t['Close'],
                        'RS vs Sector': rs_trend,
                        f'Daily Vol vs {vol_window}D Avg': round(latest_t['Volume'] / latest_t['Vol_Baseline'], 2) if latest_t['Vol_Baseline'] > 0 else 0,
                        'Volume Breakout (>150%)': vol_breakout
                    })
            
            if stock_metrics:
                final_stocks = pd.DataFrame(stock_metrics)
                strong_stocks = final_stocks[(final_stocks['RS vs Sector'] == 'Uptrend')].sort_values(by=f'Daily Vol vs {vol_window}D Avg', ascending=False)
                st.dataframe(strong_stocks, use_container_width=True)
            else:
                st.warning(f"No stocks in {selected_sector} have enough historical data.")
    else:
        st.info("👈 Please upload your TSX_Export file in the sidebar to begin.")


# ==========================================
# PAGE 2: SECTOR HEATMAP
# ==========================================
elif page == "Sector Heatmap 📊":
    st.title("📊 Market Heatmap")
    st.markdown("Visualize capital flow and performance across all sectors simultaneously. Block size represents **Volume**, while color represents **Percentage Return**.")

    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your TSX_Export file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        with st.spinner("Generating heatmap..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df.columns = df.columns.str.strip()
                df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by=['Symbol', 'Date'])
                
                trading_days = len(df['Date'].unique())
                if trading_days < 2:
                    st.error("⚠️ Not enough data. You need at least 2 days to calculate percentage returns.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error loading file. Error: {e}")
                st.stop()

        st.sidebar.header("2. Heatmap Settings")
        max_map_periods = trading_days - 1
        map_window = st.sidebar.slider("Heatmap Timeframe (Days to measure Return)", min_value=1, max_value=max_map_periods, value=1)
        
        color_sensitivity = st.sidebar.slider("Color Sensitivity (Max Return %)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
        st.sidebar.caption("💡 **Tip:** Lower this number to make the big Sector blocks pop. Raise it to highlight extreme individual stock moves.")

        df['Return (%)'] = df.groupby('Symbol')['Close'].pct_change(periods=map_window) * 100
        latest_date = df['Date'].max()
        latest_df = df[df['Date'] == latest_date].copy()
        
        latest_df = latest_df.dropna(subset=['Return (%)'])
        latest_df = latest_df[latest_df['Volume'] > 0] 
        
        latest_df['Sector'] = latest_df['Sector'].astype(str).replace(['nan', 'None', ''], 'Unclassified')
        latest_df['Symbol'] = latest_df['Symbol'].astype(str).replace(['nan', 'None', ''], 'Unknown')
        latest_df = latest_df.drop_duplicates(subset=['Symbol'], keep='last')

        if not latest_df.empty:
            fig = px.treemap(
                latest_df,
                path=[px.Constant("Overall Market"), 'Sector', 'Symbol'],
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
            st.warning("No data available to plot for the selected timeframe.")
    else:
        st.info("👈 Please upload your TSX_Export file in the sidebar to generate the heatmap.")


# ==========================================
# PAGE 3: HOW IT WORKS (METRICS GUIDE)
# ==========================================
elif page == "How It Works (Metrics)":
    st.title("📚 How It Works: A Trader's Guide to the Data")
    st.markdown("Trading isn't about predicting the future; it's about following the footprints of big money. This guide breaks down the core calculations powering the scanner so you can read the market's story with confidence.")
    st.divider()

    # --- 1. Momentum Acceleration ---
    st.header("1. Momentum Acceleration")
    st.markdown("**The Concept:** This measures whether a sector's trend is speeding up or running out of gas. A sector might be up overall, but if its recent gains are shrinking, the trend is dying.")
    st.markdown("**🚗 The Novice Analogy:** Imagine you are driving a car. Your *Long-Term Return* is your average speed on the highway (60 mph). Your *Short-Term Return* is pressing the gas pedal to pass someone (80 mph). This metric measures how hard you are pressing the gas right now.")
    st.latex(r"Acceleration = R_{short} - R_{long}")
    st.success("**💡 Real-World Insight:** If the Energy sector is up 5% over the long-term, but up 8% in the short-term, its acceleration is **+3.00**. This tells you new buyers are aggressively stepping in *right now*. You want to look for positive numbers at the top of the Phase 1 chart.")
    st.write("---")

    # --- 2. Volume Surge ---
    st.header("2. Volume Surge (Sector Level)")
    st.markdown("**The Concept:** Price moves can be faked by a lack of liquidity, but massive trading volume cannot be faked. This metric measures the influx of broad capital into an entire sector relative to its normal baseline.")
    st.markdown("**👣 The Novice Analogy:** Think of volume like footprints in the snow. A few retail traders leave a light trail. But when hedge funds and mutual funds buy into a sector, they march an entire army through the snow. This metric looks for the army.")
    st.latex(r"\text{Sector Surge} = \frac{V_{current}}{\overline{V}_{baseline}}")
    st.success("**💡 Real-World Insight:** A ratio of **1.50** means today's sector volume is 50% higher than its recent average. If a sector is accelerating in price *and* has a volume ratio above 1.0, the move is legitimate. Institutions are actively buying.")
    st.write("---")

    # --- 3. RS vs Sector ---
    st.header("3. RS vs Sector (Relative Strength)")
    st.markdown("**The Concept:** This evaluates a stock's performance directly against its own sector peers rather than the broad market.")
    st.markdown("**🏃‍♂️ The Novice Analogy:** Imagine a track team running a race with a heavy tailwind. The whole team will run faster times (the sector is going up). But Relative Strength asks: *Who is the fastest runner on the team?* You don't just want a stock that is being dragged up by the wind; you want the strongest athlete.")
    st.latex(r"RS = \frac{P_{stock}}{P_{sector}}")
    st.success("**💡 Real-World Insight:** If the scanner shows **'Uptrend'**, it means this specific stock is gaining more on green days and losing less on red days compared to its sector peers. This is your prime target.")
    st.write("---")

    # --- 4. Daily Vol vs Baseline Avg ---
    st.header("4. Daily Vol vs Baseline Avg (Stock Level)")
    st.markdown("**The Concept:** This looks at an individual stock's trading activity against its own normal behavior, isolating it from the broader market.")
    st.markdown("**🗣️ The Novice Analogy:** Imagine a normally quiet library where suddenly 50 people start having a loud conversation. You instantly know something specific is happening at that exact table.")
    st.latex(r"\text{Stock Volume Ratio} = \frac{V_{daily}}{\overline{V}_{average}}")
    st.success("**💡 Real-World Insight:** If a stock normally trades 1 million shares but today traded 3 million, something fundamental (news, earnings, a big fund buying) just attracted eyeballs. It confirms the stock has its own catalyst.")
    st.write("---")

    # --- 5. Volume Breakout ---
    st.header("5. Volume Breakout (>150%)")
    st.markdown("**The Concept:** This is a strict, binary filter that flags stocks experiencing extreme, undeniable buying pressure.")
    st.markdown("**🚨 The Novice Analogy:** This is your smoke alarm. It only rings when the volume exceeds 1.5 times the normal limits.")
    st.latex(r"\text{Breakout Trigger} = V_{daily} > (1.5 \times \overline{V}_{average})")
    st.success("**💡 Real-World Insight:** If the tool flags **'Yes 🔥'**, it is highly probable that 'smart money' (institutional algorithms) is accumulating shares. These stocks are often the safest entries for swing trades because big money is actively supporting the price.")