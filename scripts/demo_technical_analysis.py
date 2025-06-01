import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from financial_analysis import FinancialAnalyzer

# Set the style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def analyze_stock(ticker='AAPL', period='1y'):
    """
    Perform technical analysis on a given stock ticker.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Time period to analyze (default: '1y')
    """
    print(f"Analyzing {ticker} for the last {period}...\n")
    
    # Initialize the financial analyzer
    analyzer = FinancialAnalyzer(ticker=ticker, period=period)
    
    # Fetch the stock data
    print("Fetching stock data...")
    data = analyzer.fetch_data()
    
    if data is not None:
        # Calculate technical indicators
        print("Calculating technical indicators...")
        analyzer.calculate_indicators()
        
        # Get the technical summary
        summary = analyzer.get_technical_summary()
        
        # Print the summary
        print("\n" + "="*50)
        print(f"TECHNICAL ANALYSIS SUMMARY - {ticker}")
        print("="*50)
        
        # Price information
        print("\nPRICE INFORMATION:")
        print("-" * 30)
        print(f"Current Price: ${summary['price']['close']:.2f}")
        print(f"Change: {summary['price']['change']:+.2f} ({summary['price']['change_pct']:+.2f}%)")
        
        # Moving Averages
        print("\nMOVING AVERAGES:")
        print("-" * 30)
        print(f"Price vs SMA(20): {summary['moving_averages']['price_vs_sma20']:+.2f}%")
        print(f"Price vs SMA(50): {summary['moving_averages']['price_vs_sma50']:+.2f}%")
        print(f"Price vs SMA(200): {summary['moving_averages']['price_vs_sma200']:+.2f}%")
        print(f"SMA Cross: {summary['moving_averages']['sma_cross']}")
        
        # Bollinger Bands
        print("\nBOLLINGER BANDS:")
        print("-" * 30)
        print(f"%B: {summary['bollinger_bands']['bb_percent']:.2f}")
        print(f"Position: {summary['bollinger_bands']['position']}")
        
        # RSI
        print("\nRELATIVE STRENGTH INDEX (RSI):")
        print("-" * 30)
        print(f"RSI(14): {summary['rsi']['value']:.2f}")
        print(f"Signal: {summary['rsi']['signal']}")
        
        # MACD
        print("\nMOVING AVERAGE CONVERGENCE DIVERGENCE (MACD):")
        print("-" * 30)
        print(f"MACD: {summary['macd']['value']:.4f}")
        print(f"Signal Line: {summary['macd']['signal']:.4f}")
        print(f"Histogram: {summary['macd']['histogram']:+.4f}")
        print(f"Trend: {summary['macd']['trend']}")
        
        # VWAP
        print("\nVOLUME WEIGHTED AVERAGE PRICE (VWAP):")
        print("-" * 30)
        print(f"VWAP: {summary['vwap']['value']:.2f}")
        print(f"Price vs VWAP: {summary['vwap']['price_vs_vwap']:+.2f}%")
        
        # Generate the technical analysis chart
        print("\nGenerating technical analysis chart...")
        analyzer.plot_technical_analysis()
        
        # Additional analysis and visualization
        plot_additional_analysis(analyzer)
    else:
        print(f"Failed to fetch data for {ticker}. Please check the ticker symbol and try again.")

def plot_additional_analysis(analyzer):
    """
    Generate additional analysis plots.
    
    Args:
        analyzer (FinancialAnalyzer): Initialized FinancialAnalyzer instance
    """
    data = analyzer.data
    ticker = analyzer.ticker
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot price and moving averages
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data['SMA_20'], label='20-day SMA', color='orange', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='50-day SMA', color='red', alpha=0.7)
    ax1.plot(data.index, data['SMA_200'], label='200-day SMA', color='purple', alpha=0.7)
    
    ax1.set_title(f'{ticker} - Price and Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.7, label='Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Add a second y-axis for price
    ax2_vol = ax2.twinx()
    ax2_vol.plot(data.index, data['VWAP'], color='green', label='VWAP')
    ax2_vol.set_ylabel('VWAP')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_vol.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Create a new figure for RSI and MACD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot RSI
    ax1.plot(data.index, data['RSI'], label='RSI(14)', color='purple')
    ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(data.index, 70, data['RSI'], where=(data['RSI'] >= 70), 
                     color='red', alpha=0.2, label='Overbought')
    ax1.fill_between(data.index, 30, data['RSI'], where=(data['RSI'] <= 30), 
                     color='green', alpha=0.2, label='Oversold')
    ax1.set_title(f'{ticker} - Relative Strength Index (RSI)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MACD
    ax2.plot(data.index, data['MACD_12_26_9'], label='MACD', color='blue')
    ax2.plot(data.index, data['MACDs_12_26_9'], label='Signal Line', color='orange')
    ax2.bar(data.index, data['MACDh_12_26_9'], 
            color=['green' if x > 0 else 'red' for x in data['MACDh_12_26_9']], 
            alpha=0.5, label='Histogram')
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax2.set_title(f'{ticker} - Moving Average Convergence Divergence (MACD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Perform technical analysis on a stock.')
    parser.add_argument('ticker', type=str, nargs='?', default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--period', type=str, default='1y',
                       help='Time period to analyze (default: 1y)')
    
    args = parser.parse_args()
    
    # Run the analysis
    analyze_stock(ticker=args.ticker, period=args.period)
