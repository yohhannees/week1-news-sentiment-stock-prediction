import yfinance as yf
import pandas as pd
import pandas_ta as ta
from ta import add_all_ta_features
from ta.utils import dropna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FinancialAnalyzer:
    """
    A class to perform financial analysis on stock data including technical indicators.
    """
    
    def __init__(self, ticker: str, period: str = "1y"):
        """
        Initialize the FinancialAnalyzer with a stock ticker and time period.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            period (str): Time period to fetch data for (default: "1y")
        """
        self.ticker = ticker.upper()
        self.period = period
        self.data = None
        self.indicators = {}
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            self.data = dropna(self.data)
            return self.data
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {str(e)}")
            return None
    
    def calculate_indicators(self):
        """
        Calculate various technical indicators using pandas_ta and ta libraries.
        """
        if self.data is None:
            print("No data available. Please fetch data first using fetch_data().")
            return
        
        # Simple Moving Averages
        self.data['SMA_20'] = ta.sma(self.data['Close'], length=20)
        self.data['SMA_50'] = ta.sma(self.data['Close'], length=50)
        self.data['SMA_200'] = ta.sma(self.data['Close'], length=200)
        
        # Bollinger Bands
        bollinger = ta.bbands(self.data['Close'], length=20, std=2)
        self.data = pd.concat([self.data, bollinger], axis=1)
        
        # RSI (Relative Strength Index)
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(self.data['Close'])
        self.data = pd.concat([self.data, macd], axis=1)
        
        # Volume Weighted Average Price (VWAP)
        self.data['VWAP'] = ta.vwap(self.data['High'], self.data['Low'], 
                                   self.data['Close'], self.data['Volume'])
        
        # Store the indicator names for reference
        self.indicators = {
            'moving_averages': ['SMA_20', 'SMA_50', 'SMA_200'],
            'bollinger_bands': ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'],
            'momentum': ['RSI'],
            'trend': ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'],
            'volume': ['VWAP']
        }
    
    def plot_technical_analysis(self):
        """
        Create an interactive technical analysis chart using Plotly.
        """
        if self.data is None or self.indicators == {}:
            print("No data or indicators available. Please fetch data and calculate indicators first.")
            return
        
        # Create subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, 
                          row_heights=[0.5, 0.15, 0.15, 0.2],
                          subplot_titles=('Price and Moving Averages', 
                                         'Bollinger Bands', 
                                         'RSI', 
                                         'Volume'))
        
        # Add Price and Moving Averages
        fig.add_trace(go.Candlestick(x=self.data.index,
                                    open=self.data['Open'],
                                    high=self.data['High'],
                                    low=self.data['Low'],
                                    close=self.data['Close'],
                                    name='Price'),
                     row=1, col=1)
        
        for ma in self.indicators['moving_averages']:
            fig.add_trace(go.Scatter(x=self.data.index, 
                                   y=self.data[ma], 
                                   name=ma,
                                   line=dict(width=1)),
                        row=1, col=1)
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=self.data.index, 
                               y=self.data['BBU_20_2.0'], 
                               name='Upper BB',
                               line=dict(width=1, color='gray'),
                               opacity=0.7),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, 
                               y=self.data['BBL_20_2.0'], 
                               name='Lower BB',
                               line=dict(width=1, color='gray'),
                               opacity=0.7,
                               fill='tonexty'),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, 
                               y=self.data['BBM_20_2.0'], 
                               name='Middle BB',
                               line=dict(width=1, color='black', dash='dash'),
                               opacity=0.7),
                     row=2, col=1)
        
        # Add RSI
        fig.add_trace(go.Scatter(x=self.data.index, 
                               y=self.data['RSI'], 
                               name='RSI',
                               line=dict(width=1, color='purple')),
                     row=3, col=1)
        # Add RSI boundaries
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Add Volume
        fig.add_trace(go.Bar(x=self.data.index, 
                           y=self.data['Volume'],
                           name='Volume'),
                     row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Technical Analysis - {self.ticker}',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=1000,
            template='plotly_white'
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Bollinger Bands", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        # Show the figure
        fig.show()

    def get_technical_summary(self) -> dict:
        """
        Generate a summary of technical indicators.
        
        Returns:
            dict: Dictionary containing technical analysis summary
        """
        if self.data is None or self.indicators == {}:
            print("No data or indicators available. Please fetch data and calculate indicators first.")
            return {}
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        summary = {
            'ticker': self.ticker,
            'date': latest.name.strftime('%Y-%m-%d'),
            'price': {
                'close': latest['Close'],
                'change': latest['Close'] - prev['Close'],
                'change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            },
            'moving_averages': {
                'price_vs_sma20': (latest['Close'] - latest['SMA_20']) / latest['SMA_20'] * 100,
                'price_vs_sma50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100,
                'price_vs_sma200': (latest['Close'] - latest['SMA_200']) / latest['SMA_200'] * 100,
                'sma_cross': 'Golden Cross' if latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']
                            else 'Death Cross' if latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']
                            else 'No Cross'
            },
            'bollinger_bands': {
                'bb_percent': latest['BBP_20_2.0'],
                'position': 'Upper Band' if latest['Close'] > latest['BBU_20_2.0']
                           else 'Lower Band' if latest['Close'] < latest['BBL_20_2.0']
                           else 'Between Bands'
            },
            'rsi': {
                'value': latest['RSI'],
                'signal': 'Overbought' if latest['RSI'] > 70
                         else 'Oversold' if latest['RSI'] < 30
                         else 'Neutral'
            },
            'macd': {
                'value': latest['MACD_12_26_9'],
                'signal': latest['MACDs_12_26_9'],
                'histogram': latest['MACDh_12_26_9'],
                'trend': 'Bullish' if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] and latest['MACDh_12_26_9'] > prev['MACDh_12_26_9']
                         else 'Bearish' if latest['MACD_12_26_9'] < latest['MACDs_12_26_9'] and latest['MACDh_12_26_9'] < prev['MACDh_12_26_9']
                         else 'Neutral'
            },
            'vwap': {
                'value': latest['VWAP'],
                'price_vs_vwap': (latest['Close'] - latest['VWAP']) / latest['VWAP'] * 100
            }
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer with a stock ticker and time period
    analyzer = FinancialAnalyzer(ticker="AAPL", period="6mo")
    
    # Fetch the stock data
    data = analyzer.fetch_data()
    
    if data is not None:
        # Calculate technical indicators
        analyzer.calculate_indicators()
        
        # Display the technical analysis chart
        analyzer.plot_technical_analysis()
        
        # Get and print the technical summary
        summary = analyzer.get_technical_summary()
        print("\nTechnical Analysis Summary:")
        print("-" * 30)
        print(f"Ticker: {summary['ticker']}")
        print(f"Date: {summary['date']}")
        print(f"Price: ${summary['price']['close']:.2f} ({summary['price']['change']:+.2f}, {summary['price']['change_pct']:+.2f}%)")
        print("\nMoving Averages:")
        print(f"  Price vs SMA(20): {summary['moving_averages']['price_vs_sma20']:+.2f}%")
        print(f"  Price vs SMA(50): {summary['moving_averages']['price_vs_sma50']:+.2f}%")
        print(f"  Price vs SMA(200): {summary['moving_averages']['price_vs_sma200']:+.2f}%")
        print(f"  SMA Cross: {summary['moving_averages']['sma_cross']}")
        print("\nBollinger Bands:")
        print(f"  %B: {summary['bollinger_bands']['bb_percent']:.2f}")
        print(f"  Position: {summary['bollinger_bands']['position']}")
        print("\nRSI:")
        print(f"  Value: {summary['rsi']['value']:.2f}")
        print(f"  Signal: {summary['rsi']['signal']}")
        print("\nMACD:")
        print(f"  MACD: {summary['macd']['value']:.4f}")
        print(f"  Signal: {summary['macd']['signal']:.4f}")
        print(f"  Histogram: {summary['macd']['histogram']:.4f}")
        print(f"  Trend: {summary['macd']['trend']}")
        print("\nVWAP:")
        print(f"  Value: {summary['vwap']['value']:.2f}")
        print(f"  Price vs VWAP: {summary['vwap']['price_vs_vwap']:+.2f}%")
