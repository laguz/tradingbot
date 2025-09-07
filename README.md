# Stock Price Viewer

A simple web application to view historical stock prices.

## Features

- View historical stock price charts for a given ticker symbol.
- Select different timeframes (1M, 3M, 6M, 1Y).

## Getting Started

### Prerequisites

- Python 3.6+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-viewer.git
   cd stock-price-viewer
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python run.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/`.
3. Enter a stock ticker symbol and select a timeframe to view the chart.

## Configuration

This application requires an API key from [Alpha Vantage](https://www.alphavantage.co/).

1. Create a `.env` file in the root directory of the project.
2. Add your Alpha Vantage API key to the `.env` file:
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key
   ```
