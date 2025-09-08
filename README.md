# Tradier Trading Dashboard

This is a Flask web application that provides a dashboard for interacting with the **Tradier brokerage API**. It allows users to view account metrics, analyze stock charts with support and resistance levels, and submit multi-leg option spread orders.

The application is built with a modular structure, separating front-end templates, back-end logic, and external API services.

## Features

* **Account Dashboard**: The main page displays a real-time summary of your trading account, including:
    * Account Balance, Daily P/L, and Yearly P/L.
    * Available Option Buying Power.
    * A detailed table of all open positions with calculated profit/loss.

* **Interactive Stock Charts**: A dynamic charting page that features:
    * The ability to chart any valid stock ticker for multiple timeframes (1M, 3M, 6M, 1Y).
    * Automatic calculation and display of support and resistance lines on the chart.
    * A summary list of the calculated support and resistance levels below the chart.
    * Dynamic updates via JavaScript, allowing for analysis without reloading the page.

* **Vertical Spread Orders**: A dedicated page to build and submit complex multi-leg option orders.
    * Dynamically fetches and populates valid option expiration dates for the chosen ticker.
    * Validates form data.
    * Constructs and submits the order to the Tradier API.

## Technology Stack

* **Backend**: Python with Flask
* **Frontend**: MDBootstrap, Chart.js
* **Data Analysis**: Pandas & NumPy
* **Forms**: Flask-WTF
* **API Communication**: Python Requests Library

## Setup and Installation

Follow these steps to get the application running locally.

1.  **Environment Variables**:
    Create a file named `.env` in the root of the project directory. It should contain your Tradier Sandbox API key and account ID.

    ```
    TRADIER_API_KEY="YOUR_API_KEY_HERE"
    TRADIER_ACCOUNT_ID="YOUR_ACCOUNT_ID_HERE"
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment. Once activated, install the required Python packages using the `requirements.txt` file.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install packages
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    Execute the `app.py` file to start the Flask development server.

    ```bash
    python app.py
    ```

4.  **Access the Dashboard**:
    Open your web browser and navigate to **http://127.0.0.1:5000**.