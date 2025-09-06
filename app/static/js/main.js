document.addEventListener('DOMContentLoaded', () => {
    const getChartBtn = document.getElementById('getChartBtn');
    const tickerInput = document.getElementById('tickerInput');
    const timeframeGroup = document.getElementById('timeframe-group');
    const messageSection = document.getElementById('messageSection');
    const chartSection = document.getElementById('chartSection');
    const chartTickerSymbol = document.getElementById('chart-ticker-symbol');

    let stockChart = null; // Variable to hold the chart instance
    let activeTimeframe = '3M'; // Default timeframe

    // Handle clicks on timeframe buttons
    timeframeGroup.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
            // Remove active class from all buttons
            timeframeGroup.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            // Add active class to the clicked button
            e.target.classList.add('active');
            activeTimeframe = e.target.dataset.timeframe;
        }
    });

    // Handle click on the main "Get Chart" button
    getChartBtn.addEventListener('click', fetchStockData);
    
    // Allow pressing Enter to trigger the search
    tickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            fetchStockData();
        }
    });


    async function fetchStockData() {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (!ticker) {
            showMessage('Please enter a stock ticker.');
            return;
        }

        // Show loading state
        getChartBtn.disabled = true;
        getChartBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
        showMessage('Fetching data...', false);
        chartSection.style.display = 'none';

        try {
            // Fetch data from our Flask backend API
            const response = await fetch(`/get_stock_data?ticker=${ticker}&timeframe=${activeTimeframe}`);
            const data = await response.json();

            if (!response.ok) {
                // Handle errors from the backend (like invalid ticker)
                throw new Error(data.error || 'Something went wrong');
            }
            
            // If we get data, render the chart
            renderChart(data);

        } catch (error) {
            showMessage(`Error: ${error.message}`, true);
            chartSection.style.display = 'none';
        } finally {
            // Reset button state
            getChartBtn.disabled = false;
            getChartBtn.innerHTML = 'Get Chart';
        }
    }

    function renderChart(data) {
        const ctx = document.getElementById('stockChart').getContext('2d');
        
        // Update the ticker symbol title
        chartTickerSymbol.textContent = data.company;
        
        // Hide the initial message and show the chart
        messageSection.style.display = 'none';
        chartSection.style.display = 'block';

        const chartData = {
            labels: data.labels,
            datasets: [{
                label: `Closing Price for ${data.company}`,
                data: data.data,
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                fill: true,
                tension: 0.1,
                borderWidth: 2,
                pointRadius: 1,
            }]
        };

        // If a chart instance already exists, update it. Otherwise, create a new one.
        if (stockChart) {
            stockChart.data = chartData;
            stockChart.update();
        } else {
            stockChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
    }
    
    function showMessage(message, isError = false) {
        messageSection.style.display = 'block';
        const p = messageSection.querySelector('p');
        p.textContent = message;
        p.className = isError ? 'text-danger' : 'text-muted';
    }
});