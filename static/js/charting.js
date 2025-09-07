// Wait for the entire HTML document to be loaded and parsed
document.addEventListener('DOMContentLoaded', () => {

    // Get references to our HTML elements
    const tickerInput = document.getElementById('tickerInput');
    const timeframeControls = document.getElementById('timeframe-controls');
    const ctx = document.getElementById('stockChart').getContext('2d');

    // This variable will hold our chart instance, so we can destroy it before creating a new one
    let stockChart;

    /**
     * Main function to fetch data and update the chart
     */
    async function updateChart() {
        // Get the current values from the user controls
        const ticker = tickerInput.value.toUpperCase();
        const selectedTimeframe = document.querySelector('input[name="timeframe"]:checked').value;

        // Construct the API URL
        const apiUrl = `/api/history/${ticker}/${selectedTimeframe}`;

        try {
            // Fetch data from our Flask API
            const response = await fetch(apiUrl);
            if (!response.ok) {
                // Handle errors like a 404 for an invalid ticker
                throw new Error('Stock data not found. Please check the ticker.');
            }
            const chartData = await response.json();

            // If a chart instance already exists, destroy it
            if (stockChart) {
                stockChart.destroy();
            }

            // Create a new chart using the data we fetched
            stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels, // Dates for the X-axis
                    datasets: [{
                        label: `${ticker} Closing Price`,
                        data: chartData.data, // Prices for the Y-axis
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 1,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Price (USD)'
                            }
                        }
                    }
                }
            });

        } catch (error) {
            console.error('Error updating chart:', error);
            // Optionally, display an error message to the user on the page
            if (stockChart) {
                stockChart.destroy(); // Clear the old chart on error
            }
        }
    }

    // --- Event Listeners ---

    // Update chart when the ticker input changes (e.g., user types and hits Enter or clicks away)
    tickerInput.addEventListener('change', updateChart);

    // Update chart when a different timeframe button is selected
    timeframeControls.addEventListener('change', updateChart);

    // --- Initial Chart Load ---

    // Load the default chart (GOOGL, 3M) when the page first loads
    updateChart();
});