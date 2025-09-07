// Wait for the entire HTML document to be loaded and parsed
document.addEventListener('DOMContentLoaded', () => {

    // Get references to our HTML elements
    const tickerInput = document.getElementById('tickerInput');
    const timeframeControls = document.getElementById('timeframe-controls');
    const chartContainer = document.getElementById('chart-container');
    const chartError = document.getElementById('chart-error');
    const ctx = document.getElementById('stockChart').getContext('2d');

    // This variable will hold our chart instance, so we can destroy it before creating a new one
    let stockChart;

    /**
     * Hides the chart and displays an error message.
     * @param {string} message The error message to display.
     */
    function showError(message) {
        chartContainer.classList.add('d-none'); // Hide chart
        chartError.classList.remove('d-none'); // Show error
        chartError.textContent = message;
        if (stockChart) {
            stockChart.destroy(); // Ensure old chart is gone
        }
    }

    /**
     * Hides the error message and shows the chart.
     */
    function showChart() {
        chartError.classList.add('d-none'); // Hide error
        chartContainer.classList.remove('d-none'); // Show chart
    }

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
                const errorData = await response.json().catch(() => ({ error: 'Stock data not found. Please check the ticker.' }));
                throw new Error(errorData.error || 'An unknown error occurred.');
            }
            const chartData = await response.json();

            // If the backend returns its own error message, throw it
            if (chartData.error) {
                throw new Error(chartData.error);
            }

            // Success: show the chart container
            showChart();

            // If a chart instance already exists, destroy it
            if (stockChart) {
                stockChart.destroy();
            }

            const datasets = [{
                label: `${ticker} Closing Price`,
                data: chartData.data, // Prices for the Y-axis
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                fill: true,
                tension: 0.1,
                pointRadius: 1,
            }];

            // Add support line if available
            if (chartData.support > 0) {
                datasets.push({
                    label: 'Support',
                    data: Array(chartData.labels.length).fill(chartData.support),
                    borderColor: 'rgba(40, 167, 69, 0.8)', // Green
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                });
            }

            // Add resistance line if available
            if (chartData.resistance > 0) {
                datasets.push({
                    label: 'Resistance',
                    data: Array(chartData.labels.length).fill(chartData.resistance),
                    borderColor: 'rgba(220, 53, 69, 0.8)', // Red
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                });
            }

            // Create a new chart using the data we fetched
            stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels, // Dates for the X-axis
                    datasets: datasets
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
            showError(error.message);
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