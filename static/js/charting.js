// laguz/tradingbot/tradingbot-bc1f680a95b47c592f111a193bf8d1c99a0bd96d/static/js/charting.js
document.addEventListener('DOMContentLoaded', () => {

    // --- HTML Element References ---
    const tickerInput = document.getElementById('tickerInput');
    const timeframeControls = document.getElementById('timeframe-controls');
    const chartContainer = document.getElementById('chart-container');
    const chartError = document.getElementById('chart-error');
    const ctx = document.getElementById('stockChart').getContext('2d');

    // New references for the analysis card
    const analysisCard = document.getElementById('analysis-card');
    const analysisTicker = document.getElementById('analysis-ticker');
    const supportContainer = document.getElementById('support-levels-container');
    const resistanceContainer = document.getElementById('resistance-levels-container');

    let stockChart;

    /**
     * Hides the chart and analysis, and displays an error message.
     * @param {string} message The error message to display.
     */
    function showError(message) {
        chartContainer.classList.add('d-none');
        analysisCard.classList.add('d-none');
        chartError.classList.remove('d-none');
        chartError.textContent = message;
        if (stockChart) {
            stockChart.destroy();
        }
    }

    /**
     * Hides the error message and shows the chart and analysis card.
     */
    function showChartAndAnalysis() {
        chartError.classList.add('d-none');
        chartContainer.classList.remove('d-none');
        analysisCard.classList.remove('d-none');
    }

    /**
     * Updates the support or resistance list in the DOM.
     * @param {HTMLElement} container - The container element for the list.
     * @param {Array<number>} levels - The array of support/resistance levels.
     * @param {boolean} isSupport - True for support, false for resistance.
     */
    function updateLevelsList(container, levels, isSupport) {
        container.innerHTML = ''; // Clear previous content

        if (levels && levels.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'list-group';

            const iconClass = isSupport ? 'fa-arrow-trend-up text-success' : 'fa-arrow-trend-down text-danger';

            levels.forEach(level => {
                const li = document.createElement('li');
                li.className = 'list-group-item d-flex justify-content-between align-items-center';
                li.innerHTML = `<div><i class="fas ${iconClass} me-2"></i>$${level}</div>`;
                ul.appendChild(li);
            });
            container.appendChild(ul);
        } else {
            const p = document.createElement('p');
            p.className = 'text-muted';
            p.textContent = `No significant ${isSupport ? 'support' : 'resistance'} levels found.`;
            container.appendChild(p);
        }
    }


    /**
     * Main function to fetch data and update the chart and analysis
     */
    async function updateChart() {
        const ticker = tickerInput.value.toUpperCase();
        const selectedTimeframe = document.querySelector('input[name="timeframe"]:checked').value;
        const apiUrl = `/api/history/${ticker}/${selectedTimeframe}`;

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Stock data not found. Please check the ticker.' }));
                throw new Error(errorData.error || 'An unknown error occurred.');
            }
            const chartData = await response.json();

            if (chartData.error) {
                throw new Error(chartData.error);
            }

            // --- Success ---
            showChartAndAnalysis();

            // 1. Update Analysis Card Header
            analysisTicker.textContent = ticker;

            // 2. Update Support and Resistance Lists
            updateLevelsList(supportContainer, chartData.levels.support, true);
            updateLevelsList(resistanceContainer, chartData.levels.resistance, false);


            // 3. Update Chart
            if (stockChart) {
                stockChart.destroy();
            }

            const datasets = [{
                label: `${ticker} Closing Price`,
                data: chartData.data,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                fill: true,
                tension: 0.1,
                pointRadius: 1,
            }];

            if (chartData.support && chartData.support.length > 0) {
                chartData.support.forEach(level => {
                    datasets.push({
                        data: Array(chartData.labels.length).fill(level),
                        borderColor: 'rgba(40, 167, 69, 0.8)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                        tension: 0,
                    });
                });
            }

            if (chartData.resistance && chartData.resistance.length > 0) {
                chartData.resistance.forEach(level => {
                    datasets.push({
                        data: Array(chartData.labels.length).fill(level),
                        borderColor: 'rgba(220, 53, 69, 0.8)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                        tension: 0,
                    });
                });
            }

            stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    // ---------------------------------------------
                    scales: {
                        x: {
                            title: { display: true, text: 'Date', color: '#a0a0a0' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#a0a0a0' }
                        },
                        y: {
                            title: { display: true, text: 'Price (USD)', color: '#a0a0a0' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#a0a0a0' }
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
    tickerInput.addEventListener('change', updateChart);
    timeframeControls.addEventListener('change', updateChart);

    // --- Initial Chart Load ---
    updateChart();
});