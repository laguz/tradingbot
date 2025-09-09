document.addEventListener('DOMContentLoaded', () => {

    /**
     * Sets up a fetcher for option expiration dates for a given form.
     * @param {string} symbolInputId The ID of the stock ticker input field.
     * @param {string} expirationSelectId The ID of the expiration date select field.
     */
    function setupExpirationFetcher(symbolInputId, expirationSelectId) {
        const symbolInput = document.getElementById(symbolInputId);
        const expirationSelect = document.getElementById(expirationSelectId);

        if (!symbolInput || !expirationSelect) {
            return;
        }

        async function fetchExpirations(symbol) {
            if (!symbol) {
                expirationSelect.innerHTML = '<option value="">Enter a symbol first</option>';
                return;
            }
            
            expirationSelect.innerHTML = '<option value="">Loading...</option>';

            try {
                const response = await fetch(`/api/expirations/${symbol.toUpperCase()}`);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => null);
                    const errorMessage = errorData ? errorData.error : 'Invalid Ticker or API Error';
                    expirationSelect.innerHTML = `<option value="">Error: ${errorMessage}</option>`;
                    throw new Error(errorMessage);
                }

                const dates = await response.json();
                
                expirationSelect.innerHTML = ''; 

                if (dates && dates.length > 0) {
                    dates.forEach(date => {
                        const option = new Option(date, date);
                        expirationSelect.add(option);
                    });

                    if (dates.length >= 3) {
                        expirationSelect.selectedIndex = 2;
                    }
                } else {
                    expirationSelect.innerHTML = '<option value="">No expirations found</option>';
                }
            } catch (error) {
                console.error(`Error fetching expiration dates for ${symbolInputId}:`, error);
            }
        }

        symbolInput.addEventListener('change', () => {
            fetchExpirations(symbolInput.value);
        });

        if (symbolInput.value) {
            fetchExpirations(symbolInput.value);
        }
    }

    /**
     * Toggles the visibility of a price input field based on the value of an order type dropdown.
     * @param {string} orderTypeSelectId The ID of the <select> element for the order type.
     * @param {string} priceFieldId The ID of the wrapper element for the price input.
     */
    function setupOrderTypeToggle(orderTypeSelectId, priceFieldId) {
        const orderTypeSelect = document.getElementById(orderTypeSelectId);
        const priceField = document.getElementById(priceFieldId);

        if (!orderTypeSelect || !priceField) return;

        const togglePriceField = () => {
            if (orderTypeSelect.value === 'limit') {
                priceField.style.display = 'block';
            } else {
                priceField.style.display = 'none';
            }
        };

        orderTypeSelect.addEventListener('change', togglePriceField);
        togglePriceField(); // Run on page load
    }

    // Setup expiration date fetchers for all relevant forms
    setupExpirationFetcher('symbol', 'expiration'); 
    setupExpirationFetcher('symbol_single', 'expiration_single');
    setupExpirationFetcher('symbol_condor', 'expiration_condor');

    // Setup order type toggles for price fields
    setupOrderTypeToggle('stock_order_type', 'stock_price_field');
    setupOrderTypeToggle('single_option_order_type', 'single_option_price_field');
});