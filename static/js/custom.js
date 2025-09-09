<script>
document.addEventListener('DOMContentLoaded', () => {
    const symbolInput = document.getElementById('symbol');
    const expirationSelect = document.getElementById('expiration');

    async function fetchExpirations(symbol) {
        if (!symbol) {
            expirationSelect.innerHTML = '<option value="">Enter a symbol first</option>';
            return;
        }
        
        // Add a loading indicator
        expirationSelect.innerHTML = '<option value="">Loading...</option>';

        try {
            const response = await fetch(`/api/expirations/${symbol.toUpperCase()}`);
            
            // --- UPDATED ERROR HANDLING ---
            if (!response.ok) {
                const errorData = await response.json().catch(() => null); // Try to parse error
                const errorMessage = errorData ? errorData.error : 'Invalid Ticker or API Error';
                expirationSelect.innerHTML = `<option value="">Error: ${errorMessage}</option>`;
                throw new Error(errorMessage);
            }
            // --- END OF UPDATE ---

            const dates = await response.json();
            
            expirationSelect.innerHTML = ''; // Clear existing options

            if (dates && dates.length > 0) {
                dates.forEach(date => {
                    const option = new Option(date, date);
                    expirationSelect.add(option);
                });

                // Set the default to the third expiration date, if available
                if (dates.length >= 3) {
                    expirationSelect.selectedIndex = 2; // 0-indexed, so 2 is the third item
                }
            } else {
                expirationSelect.innerHTML = '<option value="">No expirations found</option>';
            }
        } catch (error) {
            console.error('Error fetching expiration dates:', error);
        }
    }

    // Add event listener to fetch expirations when the user changes the ticker
    symbolInput.addEventListener('change', () => {
        fetchExpirations(symbolInput.value);
    });

    // Fetch expirations for the default ticker on page load
    if (symbolInput.value) {
        fetchExpirations(symbolInput.value);
    }
});
</script>