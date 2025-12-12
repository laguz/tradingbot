/**
 * Animation Helpers
 * Micro-interactions and smooth transitions
 */

/**
 * Animate a number from start to end
 * @param {HTMLElement} element - Element to update
 * @param {number} start - Start value
 * @param {number} end - End value
 * @param {number} duration - Animation duration in ms
 * @param {Function} formatter - Optional formatter function
 */
function animateNumber(element, start, end, duration = 1000, formatter = null) {
    const startTime = performance.now();
    const delta = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (delta * easeOut);

        // Update element
        if (formatter) {
            element.textContent = formatter(current);
        } else {
            element.textContent = current.toFixed(2);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Add pulse effect to element
 * @param {HTMLElement} element - Element to pulse
 * @param {string} className - Class name for pulse effect
 */
function pulseElement(element, className = 'pulse-update') {
    element.classList.add(className);
    setTimeout(() => element.classList.remove(className), 600);
}

/**
 * Smooth scroll to element
 * @param {string|HTMLElement} target - Element or selector
 * @param {number} offset - Offset from top in pixels
 */
function smoothScrollTo(target, offset = 80) {
    const element = typeof target === 'string' ? document.querySelector(target) : target;
    if (!element) return;

    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - offset;

    window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
    });
}

/**
 * Add ripple effect to button click
 * @param {Event} event - Click event
 */
function createRipple(event) {
    const button = event.currentTarget;

    const circle = document.createElement('span');
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    const radius = diameter / 2;

    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${event.clientX - button.offsetLeft - radius}px`;
    circle.style.top = `${event.clientY - button.offsetTop - radius}px`;
    circle.classList.add('ripple');

    const ripple = button.getElementsByClassName('ripple')[0];
    if (ripple) {
        ripple.remove();
    }

    button.appendChild(circle);
}

/**
 * Animate cards on page load
 */
function animateCardsOnLoad() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

/**
 * Initialize animations on DOM ready
 */
document.addEventListener('DOMContentLoaded', () => {
    // Add ripple effect to all buttons with .btn-ripple class
    document.querySelectorAll('.btn-ripple').forEach(button => {
        button.addEventListener('click', createRipple);
    });

    // Animate cards on load
    if (document.body.classList.contains('animate-on-load')) {
        animateCardsOnLoad();
    }
});

// Export to window for global access
window.animateNumber = animateNumber;
window.pulseElement = pulseElement;
window.smoothScrollTo = smoothScrollTo;
window.createRipple = createRipple;
