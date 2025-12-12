/**
 * Theme Management System
 * Handles dark/light theme switching with persistence
 */

class ThemeManager {
    constructor() {
        this.currentTheme = this.getStoredTheme() || 'dark';
        this.init();
    }

    init() {
        // Apply stored theme
        this.applyTheme(this.currentTheme);

        // Listen for theme toggle clicks
        document.addEventListener('DOMContentLoaded', () => {
            const toggleBtn = document.getElementById('theme-toggle');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => this.toggle());
                this.updateToggleButton(toggleBtn);
            }
        });
    }

    getStoredTheme() {
        return localStorage.getItem('theme');
    }

    setStoredTheme(theme) {
        localStorage.setItem('theme', theme);
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.currentTheme = theme;
        this.setStoredTheme(theme);
    }

    toggle() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);

        // Update button if it exists
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            this.updateToggleButton(toggleBtn);
        }

        // Show feedback
        if (window.toast) {
            window.toast.info(`Switched to ${newTheme} mode`, 2000);
        }
    }

    updateToggleButton(button) {
        const icon = button.querySelector('i');
        if (icon) {
            if (this.currentTheme === 'dark') {
                icon.className = 'fas fa-sun';
                button.setAttribute('aria-label', 'Switch to light mode');
                button.title = 'Switch to light mode';
            } else {
                icon.className = 'fas fa-moon';
                button.setAttribute('aria-label', 'Switch to dark mode');
                button.title = 'Switch to dark mode';
            }
        }
    }

    getCurrentTheme() {
        return this.currentTheme;
    }
}

// Create global instance
window.themeManager = new ThemeManager();
