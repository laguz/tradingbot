from flask import Flask
from config import Config

# Create the Flask application instance
app = Flask(__name__)

# Load the configuration from the Config class
app.config.from_object(Config)

# Import the routes after the app is created to avoid circular imports
from app import routes