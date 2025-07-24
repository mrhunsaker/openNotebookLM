# marker_service/download_marker_models.py
import os
import sys

# Add Marker's root directory to sys.path to import its modules
sys.path.insert(0, os.path.abspath('.'))

try:
    from marker.models import load_all_models
    print("Attempting to download Marker models...")
    # This will load and cache models. Marker typically saves them to ~/.cache/marker
    # or a similar location. We want them within the container.
    # The default behavior of load_all_models should store them in a discoverable path.
    # If it writes to home directory, ensure the container's HOME is /app.
    os.environ['HOME'] = '/app' # Direct models to /app/.cache/marker if possible
    # Marker loads models into memory and also caches them to disk.
    # The first call to load_all_models will download them if not present.
    models, tokenizer = load_all_models()
    print("Marker models downloaded and cached successfully.")
except ImportError:
    print("Marker library not found. Skipping model download.")
except Exception as e:
    print(f"Error downloading Marker models: {e}")
    # Exit with an error if model download fails during build
    sys.exit(1)
