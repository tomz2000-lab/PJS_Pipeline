"""
Load environment variables from .env file.
    
This function uses python-dotenv to load environment variables from a .env file
located in the same directory as this script. These
variables contain sensitive information like API tokens and connection strings.

After loading those the code starts ``extraction.py``, which then starts the whole 
pipeline process.

:return: None
:rtype: NoneType

:raises FileNotFoundError: If the .env file cannot be found
:raises PermissionError: If the .env file cannot be read due to permission issues

:example:
    >>> load_environment()
    # Environment variables are now accessible via os.environ or os.getenv
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv


# Get the absolute path to the .env file
load_dotenv(find_dotenv())
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
print(f"Looking for .env file at: {dotenv_path}")

# Check if file exists
if not os.path.exists(dotenv_path):
    print(f"ERROR: .env file not found at {dotenv_path}")
    sys.exit(1)

# Load with verbose output
load_dotenv(dotenv_path=dotenv_path, verbose=True)

# Verify variables are loaded
mongodb_uri = os.environ.get("MONGODB_URI")
hf_api_key = os.environ.get("HF_API_KEY")

# Star_Pipeline

import extraction

if __name__ == "__main__":
    """
    Main entry point for the application.
    
    When the script is run directly (not imported), this code block executes.
    """
    pass
