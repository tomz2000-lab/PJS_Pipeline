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
from dotenv import load_dotenv
import extraction

# Load Toekens and Paswords

load_dotenv()


# Star_Pipeline
if __name__ == "__main__":
    """
    Main entry point for the application.

    When the script is run directly (not imported), this code block executes.
    """
    pass
