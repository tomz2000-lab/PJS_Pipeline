import os
import sys
import importlib
sys.path.insert(0, os.path.abspath('/home/lukas/Dokumente/Working Code/sphinx'))

for module in sys.modules.values():
    if hasattr(module, '__file__') and module.__file__ and '/home/lukas/Dokumente/Working Code/sphinx' in module.__file__:
        try:
            importlib.reload(module)
        except:
            pass


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Using LLMs to structure job data'
copyright = '2025, Tom Ziegler'
author = 'Tom Ziegler'
release = '16.05.2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']


