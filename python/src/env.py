

import os

mypath = os.path.abspath(__file__)

# Export only selected symbols
__all__ = ['graphdir']

# Directory of current script file
basedir = os.path.dirname(mypath)

# Output directory for graphs
graphdir = os.path.abspath(os.path.join(basedir, '..', 'graphs'))

# Create directory if it does not exist
if not os.path.isdir(graphdir):
    os.makedirs(graphdir, exist_ok=True)

