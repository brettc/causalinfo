"""Import the main causalinfo namespace into here

This makes life easier for testing
"""

import os
import sys

__all__ = ['causalinfo']

try:
    # Try and get it
    import causalinfo
except ImportError:

    # Put the parent folder into the import path
    def parent(p):
        return os.path.split(p)[0]

    sys.path.append(parent(parent(os.path.abspath(__file__))))

    # Now import it
    import causalinfo
