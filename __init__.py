"""bnspy: A Python package for binary neutron star mergers.
* Code: https://
* Docs: https://

It contains:
::
 test              --- Run Gammapy unit tests
 __version__       --- Gammapy version string
The Gammapy functionality is available for import from
the following sub-packages (e.g. `gammapy.makers`):
::
 astro        --- Astrophysical source and population models
 catalog      --- Source catalog tools
 makers       --- Data reduction functionality
 data         --- Data and observation handling
 irf          --- Instrument response functions (IRFs)
 maps         --- Sky map data structures
 modeling     --- Models and fitting
 estimators   --- High level flux estimation
 stats        --- Statistics tools
 utils        --- Utility functions and classes
"""

from . import shock
from . import utils
from . import jet
from . import models