#!/usr/bin/env python3
"""Fail fast if the Python dependencies required by the Figure 2 rerun are absent."""

import dask
import distributed
import fcswrite
import matplotlib
import numpy
import pandas
import readfcs
import scipy


print(
    "PYTHON_OK",
    f"numpy={numpy.__version__}",
    f"pandas={pandas.__version__}",
    f"scipy={scipy.__version__}",
    f"matplotlib={matplotlib.__version__}",
)
