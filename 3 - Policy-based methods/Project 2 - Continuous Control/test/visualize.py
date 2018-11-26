"""
Utility functions for generating training graphs.
"""

import numpy as np
import matplotlib.pyplot as plt

def sub_plot(coords, data, y_label='', x_label=''):
    """Plot a single graph (subplot)."""

    plt.subplot(coords)
    plt.plot(np.arange(len(data)), data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
