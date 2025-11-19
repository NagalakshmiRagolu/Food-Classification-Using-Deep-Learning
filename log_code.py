
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import os

import logging
import os


def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Create log directory if it doesn't exist
    log_d = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\log_files'
    os.makedirs(log_d, exist_ok=True)

    log_path = os.path.join(log_d, f"{script_name}.log")
    handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
