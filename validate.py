import pandas as pd
import numpy as np
import os
import shutil
from scipy.stats import spearmanr
from src.main import train, infer, TARGET_NAME, PREDICTION_NAME

# --- Validation Configuration ---
NUM_VALIDATION_MOONS = 13