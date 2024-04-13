import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt

dt = 20e-9 #time sample is 20 ns
fs = 1 / dt #sampling frequency

