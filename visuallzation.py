import util
import os
import scipy.signal as signal
import numpy as np

base_path = "data/consonant"
filenames = os.listdir(base_path)

for filename in filenames:
    if ".wav" not in filename:
        continue
    data, sample_rate = util.reader(os.path.join(base_path, filename))
    util.plot_frequency_domain(data, sample_rate)

