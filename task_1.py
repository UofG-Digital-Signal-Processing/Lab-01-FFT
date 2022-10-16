import numpy as np

import util
import matplotlib.pyplot as plt
import constant

# Prepare the data
data, sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
# Prepare the figure
fig = plt.figure(figsize=(10, 10))
# Plot the time domain
fig.add_subplot(2, 1, 1)
time, amplitude = util.cal_time_domain(data, sample_rate)
# Normalize amplitude
amplitude = amplitude / np.max(amplitude)
plt.plot(time, amplitude)
plt.xlabel('Time')
plt.ylabel('Amplitude')
# Plot the frequency domain
fig.add_subplot(2, 1, 2)
frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
frequency = np.log10(frequency)
plt.plot(frequency, amplitude)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.show()
