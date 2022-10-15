import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

max_val = 32768
cutoff = 1000


# Function to identify vowels
def vowel_detector(wav_file):
    # loading the sample rate of the signal along with the data points
    sample_rate, data = wavfile.read(wav_file)
    # loading the data of the sample into an array
    amplitude = np.array(data)
    amplitude_norm = amplitude / max_val
    # calculating the total number of samples
    total_samples = np.size(amplitude)
    # calculating the frequency step size for the signal
    freq_step = sample_rate / total_samples
    # calculating the frequency response of the signal
    freq_mag = np.fft.fft(amplitude_norm) / total_samples
    freq_mag_abs = np.abs(freq_mag)
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]

    left = int(100 / freq_step)
    right = int(150 / freq_step)

    max_amplitude = np.max(freq_mag_abs_plt[left:right])
    print(max_amplitude)
    if max_amplitude > 0.028:
        return "a"
    if 0.018 < max_amplitude < 0.025:
        return "i"
    else:
        return "e"
