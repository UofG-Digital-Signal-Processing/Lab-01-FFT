import numpy as np

import util

INPUT_FILENAME = 'original.wav'
OUTPUT_FILENAME = 'Assignment 1_modified.wav'

LOWEST_HARMONIC_VOICE_FREQUENCY = 3000
HIGHEST_HARMONIC_VOICE_FREQUENCY = 6000
AMPLIFICATION = 10


def increase_voice_quality(data, sample_rate, amplification):
    ft = np.fft.fft(data)  # 傅里叶变换
    frequency = np.fft.fftfreq(data.size, d=1.0 / sample_rate)  # 频率
    amplitude = np.copy(ft)
    # determine the region of the highest harmonic voice frequencies in the spectrum
    frequency_scope = np.where(
        (LOWEST_HARMONIC_VOICE_FREQUENCY < frequency) & (frequency < HIGHEST_HARMONIC_VOICE_FREQUENCY))

    amplitude[frequency_scope] *= amplification
    # inverse Fourier transform
    modified_data = np.fft.ifft(amplitude)
    return np.real(modified_data)


data, sampling_rate = util.reader(INPUT_FILENAME)
modified_data = increase_voice_quality(data, sampling_rate, AMPLIFICATION)
util.writer(OUTPUT_FILENAME, modified_data, sampling_rate)
