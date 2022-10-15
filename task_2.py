import util
import os
import scipy.signal as signal
import constant
import matplotlib.pyplot as plt
import numpy as np


def get_vowel_frequency():
    base_path = "data/vowel"
    filenames = os.listdir(base_path)
    vowel_frequency = []

    for filename in filenames:
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))

        frequency, amplitude = util.cal_frequency_domain(data, sample_rate)
        # ft = np.fft.fft(data)  # 傅里叶变换
        # frequency = np.fft.fftfreq(data.size, d=1.0 / sample_rate)  # 频率
        # frequency = frequency[:int(len(frequency) / 2)]  # 由于对称性，只取一半区间
        #
        # amplitude = np.abs(ft)  # 取绝对值
        # amplitude = amplitude[:int(len(amplitude) / 2)]  # 由于对称性，只取一半区间

        peak_idxs = signal.find_peaks(amplitude[:2500], distance=1000)[0]
        vowel_frequency += list(frequency[peak_idxs])
        print(vowel_frequency)
    return vowel_frequency


def get_consonant_frequency():
    base_path = "data/consonant"
    filenames = os.listdir(base_path)
    consonant_frequency = []

    for filename in filenames:
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))

        frequency, amplitude = util.cal_frequency_domain(data, sample_rate)
        # ft = np.fft.fft(data)  # 傅里叶变换
        # frequency = np.fft.fftfreq(data.size, d=1.0 / sample_rate)  # 频率
        # frequency = frequency[:int(len(frequency) / 2)]  # 由于对称性，只取一半区间
        #
        # amplitude = np.abs(ft)  # 取绝对值
        # amplitude = amplitude[:int(len(amplitude) / 2)]  # 由于对称性，只取一半区间

        peak_idxs = signal.find_peaks(amplitude[:5000], distance=2000)[0]
        consonant_frequency += list(frequency[peak_idxs])
    return consonant_frequency


vowel_frequency = get_vowel_frequency()
consonant_frequency = get_consonant_frequency()
original_data, sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
# fig = plt.figure(figsize=(10, 10))
# Plot the frequency domain
# fig.add_subplot(2, 1, 2)
frequency, amplitude = util.cal_frequency_domain(original_data, sample_rate)
plt.plot(frequency, amplitude)
# Mark vowel_frequency peaks in original video
idxs = np.isin(frequency, vowel_frequency)
print(frequency[idxs])
plt.plot(frequency[idxs], amplitude[idxs], 'ro:')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

# plt.show()