import util
import os
import scipy.signal as signal
import constant
import matplotlib.pyplot as plt
import numpy as np

CONSONANT_AMPLITUDE_LOW_THRESHOLD = 110
SPEECH_AMPLITUDE_LOW_THRESHOLD = 0
original_data, sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
frequency, amplitude = util.cal_frequency_domain_db(original_data, sample_rate)


def cal_vowel_frequency_peak():
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
    return vowel_frequency


def cal_consonant_frequency_range():
    base_path = "data/consonant"
    filenames = os.listdir(base_path)
    consonant_frequency_range = []

    for filename in filenames:
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))

        frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
        # Search consonant frequency range through amplitude
        idxs = amplitude > CONSONANT_AMPLITUDE_LOW_THRESHOLD
        print(len(frequency[idxs]))
        consonant_frequency_range.append(frequency[idxs])
    return consonant_frequency_range


# Mark the frequency peaks of vowel in original video
def mark_vowel_frequency_peak():
    vowel_frequency_peaks = cal_vowel_frequency_peak()
    idxs = np.searchsorted(frequency, vowel_frequency_peaks)
    plt.plot(frequency, amplitude)
    plt.plot(frequency[idxs], amplitude[idxs], 'r.')
    plt.show()


# Mark the frequency range of consonant in original video
def mark_consonant_frequency_range():
    consonant_frequency_range = cal_consonant_frequency_range()
    idxs = np.array([], dtype=int)
    for frequency_range in consonant_frequency_range:
        idxs = np.append(idxs, np.searchsorted(frequency, frequency_range))
    plt.plot(frequency, amplitude)
    plt.plot(frequency[idxs], amplitude[idxs], 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


# Mark the frequency range of vowel and consonant in original video
def mark_vowel_and_consonant_frequency_range():
    time, amplitude = util.cal_time_domain(original_data, sample_rate)
    idxs = abs(amplitude) > SPEECH_AMPLITUDE_LOW_THRESHOLD
    time = time[idxs]
    amplitude = amplitude[idxs]
    plt.plot(time, amplitude)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()




# mark_vowel_frequency_peak()
# mark_consonant_frequency_range()
mark_vowel_and_consonant_frequency_range()