import util
import os
import scipy.signal as signal
import constant
import matplotlib.pyplot as plt
import numpy as np

original_data, original_sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
original_time, original_time_amplitude = util.cal_time_domain(original_data, original_sample_rate)
original_time_amplitude = original_time_amplitude / np.max(original_time_amplitude)
original_frequency, original_frequency_amplitude = util.cal_frequency_domain_db(original_data, original_sample_rate)
original_frequency = np.log10(original_frequency)
VOWEL_AMPLITUDE_LOW_THRESHOLD = 115
CONSONANT_AMPLITUDE_LOW_THRESHOLD = 110
VOICE_AMPLITUDE_LOW_THRESHOLD = 0.06


def cal_vowel_frequency_peak():
    base_path = "data/vowel"
    filenames = os.listdir(base_path)
    vowel_frequency_peaks = []

    for filename in filenames:
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))

        frequency, amplitude = util.cal_frequency_domain(data, sample_rate)
        # Search vowel frequency peak through amplitude
        peak_idxs = signal.find_peaks(amplitude[:2500], distance=1000)[0]
        vowel_frequency_peaks += list(frequency[peak_idxs])
    return vowel_frequency_peaks


def cal_vowel_frequency_range():
    base_path = "data/vowel"
    filenames = os.listdir(base_path)
    vowel_frequency_range = []

    for filename in filenames:
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))

        frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
        # Search consonant frequency range through amplitude
        idxs = amplitude > VOWEL_AMPLITUDE_LOW_THRESHOLD
        vowel_frequency_range.append(frequency[idxs])
    return vowel_frequency_range


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
        consonant_frequency_range.append(frequency[idxs])
    return consonant_frequency_range


# Mark the frequency peaks of vowel in original video
def mark_vowel_frequency_peak():
    vowel_frequency_peaks = cal_vowel_frequency_peak()
    idxs = np.searchsorted(original_frequency, vowel_frequency_peaks)
    plt.plot(original_frequency, original_frequency_amplitude)
    plt.plot(original_frequency[idxs], original_frequency_amplitude[idxs], 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


def mark_vowel_frequency_range():
    vowel_frequency_range = cal_vowel_frequency_range()
    idxs = np.array([], dtype=int)
    for frequency_range in vowel_frequency_range:
        idxs = np.append(idxs, np.searchsorted(original_frequency, frequency_range))
    plt.plot(original_frequency, original_frequency_amplitude)
    plt.plot(original_frequency[idxs], original_frequency_amplitude[idxs], 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


# Mark the frequency range of consonant in original video
def mark_consonant_frequency_range():
    consonant_frequency_range = cal_consonant_frequency_range()
    plt.plot(original_frequency, original_frequency_amplitude)
    for range_item in consonant_frequency_range:
        range_item = np.log10(range_item)
        idxs = np.searchsorted(original_frequency, range_item)
        plt.plot(original_frequency[idxs], original_frequency_amplitude[idxs], 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


# Mark the frequency range of vowel and consonant in original video
def mark_vowel_and_consonant_frequency_range():
    frequency_range = np.append(cal_consonant_frequency_range(), cal_vowel_frequency_range())
    # frequency_range = cal_consonant_frequency_range()
    plt.plot(original_frequency, original_frequency_amplitude)
    for range_item in frequency_range:
        range_item = np.log10(range_item)
        idxs = np.searchsorted(original_frequency, range_item)
        plt.plot(original_frequency[idxs], original_frequency_amplitude[idxs], 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


mark_vowel_frequency_peak()
mark_consonant_frequency_range()
mark_vowel_and_consonant_frequency_range()
