import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import constant
import util

original_data, original_sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
original_time, original_time_amplitude = util.cal_time_domain(original_data, original_sample_rate)
original_time_amplitude = original_time_amplitude / np.max(original_time_amplitude)
original_frequency, original_frequency_amplitude = util.cal_frequency_domain_db(original_data, original_sample_rate)
original_frequency = np.log10(original_frequency)


def cal_vowel_frequency_peak():
    base_path = constant.VOWEL_VIDEO_BASE_PATH
    filenames = os.listdir(base_path)
    vowel_frequency_peaks = []
    # Iterate audio files
    for filename in filenames:
        # Avoid non-formatted files
        if ".wav" not in filename:
            continue
        # Load audio data
        data, sample_rate = util.reader(os.path.join(base_path, filename))
        # Calculate frequency domain
        frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
        # Search the peaks of frequency through amplitude
        peak_idxs = signal.find_peaks(amplitude[:constant.VOWEL_FREQUENCY_HIGH_THRESHOLD], distance=1000)[0]
        for peak_idx in peak_idxs:
            vowel_frequency_peaks.append(np.log10(frequency[peak_idx]))
    return vowel_frequency_peaks


def cal_vowel_frequency_range():
    base_path = constant.VOWEL_VIDEO_BASE_PATH
    filenames = os.listdir(base_path)
    vowel_frequency_range = []
    # Iterate audio files
    for filename in filenames:
        # Avoid non-formatted files
        if ".wav" not in filename:
            continue
        data, sample_rate = util.reader(os.path.join(base_path, filename))
        # Calculate frequency domain
        frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
        # Search vowel frequency range through amplitude
        idxs = amplitude > constant.VOWEL_AMPLITUDE_LOW_THRESHOLD
        vowel_frequency_range.append(np.log10(frequency[idxs]))
    return vowel_frequency_range


def cal_consonant_frequency_range():
    base_path = constant.CONSONANT_VIDEO_BASE_PATH
    filenames = os.listdir(base_path)
    consonant_frequency_range = []
    # Iterate audio files
    for filename in filenames:
        # Avoid non-formatted files
        if ".wav" not in filename:
            continue
        # Load audio data
        data, sample_rate = util.reader(os.path.join(base_path, filename))
        # Calculate frequency domain
        frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
        # Search consonant frequency range through amplitude
        idxs = amplitude > constant.CONSONANT_AMPLITUDE_LOW_THRESHOLD
        consonant_frequency_range.append(np.log10(frequency[idxs]))
    return consonant_frequency_range


def mark_vowel_frequency_peak():
    vowel_frequency_peaks = cal_vowel_frequency_peak()
    # Determine the position of the vowel peak in the original audio frequency
    idxs = np.searchsorted(original_frequency, vowel_frequency_peaks)
    # Plot the figure
    plt.plot(original_frequency, original_frequency_amplitude, label='Original Audio')
    plt.plot(original_frequency[idxs], original_frequency_amplitude[idxs], 'r.', label='Vowel Frequency Peak')
    plt.title('Vowel Frequency Peak in Original Audio')
    plt.xlabel('Frequency (log)')
    plt.ylabel('Amplitude (dB)')
    # plt.show()
    plt.savefig('res/task_2_a.svg')
    plt.close()


def mark_vowel_frequency_range():
    vowel_frequency_range = cal_vowel_frequency_range()
    # Plot the figure
    plt.plot(original_frequency, original_frequency_amplitude)
    for range_item in vowel_frequency_range:
        # Determine the range of the vowel frequency in the original audio frequency
        range_idx = np.searchsorted(original_frequency, range_item)
        plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.',
                 label='Consonant Frequency Range')
    plt.title('Vowel Frequency Frequency in Original Audio')
    plt.xlabel('Frequency (log)')
    plt.ylabel('Amplitude (dB)')
    # plt.show()
    plt.savefig('res/task_2_b.svg')
    plt.close()


# Mark the frequency range of consonant in original video
def mark_consonant_frequency_range():
    consonant_frequency_range = cal_consonant_frequency_range()
    # Plot the figure
    plt.plot(original_frequency, original_frequency_amplitude)
    for range_item in consonant_frequency_range:
        # Determine the range of the consonant frequency in the original audio frequency
        range_idx = np.searchsorted(original_frequency, range_item)
        plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.',
                 label='Consonant Frequency Range')
    plt.title('Consonant Frequency Range in Original Audio')
    plt.xlabel('Frequency (log)')
    plt.ylabel('Amplitude (dB)')
    # plt.show()
    plt.savefig('res/task_2_b.svg')
    plt.close()


# Mark the frequency range of vowel and consonant in original video
def mark_vowel_and_consonant_frequency_range():
    frequency_range = np.append(cal_consonant_frequency_range(), cal_vowel_frequency_range())
    plt.plot(original_frequency, original_frequency_amplitude)
    for range_item in frequency_range:
        range_idx = np.searchsorted(original_frequency, range_item)
        plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.',
                 label='Vowel and Consonant Frequency Range')
    plt.title('Vowel and Consonant Frequency Range in Original Audio')
    plt.xlabel('Frequency (log)')
    plt.ylabel('Amplitude (dB)')
    # plt.show()
    plt.savefig('res/task_2_c.svg')


mark_vowel_frequency_peak()
mark_consonant_frequency_range()
mark_vowel_and_consonant_frequency_range()
