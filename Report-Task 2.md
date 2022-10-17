## Task 2

According to the topic description, the peaks of frequency of vowels, the range of frequency of consonants and the whole
range of frequency of voice, including vowels and consonants, should be plotted in the figures of Task 1.

First of all, For vowel detection, a reasonable method is formant. The muscles of the human vocal organs are softer and
have greater damping, and will resonate more frequencies; the resonance vibrates the resonant cavity, and then the vocal
tract will amplify some frequency components and attenuate other frequency components, resulting in For some resonant
frequencies, the resonant frequencies that are amplified in frequency characteristics will peak one after another.
Generally, these resonant frequencies are called resonant frequencies, and these peaks are called resonant peaks. Since
the voiced sound is produced by the vibration of the vocal cords, the voiced sound is closely related to the formant,
and it can be considered that the formant is the vowel. In most cases, the first two formants, $f_1$ and $f_2$, are
sufficient to separate the different vowels.

However, for consonants, the characteristics of formants are not significant, so it is not suitable to use such methods
to calculate, but due to time constraints, other more suitable methods have not been adopted, and the range of frequency
can only be roughly determined by the trend of the directly observed image.

### Mark the vowel frequencey peak in the spectrum of original audio

#### Step

1. Record every vowel used in the original audio.

   ```python
   def reader(filename):
       sample_rate, data = wavfile.read(filename)
       return data, sample_rate
   ```

1. Use function `signal.find_peaks` to roughly search 2-3 frequency peaks of every vowel, which called formant is the
   significant feature of vowel.

   ```python
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
   ```

1. Mark the above peaks in the spectrum of original audio, but due to precision issues, it is likely that there is no
   corresponding frequency in the spectrum, so the function `np.searchsorted` needs to be used for nearest matching
   during processing.

   ```python
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
       plt.show()
   ```

#### Result

![image-20221017215243909](https://chrisgray.oss-cn-beijing.aliyuncs.com/Imageshack/image-20221017215243909.png)

### Mark the frequency consonant range in the spectrum of original audio

#### Step

1. Record every consonant used in the original audio.

   ```python
   def reader(filename):
       sample_rate, data = wavfile.read(filename)
       return data, sample_rate
   ```

1. Search the consonant frequency range roughly through the amplitude low threshold.

   ```python
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
   ```

1. Mark the above range in the spectrum of original audio, but due to precision issues, it is likely that there is no
   corresponding frequency in the spectrum, so the function `np.searchsorted` needs to be used for nearest matching
   during processing.

   ```python
   def mark_consonant_frequency_range():
       consonant_frequency_range = cal_consonant_frequency_range()
       # Plot the figure
       plt.plot(original_frequency, original_frequency_amplitude)
       for range_item in consonant_frequency_range:
           # Determine the range of the consonant frequency in the original audio frequency
           range_idx = np.searchsorted(original_frequency, range_item)
           plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.', label='Consonant Frequency Range')
       plt.title('Consonant Frequency Range in Original Audio')
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('Amplitude (dB)')
       plt.show()
   ```

#### Result

![image-20221017215255281](https://chrisgray.oss-cn-beijing.aliyuncs.com/Imageshack/image-20221017215255281.png)

### Mark the whole speech spectrum including the vowels and consonants

#### Step

1. Use the above vowels and consonants files

1. Search the vowel frequency range roughly through the amplitude low threshold.

   ```python
   def mark_vowel_frequency_range():
       vowel_frequency_range = cal_vowel_frequency_range()
       # Plot the figure
       plt.plot(original_frequency, original_frequency_amplitude)
       for range_item in vowel_frequency_range:
           # Determine the range of the vowel frequency in the original audio frequency
           range_idx = np.searchsorted(original_frequency, range_item)
           plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.', label='Consonant Frequency Range')
       plt.title('Vowel Frequency Frequency in Original Audio')
       plt.xlabel('Frequency (log)')
       plt.ylabel('Amplitude (dB)')
       plt.show()
   ```

1. Mark both of vowel and consonant frequency range in the spectrum of original audio, but due to precision issues, it
   is likely that there is no corresponding frequency in the spectrum, so the function `np.searchsorted` needs to be
   used for nearest matching during processing.

   ```python
   def mark_vowel_and_consonant_frequency_range():
       frequency_range = np.append(cal_consonant_frequency_range(), cal_vowel_frequency_range())
       plt.plot(original_frequency, original_frequency_amplitude)
       for range_item in frequency_range:
           range_idx = np.searchsorted(original_frequency, range_item)
           plt.plot(original_frequency[range_idx], original_frequency_amplitude[range_idx], 'r.', label='Vowel and Consonant Frequency Range')
       plt.title('Vowel and Consonant Frequency Range in Original Audio')
       plt.xlabel('Frequency (log)')
       plt.ylabel('Amplitude (dB)')
       plt.show()
   ```

#### Result

![image-20221017220517996](https://chrisgray.oss-cn-beijing.aliyuncs.com/Imageshack/image-20221017220517996.png)
