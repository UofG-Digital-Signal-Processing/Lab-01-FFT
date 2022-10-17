### ENG5027 Assesment 1

###### Team Members: Jinming Zhang, Xiaohui Yu, Jianyu Zhao, Ziyuan Cheng

## Task 1: Loading Audio into Python

#### I. Read the audio samples into a python application

##### **Steps:**

  1. Use "*open*" function from "*pyaudio*" library to record the origin wavfile: set Format = 16bit, Channel = 1 and Sample Rate = 44100. Save it as "original.wav".	

2. Use the "*wavfile.read*" function from library "*scipy.io*" to convert the .wav file into an array "data" (which saving the amplitudes) and an Integer "sample_rate" (which saving the sample rate)

   ```python
   #Read wavfile    
   data, sample_rate = util.reader(constant.ORIGINAL_VIDEO_URL)
   ```

#### II. Plot the audio signal

##### **Steps:**

1. Calculate the time domain and divided the amplitude by its max value to normalise it.

   ```python
   #Normalise the amplitude    
   time, amplitude = util.cal_time_domain(data, sample_rate)
   amplitude = amplitude / np.max(amplitude)
   ```

​		![Amplitude_vs_Time](/Users/chengziyuan/Documents/GitHub/DSP-Lab-1/Amplitude_vs_Time.svg)
2. Calculate the Amplitude (dB) by  $20\times ln (amplitude)$  and logarithmic the frequency by $\ln(frequency)$.

   ```python
   #Calculate frequency domain and dB
   frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
   #convert frequency to logarithmic
   frequency = np.log10(frequency)
   ```

​		![Amplitude(dB)_vs_Frequency](/Users/chengziyuan/Documents/GitHub/DSP-Lab-1/Amplitude(dB)_vs_Frequency.svg)