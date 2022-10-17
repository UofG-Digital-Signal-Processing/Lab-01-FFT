import wave

import matplotlib
import numpy
import numpy as np
import pyaudio
import pylab

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import wavfile


def get_framerate(wavefile):
    """
        Enter the file path and get the frame rate
    """
    wf = wave.open(wavefile, "rb")  # 打开wav
    p = pyaudio.PyAudio()  # 创建PyAudio对象
    params = wf.getparams()  # 参数获取
    nchannels, sampwidth, framerate, nframes = params[:4]
    return framerate


def get_nframes(wavefile):
    """
        Enter the file path and get the number of frames
    """
    wf = wave.open(wavefile, "rb")
    p = pyaudio.PyAudio()
    params = wf.getparams()  # Parameter acquisition
    nchannels, sampwidth, framerate, nframes = params[:4]
    return nframes


def get_wavedata(wavefile):
    """
        Enter the file path to get the processed array
    """
    sample_rate, data = wavfile.read(wavefile)
    return data


def plot_timedomain(wavfile):
    '''
        Draw the time domain diagram
    '''
    wave_data = get_wavedata(wavfile)  # Get processed wave data
    framerate = get_framerate(wavfile)  # For frame rate
    nframes = get_nframes(wavfile)  # Get the number of frames

    # Construct abscissa
    time = numpy.arange(0, nframes) * (1.0 / framerate)

    # Paint
    pylab.figure(figsize=(40, 10))
    pylab.subplot(111)
    pylab.plot(time, wave_data)
    pylab.xlabel("time (seconds)")
    pylab.show()
    return None


def plot_freqdomain(wavefile):
    """
        Draw the frequency domain
    """
    max_val = 32767
    sample_rate = get_nframes(wavefile)
    data = get_wavedata(wavefile)
    amplitude = np.array(data)

    amplitude_norm = amplitude / max_val

    # calculating the total number of samples
    total_samples = np.size(amplitude)

    # calculating the frequency step size for the signal
    freq_step = sample_rate / total_samples

    # calculating the frequency domain for the signal
    freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
    freq_domain_plt = freq_domain[:int(total_samples / 2) + 1]

    # calculating the frequency response of the signal
    pos_x = int(200 / freq_step)
    pos_y = int(1000 / freq_step)

    freq_mag = np.fft.fft(amplitude_norm)
    freq_mag_norm = freq_mag / total_samples
    freq_mag_abs = np.abs(freq_mag_norm)
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]

    freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

    # graphing the frequency response of the signal in logarithmic scale
    plt.figure("Plot of frequency spectrum with logarithmic scales")
    plt.plot(freq_domain_plt, freq_mag_dB)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid()

    plt.show()


def plot_freqdomain_enhance(wavefile, start, end, multiple):
    '''
        Plot the frequency domain (increase the maximum sine wave)
        start: The beginning of the interval
        end:The end of the increment interval
        multiple:A multiple of increase
    '''
    max_val = 32767
    sample_rate = get_nframes(wavefile)
    data = get_wavedata(wavefile)
    amplitude = np.array(data)

    amplitude_norm = amplitude / max_val

    # calculating the total number of samples
    total_samples = np.size(amplitude)

    # calculating the frequency step size for the signal
    freq_step = sample_rate / total_samples

    # calculating the frequency domain for the signal
    freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
    freq_domain_plt = freq_domain[:int(total_samples / 2) + 1]

    # calculating the frequency response of the signal
    pos_x = int(start / freq_step)
    pos_y = int(end / freq_step)

    freq_mag = np.fft.fft(amplitude_norm)

    freq_mag_rec = np.copy(freq_mag)
    freq_mag_rec[pos_x:pos_y] = freq_mag_rec[pos_x:pos_y] * multiple
    freq_mag_rec[total_samples - pos_y: total_samples - pos_x] = freq_mag_rec[
                                                                 total_samples - pos_y: total_samples - pos_x] * multiple

    amp_rec = np.fft.ifft(freq_mag_rec)

    freq_mag_norm = freq_mag_rec / total_samples
    freq_mag_abs = np.abs(freq_mag_norm)
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]

    freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

    # graphing the frequency response of the signal in logarithmic scale
    plt.figure("Plot of frequency spectrum with logarithmic scales")
    plt.plot(freq_domain_plt, freq_mag_dB)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid()

    # wavfile.write("enhanced.wav", 44100, np.float32(amp_rec))
    # display all of the graphs for the signal
    wavfile.write("enhanced1.wav", 44100, np.float32(amp_rec))
    plt.show()


if __name__ == "__main__":
    plot_freqdomain_enhance("original_mono.wav", 50, 2000, 1.5)
