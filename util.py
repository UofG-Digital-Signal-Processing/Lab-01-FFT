import pyaudio
import librosa
from scipy.io import wavfile
import wave
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

PLAYER_CHUNK = 1024

RECORDER_CHUNK = 1024
RECORDER_FORMAT = pyaudio.paInt16
RECORDER_CHANNELS = 1
RECORDER_RATE = 44100
RECORDER_INPUT_DEVICE_INDEX = 0


def recorder(filename, duration):
    p = pyaudio.PyAudio()
    stream = p.open(format=RECORDER_FORMAT,
                    channels=RECORDER_CHANNELS,
                    rate=RECORDER_RATE,
                    input=True,
                    input_device_index=RECORDER_INPUT_DEVICE_INDEX,
                    frames_per_buffer=RECORDER_CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RECORDER_RATE / RECORDER_CHUNK * duration)):
        data = stream.read(RECORDER_CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(RECORDER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(RECORDER_FORMAT))
    wf.setframerate(RECORDER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def writer(filename, data, sample_rate):
    sf.write(filename, data, sample_rate)


def reader(filename):
    sample_rate, data = wavfile.read(filename)
    # data, sample_rate = sf.read(filename)
    return data, sample_rate


def player(filename):
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(PLAYER_CHUNK)

    while data != b'':
        stream.write(data)
        data = wf.readframes(PLAYER_CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()


def cal_time_domain(data, sample_rate):
    amplitude = data
    time = np.linspace(0, len(amplitude) / sample_rate, num=len(amplitude))
    return time, amplitude


def cal_frequency_domain(data, sample_rate):
    ft = np.fft.fft(data)  # 傅里叶变换

    frequency = np.fft.fftfreq(data.size, d=1.0 / sample_rate)  # 频率
    frequency = frequency[:int(len(frequency) / 2)]  # 由于对称性，只取一半区间
    amplitude = np.abs(ft)  # 取绝对值
    amplitude = amplitude[:int(len(amplitude) / 2)]  # 由于对称性，只取一半区间
    return frequency, amplitude


def cal_frequency_domain_db(data, sample_rate):
    frequency, amplitude = cal_frequency_domain(data, sample_rate)
    amplitude = 20 * np.log10(amplitude)
    return frequency, amplitude


def plot_time_domain(data, sample_rate):
    amplitude = data
    time = np.linspace(0, len(amplitude) / sample_rate, num=len(amplitude))
    plt.plot(time, amplitude)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_frequency_domain(frequency, amplitude):
    plt.plot(frequency, amplitude)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


def _db():
    pass