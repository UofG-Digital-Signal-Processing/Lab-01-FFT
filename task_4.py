import librosa
import numpy as np
from scipy.signal import lfilter

import constant


def local_maximum(x):
    """
    Find the extreme value of a sequence
    :param x:
    :return:
    """
    d = np.diff(x)
    l_d = len(d)
    maximum = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            maximum.append(x[i + 1])
            loc.append(i + 1)
    return maximum, loc


def formant_cepst(u, cepst_l):
    """
    Resonance peak estimation function by inverse spectroscopy
    :param u:Input signal
    :param cepst_l:Width of the window function on frequency
    :return: val resonance peak amplitude 
    :return: loc resonance peak position 
    :return: spec envelope
    """
    wlen2 = len(u) // 2
    u_fft = np.fft.fft(u)
    U = np.log(np.abs(u_fft[:wlen2]))
    Cepst = np.fft.ifft(U)
    cepst = np.zeros(wlen2, dtype=np.complex)
    cepst[:cepst_l] = Cepst[:cepst_l]
    # Take the opposite of the second equation
    cepst[-cepst_l + 1:] = Cepst[-cepst_l + 1:]
    spec = np.real(np.fft.fft(cepst))
    #  Finding extreme values on the envelope
    val, loc = local_maximum(spec)
    return val, loc, spec


def detect_vowel(wavfile):
    path1 = constant.VOWEL_A_VIDEO_PATH
    path2 = constant.VOWEL_AE_VIDEO_PATH
    path3 = wavfile
    # sr=None Sound maintains original sampling frequency， mono=False Sound maintains original number of channels
    data1, fs1 = librosa.load(path1, sr=None, mono=False)
    data2, fs2 = librosa.load(path2, sr=None, mono=False)
    data3, fs3 = librosa.load(path3, sr=None, mono=False)
    # Pre-treatment - pre-emphasis
    u_1 = lfilter([1, -0.99], [1], data1)
    u_2 = lfilter([1, -0.99], [1], data2)
    u_3 = lfilter([1, -0.99], [1], data3)

    cepstL = 7
    wlen1 = len(u_1)
    wlen2 = len(u_2)
    wlen3 = len(u_3)
    wlenn1 = wlen1 // 2
    wlenn2 = wlen2 // 2
    wlenn3 = wlen3 // 2
    # Pre-treatment - window-added 
    freq1 = [i * fs1 / wlen1 for i in range(wlenn1)]
    freq2 = [i * fs2 / wlen2 for i in range(wlenn2)]
    freq3 = [i * fs3 / wlen3 for i in range(wlenn3)]
    # val （resonance peak amplitude），loc （resonance peak position），spec（envelope）
    val1, loc1, spec1 = formant_cepst(u_1, cepstL)
    val2, loc2, spec2 = formant_cepst(u_2, cepstL)
    val3, loc3, spec3 = formant_cepst(u_3, cepstL)
    # Resonance peak frequency
    f_a = [freq1[loc1[0]], freq1[loc1[1]]]
    f_ae = [freq2[loc2[0]], freq2[loc2[1]]]
    f_unk = [freq3[loc3[0]], freq3[loc3[1]]]

    if f_unk == f_a:
        return "a"
    if f_unk == f_ae:
        return "ae"
    else:
        return "unknown"
