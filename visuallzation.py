import util
import os
import matplotlib.pyplot as plt


base_path = "data/vowel"
filenames = os.listdir(base_path)

for filename in filenames:
    if ".wav" not in filename:
        continue
    data, sample_rate = util.reader(os.path.join(base_path, filename))
    frequency, amplitude = util.cal_frequency_domain_db(data, sample_rate)
    plt.plot(frequency, amplitude)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title(filename)
    plt.show()

