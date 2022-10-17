import task_4

if __name__ == '__main__':
    vowel1_path = "vowel1.wav"
    vowel2_path = "vowel2.wav"
    for path in [vowel1_path, vowel2_path]:
        vowel = task_4.vowel_detect(path)
        print("The vowel is " + vowel + " according to the vowel detector.")
