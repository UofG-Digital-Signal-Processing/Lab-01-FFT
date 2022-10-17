import constant
import task_3
import util

if __name__ == "__main__":
    left = 50
    right = 2000
    amplification = 1.5
    improved_data, sample_rate = task_3.enhance_voice(constant.ORIGINAL_VIDEO_URL, left, right, amplification)
    util.writer(constant.IMPROVED_VIDEO_URL, improved_data, sample_rate)
