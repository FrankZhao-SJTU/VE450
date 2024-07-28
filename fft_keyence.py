import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter, find_peaks

# 读取文件
filepath = input("Enter the filepath: ")
if not os.path.exists(filepath):
    raise FileNotFoundError("File not found")
if not os.path.splitext(filepath)[1] == ".xlsx":
    raise ValueError("File is not an Excel file")

# 尝试读取文件
try:
    Keyence = pd.read_excel(filepath)
except Exception as e:
    raise ValueError(f"Error reading the Excel file: {e}")

# 输入采样率，获取distance数据和样本数
sample_rate = int(input("Enter the sample rate: "))
if sample_rate <= 0:
    raise ValueError("Sample rate must be a positive integer")

data = Keyence.iloc[:, 2]  # 假设数据在第三列
N = len(data)
if N < 2:
    raise ValueError("Not enough data points")

# 进行FFT
fft_result = np.fft.fft(data)
frequencies = np.fft.fftfreq(N, d=1/sample_rate)
positive_frequencies = frequencies[:N//2]
positive_fft_result = np.abs(fft_result[:N//2])

# 平滑FFT结果， window_length和polyorder可以调整, window_length必须为奇数而且小于样本数，越大越平滑
window_length = 11
polyorder = 2
if window_length >= N or window_length % 2 == 0:
    raise ValueError("window_length must be an odd integer less than the number of data points")
smoothed_fft_result = savgol_filter(positive_fft_result, window_length=window_length, polyorder=polyorder)

# 手动设置噪声的最大频率，然后找到三个频率峰值，noise_freq可调节
noise_freq = 2
non_noise_min_index = next((i for i, freq in enumerate(positive_frequencies) if freq > noise_freq), None)
if non_noise_min_index is None:
    raise ValueError("No frequencies found above the noise frequency threshold")

peaks, _ = find_peaks(smoothed_fft_result[non_noise_min_index:])

# 根据振幅大小对峰值进行排序，并选择振幅最大的三个峰值
peak_freq_num = 3
sorted_peaks = sorted(peaks, key=lambda x: smoothed_fft_result[non_noise_min_index + x], reverse=True)
top_peaks = sorted_peaks[:peak_freq_num]
peak_frequencies = positive_frequencies[non_noise_min_index:][top_peaks]
peak_amplitudes = smoothed_fft_result[non_noise_min_index:][top_peaks]

# 找到第peak_freq_num+1高的峰值
next_peak = sorted_peaks[peak_freq_num] if len(sorted_peaks) > peak_freq_num else None
next_peak_frequency = positive_frequencies[non_noise_min_index + next_peak] if next_peak is not None else None
next_peak_amplitude = smoothed_fft_result[non_noise_min_index + next_peak] if next_peak is not None else None

print(f"Top {peak_freq_num} peak frequencies:", peak_frequencies)
print(f"Top {peak_freq_num} peak amplitudes:", peak_amplitudes)
if next_peak is not None:
    print(f"{peak_freq_num+1} peak frequency:", next_peak_frequency)
    print(f"{peak_freq_num+1} peak amplitude:", next_peak_amplitude)

plt.plot(positive_frequencies, smoothed_fft_result)
plt.plot(peak_frequencies, peak_amplitudes, 'ro') 
if next_peak is not None:
    plt.axhline(y=next_peak_amplitude, color='g', linestyle='--', label=f'{peak_freq_num+1} Peak Amplitude ({next_peak_amplitude:.2f})')
plt.title("Keyence FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.ylim(0,5)
plt.legend()
plt.show()
