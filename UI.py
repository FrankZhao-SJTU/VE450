import sys
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VibrationFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 现在的设置为每次获取一个新的distance数据，FFT将会重新进行，生成新的图像
        # 但是实际情况，可能是从arduino一次性读取若干个数据点，然后一起更新，也就是更新的频率没有现在这么快
        self.sampling_rate = 1000
        self.window_size = 500 
        self.fft_size = 2000
        self.update_interval = 1  # 动画更新间隔为1毫秒，与采样频率相匹配

        # 初始化数据
        self.data = np.zeros(self.window_size)
        self.fft_data = np.zeros(self.fft_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]

        self.frame = 0
        
        self.initUI()

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)

    def initUI(self):
        self.setWindowTitle('Real-time Vibration and FFT')

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建垂直布局
        self.layout = QVBoxLayout(self.central_widget)

        # 创建振动数据图
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_ylim(-2, 2)
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_title('Real-time Vibration Data')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Amplitude')
        self.line1, = self.ax1.plot(self.data)
        self.canvas1 = FigureCanvas(self.fig1)
        self.layout.addWidget(self.canvas1)

        # 创建FFT结果图
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlim(0, self.sampling_rate / 2)
        self.ax2.set_ylim(0, 500)
        self.ax2.set_title('FFT Result')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.line2, = self.ax2.plot(self.frequencies, self.fft_result)
        self.canvas2 = FigureCanvas(self.fig2)
        self.layout.addWidget(self.canvas2)

        # 创建显示频率峰值的标签
        self.freq_label = QLabel(self)
        self.layout.addWidget(self.freq_label)


    def get_newdata(self):
        # 目前的data默认是处理后的数据，也就是distance, 输出单个数据点
        # 之后修改这个函数，data改为从arduino中读取
        # 生成新的数据点，sin(20x) + sin(300x) 并添加噪声
        t = self.frame / self.sampling_rate
        return np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 300 * t) + np.random.normal(0, 0.5)

    def update_plot(self):
        new_value = self.get_newdata()
        # 更新实时显示数据
        self.data = np.roll(self.data, -1)
        self.data[-1] = new_value
        # 更新FFT数据
        self.fft_data = np.roll(self.fft_data, -1)
        self.fft_data[-1] = new_value

        # 计算FFT
        self.fft_result = np.abs(np.fft.fft(self.fft_data)[:self.fft_size // 2])
        # 更新图表数据
        self.line1.set_ydata(self.data)
        self.line2.set_ydata(self.fft_result)

        self.canvas1.draw()
        self.canvas2.draw()

        # 找到频率峰值并更新标签
        peaks, _ = find_peaks(self.fft_result, height=100)
        peak_frequencies = self.frequencies[peaks]
        peak_info = ", ".join([f"{freq:.2f} Hz" for freq in peak_frequencies])
        self.freq_label.setText(f'Peak Frequencies: {peak_info}')

        self.frame += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
