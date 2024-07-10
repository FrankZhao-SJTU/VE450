import sys
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import serial


class VibrationFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.sampling_rate = 3000
        self.window_size = 500 
        self.fft_size = 2000
        self.update_interval_ms = 1  # 1毫秒

        # 初始化数据
        self.data = np.zeros(self.window_size)
        self.fft_data = np.zeros(self.fft_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]

        # 初始化串口通信
        try:
            self.ser = serial.Serial('COM3', 921600, timeout=0.01)
            self.ser.flush()
        except serial.SerialException as e:
            print(f"Could not open serial port: {e}")
            sys.exit(1)
        
        print("Starting to read from serial port...")
        self.initUI()

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval_ms)


    def initUI(self):
        self.setWindowTitle('Real-time Vibration and FFT')

        # 设置固定窗口大小
        self.setFixedSize(2100, 1000)

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建垂直布局
        self.layout = QVBoxLayout(self.central_widget)

        # UI界面表头
        # 创建水平布局
        self.top_layout = QHBoxLayout()

        # 添加JI Logo
        logoji_label = QLabel(self)
        pixmap_ji = QPixmap('./UMJI_logo.png')
        logoji_label.setPixmap(pixmap_ji)
        logoji_label.setFixedSize(pixmap_ji.width(), pixmap_ji.height())
        self.top_layout.addWidget(logoji_label)

        # 添加Systence Logo
        logoco_label = QLabel(self)
        pixmap_co = QPixmap('./systence_logo.png')
        logoco_label.setPixmap(pixmap_co)
        logoco_label.setFixedSize(pixmap_co.width(), pixmap_co.height())
        self.top_layout.addWidget(logoco_label)

        # 添加组名
        font = QFont()
        font.setPointSize(26)  # 设置字体大小为28pt
        font.setBold(True)  # 设置字体加粗
        group_label = QLabel("ECE/ME/MSE450 Group24", self)
        group_label.setFont(font)
        self.top_layout.addWidget(group_label)

        # 将水平布局添加到主垂直布局中
        self.layout.insertLayout(0, self.top_layout)

        # 设置布局的位置顺序
        self.layout.addStretch()

        # 创建振动数据图
        xs = np.linspace(0, 1, self.window_size)  # x轴时间点
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_ylim(20, 45)
        # self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_xlim(0, 1)
        self.ax1.set_title('Real-time Vibration Data')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Amplitude')
        self.line1, = self.ax1.plot(xs, self.data)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setFixedSize(800, 400)  # 固定窗口大小
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
        self.canvas2.setFixedSize(800, 400)  # 固定窗口大小
        self.layout.addWidget(self.canvas2)

        # 创建显示频率峰值的标签
        self.freq_label = QLabel(self)
        self.layout.addWidget(self.freq_label)

    def get_newdata(self):
        if self.ser.in_waiting > 0:
            # print("readline:", self.ser.readline())
            try:
                line = self.ser.readline().decode('utf-8').rstrip()
                return float(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError: invalid byte sequence")
                return 0.0
            except ValueError:
                print("ValueError: could not convert string to float")
                return 0.0
        return 0.0

    def update_plot(self):
        new_value = self.get_newdata()
        print("new_value is:", new_value)
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
        # peaks, _ = find_peaks(self.fft_result, height=100)
        # peak_frequencies = self.frequencies[peaks]
        # peak_info = ", ".join([f"{freq:.2f} Hz" for freq in peak_frequencies])
        # self.freq_label.setText(f'Peak Frequencies: {peak_info}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
