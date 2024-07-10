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
        
        self.sampling_rate = 50
        self.window_size = 100 
        self.fft_size = 100
        self.update_interval_ms = 0  # 增加更新频率

        # 初始化数据
        self.data = np.zeros(self.window_size)
        self.fft_data = np.zeros(self.fft_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]

        # 初始化串口通信
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 57600, timeout=0.01)  # 设置合理的波特率
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
        
        # 初始化计数器
        self.counter = 0

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


    def update_plot(self):
        bytes_waiting = self.ser.in_waiting
        # print(bytes_waiting)
        if bytes_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').rstrip()
                new_value = float(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError: invalid byte sequence")
                new_value = 0.0
            except ValueError:
                print("ValueError: could not convert string to float")
                new_value = 0.0
        else:
            new_value = 0

        # 去除异常数值
        if self.data[-1] != 0 and abs(new_value - self.data[-1]) > 10:
            self.counter += 1
            return ;

        # 更新实时显示数据
        self.data = np.roll(self.data, -1)
        self.data[-1] = new_value

        self.line1.set_ydata(self.data)
        self.canvas1.draw()

        # 每5次更新一次FFT图表
        if self.counter % 3 == 0:
            self.fft_result = np.abs(np.fft.fft(self.fft_data)[:self.fft_size // 2])
            self.line2.set_ydata(self.fft_result)
            self.canvas2.draw()

        # 更新FFT数据
        self.fft_data = np.roll(self.fft_data, -1)
        self.fft_data[-1] = new_value

        self.counter += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
