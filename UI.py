import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit
from PyQt5.QtCore import QTimer, Qt
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
        self.stop_time_ms = 20000  # 实时采样在10s后会自动停止

        # 初始化数据
        self.data = np.zeros(self.window_size)
        self.fft_data = np.zeros(self.fft_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]
        self.data_storage = []
        self.fft_result_all = np.zeros(self.fft_size // 2)

        # 初始化串口通信
        try:
            self.ser = serial.Serial('COM3', 57600, timeout=0.01)  # 设置合理的波特率
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

        # 设置10秒计时器
        self.stop_timer = QTimer()
        self.stop_timer.timeout.connect(self.stop_update)
        self.stop_timer.setSingleShot(True)  # 单次触发
        self.stop_timer.start(self.stop_time_ms)

    def initUI(self):
        self.setWindowTitle('Vibration Detection and Analysis System')

        # 设置固定窗口大小
        self.setFixedSize(2000, 1000)

        # 设置整个窗口背景为白色
        self.setStyleSheet("background-color: white;")

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建水平布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(1)

        # 创建左侧布局
        self.left_layout = QVBoxLayout()

        # 添加JI Logo
        logoji_label = QLabel(self)
        pixmap_ji = QPixmap('./UMJI_logo.png')
        logoji_label.setPixmap(pixmap_ji)
        logoji_label.setFixedSize(600, 100)
        logoji_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.left_layout.addWidget(logoji_label)

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
        self.canvas1.setFixedSize(600, 400)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas1)
        self.left_layout.setAlignment(self.canvas1, Qt.AlignCenter)

        # 创建FFT结果图
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlim(0, self.sampling_rate / 2)
        self.ax2.set_ylim(0, 500)
        self.ax2.set_title('FFT Result')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.line2, = self.ax2.plot(self.frequencies, self.fft_result)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setFixedSize(600, 400)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas2)
        self.left_layout.setAlignment(self.canvas2, Qt.AlignCenter)

        # 创建显示频率峰值的标签
        self.freq_label = QLabel(self)
        self.left_layout.addWidget(self.freq_label)

        # 将左侧布局添加到主水平布局中
        self.main_layout.addLayout(self.left_layout)

        # 创建中间垂直布局
        self.middle_layout = QVBoxLayout()

        # 添加组名
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        group_label = QLabel("SU24 ECE/ME/MSE450 Group24", self)
        group_label.setFont(font)
        group_label.setFixedSize(800, 100)
        group_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.middle_layout.addWidget(group_label)

        # 创建空白初始状态的静态图
        self.fig3, self.ax3 = plt.subplots()
        self.ax3.plot([], [])
        self.ax3.set_ylim(20, 47)
        self.ax3.set_title(f"Displacement over {round(self.stop_time_ms/1000,0)} seconds")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Displacement")
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setFixedSize(800, 400)
        self.middle_layout.addWidget(self.canvas3)
        self.middle_layout.setAlignment(self.canvas3, Qt.AlignTop)

        self.fig4, self.ax4 = plt.subplots()
        self.ax4.set_xlim(0, self.sampling_rate / 2)
        self.ax4.set_ylim(0, 500)
        self.ax4.set_title('FFT Result on All Data')
        self.ax4.set_xlabel('Frequency (Hz)')
        self.ax4.set_ylabel('Magnitude')
        self.line4, = self.ax4.plot([], [])
        self.canvas4 = FigureCanvas(self.fig4)
        self.canvas4.setFixedSize(800, 400)
        self.middle_layout.addWidget(self.canvas4)
        self.middle_layout.setAlignment(self.canvas4, Qt.AlignTop)

        # 将中间布局添加到主水平布局中
        self.main_layout.addLayout(self.middle_layout)

        # 创建右侧垂直布局
        self.right_layout = QVBoxLayout()

        # 添加Systence Logo
        logoco_label = QLabel(self)
        pixmap_co = QPixmap('./systence_logo.png')
        scaled_pixmap_co = pixmap_co.scaled(100, 100)
        logoco_label.setPixmap(scaled_pixmap_co)
        logoco_label.setFixedSize(400,100)
        logoco_label.setAlignment(Qt.AlignLeft)
        self.right_layout.addWidget(logoco_label)

        # 创建右侧文本框标题
        solTitle_label = QLabel("Analysis", self)
        solTitle_label.setFont(font)
        solTitle_label.setFixedSize(400, 50)
        solTitle_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(solTitle_label)

        # 创建右侧文本框内容
        sol_font = QFont()
        sol_font.setPointSize(10)
        solution_label = QLabel("These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later. These are temporary words and will be replaced later.", self)
        solution_label.setWordWrap(True)  # 自动换行
        solution_label.setAlignment(Qt.AlignTop)
        solution_label.setFont(sol_font)
        solution_label.setFixedSize(400, 800)
        self.right_layout.addWidget(solution_label)

        # 将右侧布局添加到主水平布局中
        self.main_layout.addLayout(self.right_layout)

    def update_plot(self):
        bytes_waiting = self.ser.in_waiting
        print(bytes_waiting)
        if bytes_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').rstrip()
                new_value = float(line)
                self.data_storage.append(new_value)
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


    def stop_update(self):
        self.timer.stop()
        self.plot_final_data()

    def plot_final_data(self):
        # 创建插值对象
        x = np.linspace(0, self.stop_time_ms/1000, len(self.data_storage))
        f = interp1d(x, self.data_storage, kind='cubic')
        x_new = np.linspace(0, self.stop_time_ms/1000, 500)
        y_new = f(x_new)

        # 创建新的位移图像并绘制平滑曲线
        self.fig3, self.ax3 = plt.subplots()
        self.ax3.plot(x_new, y_new)
        self.ax3.set_ylim(20, 47)
        self.ax3.set_title(f"Displacement over {self.stop_time_ms/1000} seconds")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Displacement")
        self.middle_layout.removeWidget(self.canvas3)
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setFixedSize(800, 400)  # 固定窗口大小

        # 在所有数据上进行FFT
        self.all_data_array = np.array(self.data_storage)
        self.fft_result_all = np.abs(np.fft.fft(self.all_data_array)[:self.fft_size // 2])

        # 创建新的FFT图像
        self.fig4, self.ax4 = plt.subplots()
        self.ax4.set_xlim(0, self.sampling_rate / 2)
        self.ax4.set_ylim(0, 500)
        self.ax4.set_title('FFT Result on All Data')
        self.ax4.set_xlabel('Frequency (Hz)')
        self.ax4.set_ylabel('Magnitude')
        self.line4, = self.ax4.plot(self.frequencies, self.fft_result_all)
        self.middle_layout.removeWidget(self.canvas4)
        self.canvas4 = FigureCanvas(self.fig4)
        self.canvas4.setFixedSize(800, 400)  # 固定窗口大小
        self.middle_layout.addWidget(self.canvas4)

        # 将新图像添加进中间布局
        self.middle_layout.addWidget(self.canvas3)
        self.middle_layout.addWidget(self.canvas4)
        self.middle_layout.setAlignment(self.canvas3, Qt.AlignTop)
        self.middle_layout.setAlignment(self.canvas4, Qt.AlignTop)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
