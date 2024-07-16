import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QGridLayout, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import serial

model = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
messages = [SystemMessage(content="I am conducting vibration analysis. \
    Analysis the measured displacement and FFT results. Describe the \
    vibration amplitude and frequency. The unit of measured distance is \
    mm. The unit of frequency is Hz. Ouput format: only the description."),]


class VibrationFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.sampling_rate = 30
        self.window_size = 30 
        self.fft_size = 32
        self.update_interval_ms = 0  # 增加更新频率
        self.start_time = time.time()  # 开始时间

        
        # 初始化数据
        self.data = np.zeros(self.window_size)
        self.fft_data = np.zeros(self.fft_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]
        self.data_storage = []
        self.fft_result_all = np.zeros(self.fft_size // 2)
        self.t = 0
        self.threshold = 10  # 设置FFT振幅的阈值
        self.sampling_started = False
        self.initUI()

        # 初始化串口通信
        try:
            self.ser = serial.Serial('COM5', 57600, timeout=0.01)  # 设置合理的波特率
            self.ser.flush()
        except serial.SerialException as e:
            print(f"Could not open serial port: {e}")
            sys.exit(1)
        print("Starting to read from serial port...")

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start(self.update_interval_ms)
        
        # 初始化计数器
        self.counter = 0


    def initUI(self):
        self.setWindowTitle('SU24 ECE/ME/MSE450 Group24: Vibration Detection and Analysis System')

        # 设置固定窗口大小
        self.setFixedSize(2000, 1000)

        # 设置整个窗口背景为白色
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLineEdit {
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50; /* 绿色背景 */
                border: none; /* 无边框 */
                color: white; /* 白色字体 */
                padding: 15px 32px; /* 按钮内边距 */
                text-align: center; /* 文字居中 */
                font-size: 24px; /* 字体大小 */
                margin: 4px 2px; /* 外边距 */
                border-radius: 12px; /* 圆角 */
            }
            QPushButton:hover {
                background-color: white; /* 悬停背景颜色 */
                color: black; /* 悬停字体颜色 */
                border: 2px solid #4CAF50; /* 悬停边框 */
            }
            QPushButton:pressed {
                background-color: #45a049; /* 按下背景颜色 */
            }
        """)

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)

        # 创建左侧布局
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)  # 设置左侧布局的对齐方式

        # 创建振动数据图
        xs = np.linspace(0, 1, self.window_size)  # x轴时间点
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_ylim(20, 45)
        self.ax1.set_xlim(0, 1)
        self.ax1.set_title('Real-time Vibration Data')
        self.ax1.set_ylabel('Amplitude')
        self.line1, = self.ax1.plot(xs, self.data)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setFixedSize(1100, 500)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas1, alignment=Qt.AlignTop)

        # 创建FFT结果图
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlim(0, self.sampling_rate / 2)
        self.ax2.set_ylim(0, 500)
        self.ax2.set_title('Real-time FFT Result')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.line2, = self.ax2.plot(self.frequencies, self.fft_result)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setFixedSize(1100, 450)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas2, alignment=Qt.AlignTop)


        # 将左侧布局添加到主布局中
        self.main_layout.addLayout(self.left_layout)

        # 创建右侧垂直布局
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)  # 设置右侧布局的对齐方式

        # 创建右侧文本框内容
        sol_font = QFont("Arial", 12)  # 设置字体为Arial，大小12
        self.solution_label = QLabel("These are temporary words and will be replaced later. ...", self)
        self.solution_label.setWordWrap(True)  # 自动换行
        self.solution_label.setAlignment(Qt.AlignTop)
        self.solution_label.setFont(sol_font)
        self.solution_label.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ddd; background-color: #fff; }")
        self.solution_label.setFixedSize(550, 650)
        self.right_layout.addWidget(self.solution_label, alignment=Qt.AlignTop)

        # 创建按钮布局
        button_layout = QGridLayout()

        # 创建四个按钮
        self.button1 = QPushButton("Start Sampling")
        self.button1.clicked.connect(self.start_sampling)
        self.button2 = QPushButton("End Sampling")
        self.button2.clicked.connect(self.stop_update)  # 连接按钮到stop_update方法
        self.button3 = QPushButton("Export .CSV")
        self.button3.clicked.connect(self.export_csv)
        self.button4 = QPushButton("Export Report")
        self.button4.clicked.connect(self.export_report)

        # 将按钮添加到布局中
        button_layout.addWidget(self.button1, 0, 0)
        button_layout.addWidget(self.button2, 0, 1)
        button_layout.addWidget(self.button3, 1, 0)
        button_layout.addWidget(self.button4, 1, 1)

        # 将按钮布局添加到右侧布局中
        self.right_layout.addLayout(button_layout)

        # 创建一个水平布局来放置两个logo
        logo_layout = QHBoxLayout()

        # 添加JI Logo
        logoji_label = QLabel(self)
        pixmap_ji = QPixmap('./UMJI_logo.png')
        logoji_label.setPixmap(pixmap_ji)
        logoji_label.setFixedSize(281, 43)
        logoji_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        logo_layout.addWidget(logoji_label)

        # 添加Systence Logo
        logoco_label = QLabel(self)
        pixmap_co = QPixmap('./systence_logo.png')
        logoco_label.setPixmap(pixmap_co)
        logoco_label.setFixedSize(91, 63)
        logoco_label.setAlignment(Qt.AlignBottom)
        logo_layout.addWidget(logoco_label)

        # 在右侧布局中添加一个弹性空间和logo布局
        self.right_layout.addStretch()  # 添加弹性空间，将logo推到最下方
        self.right_layout.addLayout(logo_layout)

        # 将右侧布局添加到主布局中
        self.main_layout.addLayout(self.right_layout)

    
    def start_sampling(self):
        if not self.sampling_started:
            # 重置数据
            self.data = np.zeros(self.window_size)
            self.fft_data = np.zeros(self.fft_size)
            self.data_storage = []
            self.t = 0
            self.counter = 0
            self.sampling_started = True  # 标志开始采样
        self.timer.start(self.update_interval_ms)
        self.ser.reset_input_buffer()
        
    
    
    def get_new_value(self):
        return (20+np.random.randn())*np.sin(2*np.pi*10*self.t) + (20+np.random.randn())*np.cos(2*np.pi*15*self.t) + np.random.randn()


    def update_plot(self):
        # new_value = self.get_new_value()
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
        # self.data_storage.append(new_value)
        # self.t += 0.02

        # 更新实时显示数据
        self.data = np.roll(self.data, -1)
        self.data[-1] = new_value

        self.line1.set_ydata(self.data)
        self.canvas1.draw()

        # 每3次更新一次FFT图表
        if self.counter % 3 == 0:
            self.fft_result = np.abs(np.fft.fft(self.fft_data)[:self.fft_size // 2])
            self.line2.set_ydata(self.fft_result)
            self.canvas2.draw()

            # 找到超过阈值的峰值频率并更新标签
            peak_indices, _ = find_peaks(self.fft_result, height=self.threshold)
            peak_freqs = self.frequencies[peak_indices]
            peak_info = '\n'.join([f"Peak Frequency: {freq:.2f} Hz" for freq in peak_freqs])
            self.solution_label.setText(f"{peak_info}\n")

        # 更新FFT数据
        self.fft_data = np.roll(self.fft_data, -1)
        self.fft_data[-1] = new_value

        self.counter += 1
        


    def stop_update(self):
        if self.sampling_started == True:
            self.timer.stop()
            self.sampling_started = False  # 标志停止采样
            self.stop_time = time.time() - self.start_time  # 获取停止时间
            self.plot_final_data()
            


    def plot_final_data(self):
        # 创建插值对象
        x = np.linspace(0, self.stop_time, len(self.data_storage))
        f = interp1d(x, self.data_storage, kind='cubic')
        x_new = np.linspace(0, self.stop_time, 500)
        y_new = f(x_new)

        # 更新位移图像并绘制平滑曲线
        self.ax1.clear()
        self.ax1.plot(x_new, y_new)
        self.ax1.set_ylim(20, 45)
        self.ax1.set_title(f"Displacement over {self.stop_time:.2f} seconds")
        self.ax1.set_ylabel("Displacement")
        self.canvas1.draw()  # 更新canvas1

        # 在所有数据上进行FFT
        self.all_data_array = np.array(self.data_storage)
        self.fft_result_all = np.abs(np.fft.fft(self.all_data_array, n=self.fft_size)[:self.fft_size // 2])
        # print(self.all_data_array)
        # print(self.frequencies)
        # print(self.fft_result_all)
        messages.append(HumanMessage(content=f"sample rate: {self.sampling_rate} \
            measured distance: {self.all_data_array}. After FFT, we get frequency: {self.frequencies} and\
            amplitude: {self.fft_result_all}"))
        self.solution_label.setText(f"{model.invoke(messages).content}\n")
        
        
        # 更新FFT图像
        self.ax2.clear()
        self.ax2.set_xlim(0, self.sampling_rate / 2)
        self.ax2.set_ylim(0, 500)
        self.ax2.set_title('FFT Result on All Data')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.ax2.plot(self.frequencies, self.fft_result_all)
        self.canvas2.draw()  # 更新canvas2

        # 确保布局中添加了更新后的图像
        self.left_layout.update()


    def export_csv(self):
        if self.sampling_started == False:
            # 获取当前日期和时间
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 设置默认文件名
            default_file_name = f"vibration_data_{current_time}.csv"
            
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_file_name, "CSV Files (*.csv);;All Files (*)", options=options)
            if filePath:
                with open(filePath, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Amplitude"])
                    for i, value in enumerate(self.data_storage):
                        writer.writerow([i / self.sampling_rate, value])


    def export_report(self):
        if self.sampling_started == False:
            # 获取当前日期和时间
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 设置默认文件名
            default_file_name = f"report_{current_time}.html"

            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Report", default_file_name, "HTML Files (*.html);;All Files (*)", options=options)
            if filePath:
                # 保存当前图像
                distance_image_path = "distance_plot.png"
                fft_image_path = "fft_plot.png"
                self.fig1.savefig(distance_image_path)
                self.fig2.savefig(fft_image_path)

                # 获取文本框内容
                text_content = self.solution_label.text()

                # 生成HTML内容
                html_content = f"""
                <html>
                <head>
                    <title>Vibration Analysis Report</title>
                </head>
                <body>
                    <h1>Vibration Analysis Report</h1>
                    <h2>Displacement over Time</h2>
                    <img src="{distance_image_path}" alt="Distance Plot">
                    <h2>FFT Result</h2>
                    <img src="{fft_image_path}" alt="FFT Plot">
                    <h2>Text Content</h2>
                    <p>{text_content}</p>
                </body>
                </html>
                """

                # 将HTML内容写入文件
                with open(filePath, 'w') as file:
                    file.write(html_content)

                print(f"Report saved to {filePath}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
