import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QGridLayout, QFileDialog, QScrollArea
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import csv
from scipy.signal import savgol_filter, find_peaks
from datetime import datetime
import os
import serial
import threading

class SerialReader:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.read_thread = None
        self.stop_event = threading.Event()

    def init_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)  # 设置合理的波特率
            self.ser.flush()
        except serial.SerialException as e:
            print(f"Could not open serial port: {e}")
            sys.exit(1)
        print("Starting to read from serial port...")

    def read_from_serial(self):
        while not self.stop_event.is_set():
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').rstrip()
                    new_value = float(line)
                except UnicodeDecodeError:
                    print("UnicodeDecodeError: invalid byte sequence")
                    new_value = 0.0
                except ValueError:
                    print("ValueError: could not convert string to float")
                    new_value = 0.0
                # print("new value:", new_value)
                return new_value

    def start_receiving(self):
        self.stop_event.clear()
        self.ser.reset_input_buffer()  # 清除串口输入缓冲区中的所有数据
        self.read_thread = threading.Thread(target=self.read_from_serial)
        self.read_thread.start()
        print("Reading thread started.")

    def stop_receiving(self):
        self.stop_event.set()
        self.read_thread.join()
        print("Reading thread stopped.")

    def close_serial(self):
        if self.ser:
            self.ser.close()
        print("Serial port closed.")

        

class VibrationFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sampling_started = False
        self.sampling_rate = 30
        self.window_size = 32
        self.fft_size = 32
        self.update_interval_us = 0
        self.frequencies = np.fft.fftfreq(self.fft_size, 1 / self.sampling_rate)[:self.fft_size // 2]
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.distance_range = (20,45)
        self.FFT_magnitude_range = (0, 200)
        self.FFT_frequency_range = (0, self.sampling_rate // 2) 
        self.data = np.zeros(self.window_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.data_storage = []
        self.counter = 0
        self.port = 'COM5'  # 替换为实际的串口端口
        self.baudrate = 57600
        self.initUI()



    def initUI(self):
        print("Start initUI...")
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
        xs = np.linspace(0, self.window_size/self.sampling_rate, self.window_size)  # x轴时间点
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_ylim(self.distance_range)
        self.ax1.set_xlim(0, 1)
        self.ax1.set_title('Real-time Vibration Data', fontsize=18)
        self.ax1.set_ylabel('Amplitude', fontsize=16)
        self.ax1.tick_params(axis='both', which='major', labelsize=12)
        self.line1, = self.ax1.plot(xs, self.data)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setFixedSize(1100, 480)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas1, alignment=Qt.AlignTop)

        # 创建FFT结果图
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlim(self.FFT_frequency_range)
        self.ax2.set_ylim(self.FFT_magnitude_range)
        self.ax2.set_title('Real-time FFT Result', fontsize=18)
        self.ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        self.ax2.set_ylabel('Magnitude', fontsize=16)
        self.ax2.tick_params(axis='both', which='major', labelsize=12)
        self.line2, = self.ax2.plot(self.frequencies, self.fft_result)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setFixedSize(1100, 470)  # 固定窗口大小
        self.left_layout.addWidget(self.canvas2, alignment=Qt.AlignTop)


        # 将左侧布局添加到主布局中
        self.main_layout.addLayout(self.left_layout)

        # 创建右侧垂直布局
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)  # 设置右侧布局的对齐方式

        sol_font = QFont("Arial", 12)  # 设置字体为Arial，大小12
        self.solution_label = QLabel("", self)
        self.solution_label.setWordWrap(True)  # 自动换行
        self.solution_label.setAlignment(Qt.AlignTop)
        self.solution_label.setFont(sol_font)
        self.solution_label.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ddd; background-color: #fff; }")

        # 创建 QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(self.scroll_area, alignment=Qt.AlignTop)
        self.scroll_area.setFixedSize(550, 300)
        # 将 QLabel 设置为 QScrollArea 的小部件
        self.scroll_area.setWidget(self.solution_label)
        
        
        sol_font = QFont("Arial", 12)  # 设置字体为Arial，大小12
        self.analysis = QLabel("", self)
        self.analysis.setWordWrap(True)  # 自动换行
        self.analysis.setAlignment(Qt.AlignTop)
        self.analysis.setFont(sol_font)
        self.analysis.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ddd; background-color: #fff; }")

        # 创建 QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(self.scroll_area, alignment=Qt.AlignTop)
        self.scroll_area.setFixedSize(550, 450)
        # 将 QLabel 设置为 QScrollArea 的小部件
        self.scroll_area.setWidget(self.analysis)
        
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

    def fresh(self):
        self.data = np.zeros(self.window_size)
        self.fft_result = np.zeros(self.fft_size // 2)
        self.data_storage = []
        self.counter = 0

    def start_sampling(self):
        if self.sampling_started == False:
            self.fresh()
            self.initUI()  # 重新初始化UI
            self.sampling_started = True
            self.serial_reader = SerialReader(self.port, self.baudrate)
            self.serial_reader.init_serial()
            self.start_time = time.time()
            self.serial_reader.start_receiving()
            self.timer.start(self.update_interval_us)
        
    
    def update_plot(self):
        new_value = self.serial_reader.read_from_serial()
        self.data_storage.append(new_value)
        
        # 更新实时显示数据
        self.data = np.roll(self.data, -1)
        self.data[-1] = new_value

        self.line1.set_ydata(self.data)
        self.canvas1.draw()

        # 每3次更新一次FFT图表
        if self.counter > 31 and self.counter % 5 == 0:
            self.fft_result = np.abs(np.fft.fft(self.data)[:self.fft_size // 2])
            self.line2.set_ydata(self.fft_result)
            self.canvas2.draw()

            # 找到超过阈值的峰值频率并更新标签
            peak_indices, _ = find_peaks(self.fft_result)
            peak_info = "Peak Frequency:\n"
            peak_info += '\n'.join([f"{self.frequencies[indices]:.2f} Hz: {self.fft_result[indices]:2f}" for indices in peak_indices])
            self.solution_label.setText(f"{peak_info}\n")

        self.counter += 1
        



    def stop_update(self):
        if self.sampling_started == True:
            self.timer.stop()
            self.serial_reader.stop_receiving()
            self.serial_reader.close_serial()
            self.sampling_started = False  # 标志停止采样
            self.stop_time = time.time() - self.start_time  # 获取停止时间
            self.plot_final_data()
           
            


    def plot_final_data(self):
        analysis_result = """Relationship Between Fixture, U-shaped Middle Width, Input Pressure, and Amplitude
Model Fit: The R-squared value is 0.753, indicating that about 75.3% of the variance in Amplitude is explained by the model.
Fixture: The p-value for the fixture type is 0.435, suggesting that the difference between fixed and unfixed fixtures is not statistically significant for amplitude.
U-shaped Middle Width (mm): The p-value is 0.003, indicating a statistically significant relationship with amplitude.
Input Pressure (bars): The p-value is 0.002, also indicating a statistically significant effect on amplitude.

Relationship Between Fixture, U-shaped Middle Width, Input Pressure, and Frequency
Model Fit: The R-squared value is 0.160, meaning only 16% of the variance in Frequency is explained by this model, which is quite low.
Fixture: The p-value is 0.574, suggesting no statistically significant difference in frequency due to the fixture type.
U-shaped Middle Width (mm): The p-value is 0.531, indicating that the middle width does not significantly affect the frequency.
Input Pressure (bars): The p-value is 0.239, also suggesting no significant effect on frequency.
        """
        self.analysis.setText(analysis_result)
        print("plot final distance data")
        final_box_text = ""
        self.ax1.clear()
        x = np.linspace(0, self.stop_time, len(self.data_storage))
        self.ax1.plot(x, self.data_storage)
        self.ax1.set_ylim(self.distance_range)
        self.ax1.set_title(f"Displacement over {self.stop_time:.2f} seconds", fontsize=18)
        self.ax1.set_ylabel("Displacement", fontsize=16)
        self.ax1.tick_params(axis='both', which='major', labelsize=12)
        self.canvas1.draw()  # 更新canvas1
        final_box_text += f"Distance Range:\n{min(self.data_storage)}mm ~ {max(self.data_storage)}mm\n"

        # 在所有数据上进行FFT
        print("plot final FFT result")
        all_data_array = np.array(self.data_storage)
        fft_result = np.fft.fft(all_data_array)
        frequencies = np.fft.fftfreq(len(all_data_array), d=1/self.sampling_rate)
        positive_frequencies = frequencies[:len(all_data_array)//2]
        positive_fft_result = np.abs(fft_result[:len(all_data_array)//2])

        window_length = 11
        polyorder = 2
        smoothed_fft_result = savgol_filter(positive_fft_result, window_length=window_length, polyorder=polyorder)
        print("len of smoothed_fft_result = ", len(smoothed_fft_result))
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
        final_box_text += 'Peak Frequencies:\n'
        for i in range(peak_freq_num):
            final_box_text += f"{peak_frequencies[i]:.2f} Hz: {peak_amplitudes[i]:.2f}\n"
        # self.solution_label.setText(f"Top {peak_freq_num} peak frequencies:{peak_frequencies}\nTop {peak_freq_num} peak amplitudes:{peak_amplitudes}")
        self.solution_label.setText(final_box_text)
        # 找到第peak_freq_num+1高的峰值
        next_peak = sorted_peaks[peak_freq_num] if len(sorted_peaks) > peak_freq_num else None
        next_peak_frequency = positive_frequencies[non_noise_min_index + next_peak] if next_peak is not None else None
        next_peak_amplitude = smoothed_fft_result[non_noise_min_index + next_peak] if next_peak is not None else None

        
        # 更新FFT图像
        self.ax2.clear()
        self.ax2.set_xlim(self.FFT_frequency_range)
        self.ax2.set_ylim(self.FFT_magnitude_range)
        self.ax2.set_title('FFT Result on All Data', fontsize=18)
        self.ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        self.ax2.set_ylabel('Magnitude', fontsize=16)
        self.ax2.tick_params(axis='both', which='major', labelsize=12)
        self.ax2.plot(positive_frequencies, smoothed_fft_result)
        plt.plot(peak_frequencies, peak_amplitudes, 'ro') 
        if next_peak is not None:
            plt.axhline(y=next_peak_amplitude, color='g', linestyle='--', label=f'{peak_freq_num+1} Peak Amplitude ({next_peak_amplitude:.2f})')
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
                text1 = self.solution_label.text()
                text2 = self.analysis.text()
                
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
                    <h2>Text Box 1</h2>
                    <p>{text1}</p>
                    <h2>Text Box 2</h2>
                    <p>{text2}</p>           
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
