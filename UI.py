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
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01) 
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
                new_value = min(new_value, 45.)
                new_value = max(new_value, 20.)
                return new_value

    def start_receiving(self):
        self.stop_event.clear()
        self.ser.reset_input_buffer() 
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
        self.sampling_rate = 100
        self.update_interval_us = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.data_storage = []
        self.port = 'COM5' 
        self.baudrate = 57600
        self.serial_reader = SerialReader(self.port, self.baudrate)
        # get distance range
        self.serial_reader.init_serial()
        self.serial_reader.start_receiving()
        distances = []
        for _ in range(21):
            distances.append(self.serial_reader.read_from_serial())
        self.serial_reader.stop_receiving()
        self.serial_reader.close_serial()
        distances = sorted(distances)
        distances_median = distances[10]
        print(distances)
        print("distance median =", distances_median)
        self.distance_range = (distances_median-0.2, distances_median+0.2)
        self.initUI()



    def initUI(self):
        print("Start initUI...")
        self.setWindowTitle('SU24 ECE/ME/MSE450 Group24: Vibration Detection and Analysis System')
        self.setFixedSize(2000, 1000)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLineEdit {
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50; 
                border: none; 
                color: white;
                padding: 15px 32px; 
                text-align: center; 
                font-size: 24px; 
                margin: 4px 2px; 
                border-radius: 12px; 
            }
            QPushButton:hover {
                background-color: white; 
                color: black; 
                border: 2px solid #4CAF50; 
            }
            QPushButton:pressed {
                background-color: #45a049; 
            }
        """)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create the main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        
        # Create left side layout
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)  

        # Create a QWidget as a container
        self.text_container = QWidget()
        self.text_container.setFixedSize(1100, 900)  
        self.text_layout = QVBoxLayout()
        self.text_layout.setAlignment(Qt.AlignCenter) 

        # Create a centered text box
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setFont(QFont('Arial', 50))

        if self.sampling_started == False:
            self.text_label.setText("Click Start Sampling")
        else:
            self.text_label.setText("Sampling")

        self.text_layout.addWidget(self.text_label)
        self.text_container.setLayout(self.text_layout)
        self.left_layout.addWidget(self.text_container, alignment=Qt.AlignTop)

        self.main_layout.addLayout(self.left_layout)

        # Create the right vertical layout
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop) 

        sol_font = QFont("Arial", 12) 
        self.solution_label = QLabel("", self)
        self.solution_label.setWordWrap(True)  # Auto wrap
        self.solution_label.setAlignment(Qt.AlignTop)
        self.solution_label.setFont(sol_font)
        self.solution_label.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ddd; background-color: #fff; }")

        # create QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(self.scroll_area, alignment=Qt.AlignTop)
        self.scroll_area.setFixedSize(550, 300)
        self.scroll_area.setWidget(self.solution_label)
        
        
        sol_font = QFont("Arial", 12)  
        self.analysis = QLabel("", self)
        self.analysis.setWordWrap(True)  
        self.analysis.setAlignment(Qt.AlignTop)
        self.analysis.setFont(sol_font)
        self.analysis.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ddd; background-color: #fff; }")

        # create QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(self.scroll_area, alignment=Qt.AlignTop)
        self.scroll_area.setFixedSize(550, 450)
        self.scroll_area.setWidget(self.analysis)
        
        # create buttons
        button_layout = QGridLayout()

        # create 4 buttons
        self.button1 = QPushButton("Start Sampling")
        self.button1.clicked.connect(self.start_sampling)
        self.button2 = QPushButton("End Sampling")
        self.button2.clicked.connect(self.stop_update)
        self.button3 = QPushButton("Export .CSV")
        self.button3.clicked.connect(self.export_csv)
        self.button4 = QPushButton("Export Report")
        self.button4.clicked.connect(self.export_report)

        button_layout.addWidget(self.button1, 0, 0)
        button_layout.addWidget(self.button2, 0, 1)
        button_layout.addWidget(self.button3, 1, 0)
        button_layout.addWidget(self.button4, 1, 1)

        self.right_layout.addLayout(button_layout)

        logo_layout = QHBoxLayout()

        # add JI Logo
        logoji_label = QLabel(self)
        pixmap_ji = QPixmap('./UMJI_logo.png')
        logoji_label.setPixmap(pixmap_ji)
        logoji_label.setFixedSize(281, 43)
        logoji_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        logo_layout.addWidget(logoji_label)

        # add Systence Logo
        logoco_label = QLabel(self)
        pixmap_co = QPixmap('./systence_logo.png')
        logoco_label.setPixmap(pixmap_co)
        logoco_label.setFixedSize(91, 63)
        logoco_label.setAlignment(Qt.AlignBottom)
        logo_layout.addWidget(logoco_label)

        self.right_layout.addStretch() 
        self.right_layout.addLayout(logo_layout)

        self.main_layout.addLayout(self.right_layout)


    def start_sampling(self):
        if self.sampling_started == False:
            self.data_storage = []
            self.sampling_started = True
            self.initUI()  
            self.serial_reader = SerialReader(self.port, self.baudrate)
            self.serial_reader.init_serial()
            self.start_time = time.time()
            self.serial_reader.start_receiving()
            self.timer.start(self.update_interval_us)
        
    
    def update_plot(self):
        new_value = self.serial_reader.read_from_serial()
        self.data_storage.append(new_value)


    def stop_update(self):
        if self.sampling_started == True:
            self.timer.stop()
            self.serial_reader.stop_receiving()
            self.stop_time = time.time() - self.start_time  # get stop time
            self.serial_reader.close_serial()
            self.sampling_started = False  
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
        final_box_text = ""
        final_box_text += f"Distance Range:\n{min(self.data_storage)}mm ~ {max(self.data_storage)}mm\n"
        
        while self.left_layout.count():
            child = self.left_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        x = np.linspace(0, self.stop_time, len(self.data_storage))

        # distance final plot
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.plot(x, self.data_storage)
        self.ax1.set_ylim(self.distance_range)
        self.ax1.set_title(f"Displacement over {self.stop_time:.2f} seconds", fontsize=18)
        self.ax1.set_ylabel("Displacement", fontsize=16)
        self.ax1.tick_params(axis='both', which='major', labelsize=12)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setFixedSize(1100, 480)  
        self.left_layout.addWidget(self.canvas1, alignment=Qt.AlignTop)
        self.canvas1.draw()  
        
        # conduct FFT on all collected data
        all_data_array = np.array(self.data_storage)
        fft_result = np.fft.fft(all_data_array)
        frequencies = np.fft.fftfreq(len(all_data_array), d=1/self.sampling_rate)
        positive_frequencies = frequencies[:len(all_data_array)//2]
        positive_fft_result = np.abs(fft_result[:len(all_data_array)//2])
        noise_freq = 2
        non_noise_min_index = next((i for i, freq in enumerate(positive_frequencies) if freq > noise_freq), None)
        if non_noise_min_index is None:
            raise ValueError("No frequencies found above the noise frequency threshold")
        peaks, _ = find_peaks(positive_fft_result[non_noise_min_index:])
        # sort peaks and select 3 peaks
        peak_freq_num = 3
        sorted_peaks = sorted(peaks, key=lambda x: positive_fft_result[non_noise_min_index + x], reverse=True)
        top_peaks = sorted_peaks[:peak_freq_num]
        peak_frequencies = positive_frequencies[non_noise_min_index:][top_peaks]
        peak_amplitudes = positive_fft_result[non_noise_min_index:][top_peaks]
        final_box_text += 'Peak Frequencies:\n'
        for i in range(peak_freq_num):
            final_box_text += f"{peak_frequencies[i]:.2f} Hz: {peak_amplitudes[i]:.2f}\n"
        self.solution_label.setText(final_box_text)
        # find peak_freq_num+1 highest peak
        next_peak = sorted_peaks[peak_freq_num] if len(sorted_peaks) > peak_freq_num else None
        next_peak_frequency = positive_frequencies[non_noise_min_index + next_peak] if next_peak is not None else None
        next_peak_amplitude = positive_fft_result[non_noise_min_index + next_peak] if next_peak is not None else None
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlim(noise_freq, self.sampling_rate // 2)
        self.ax2.set_ylim(next_peak_amplitude, max(positive_fft_result[non_noise_min_index:])*1.2)
        self.ax2.set_title('FFT Result on All Data', fontsize=18)
        self.ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        self.ax2.set_ylabel('Magnitude', fontsize=16)
        self.ax2.tick_params(axis='both', which='major', labelsize=12)
        self.ax2.plot(positive_frequencies, positive_fft_result)
        self.ax2.plot(peak_frequencies, peak_amplitudes, 'ro') 
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setFixedSize(1100, 470)  
        self.left_layout.addWidget(self.canvas2, alignment=Qt.AlignTop)
        self.canvas2.draw() 
        print("plot final distance and FFT figures")


    def export_csv(self):
        if self.sampling_started == False:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_file_name = f"report_{current_time}.html"

            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Report", default_file_name, "HTML Files (*.html);;All Files (*)", options=options)
            if filePath:
                distance_image_path = "distance_plot.png"
                fft_image_path = "fft_plot.png"
                self.fig1.savefig(distance_image_path)
                self.fig2.savefig(fft_image_path)
                text1 = self.solution_label.text()
                text2 = self.analysis.text()
                
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
                with open(filePath, 'w') as file:
                    file.write(html_content)

                print(f"Report saved to {filePath}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VibrationFFTApp()
    ex.show()
    sys.exit(app.exec_())
