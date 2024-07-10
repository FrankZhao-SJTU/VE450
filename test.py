import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

# 采样频率3KHz，每秒3000个数据点
sampling_rate = 50
# 每次更新动画的间隔时间为1毫秒（1000 Hz）
update_interval = 2

def main():
    try:
        # 替换为你的串口端口和波特率
        ser = serial.Serial('/dev/ttyACM0', 57600)
        ser.flush()
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        sys.exit(1)
    
    print("Starting to read from serial port...")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    xs = np.linspace(0, 1, sampling_rate)  # x轴时间点
    ys = np.zeros(sampling_rate)  # y轴距离数据

    line1, = ax1.plot(xs, ys)
    line2, = ax2.plot(xs, np.zeros(sampling_rate))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance')
    ax1.set_title('Real-time Distance Data')
    # ax1.set_ylim(35.4, 35.8)  # 设置y轴的范围，根据实际需要调整
    ax1.set_ylim(20, 45)  # 设置y轴的范围，根据实际需要调整

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT of Distance Data')
    # ax2.set_xlim(0, sampling_rate / 2)
    ax2.set_xlim(5, 20)
    ax2.set_ylim(0, 10)  # 根据实际数据范围调整

# 12

    def update(frame):
        nonlocal ys
        try:
            for i in range(sampling_rate):
                print(ser.in_waiting)
                if ser.in_waiting > 0:
                    data_line = ser.readline().decode('utf-8').rstrip()
                    distance = float(data_line)
                    ys = np.roll(ys, -1)
                    ys[-1] = distance
            line1.set_ydata(ys)

            # 计算FFT
            fft_result = np.fft.fft(ys)
            fft_magnitude = np.abs(fft_result)[:sampling_rate // 2]
            fft_freqs = np.fft.fftfreq(sampling_rate, 1 / sampling_rate)[:sampling_rate // 2]
            line2.set_data(fft_freqs, fft_magnitude)
        except ValueError:
            pass

        return line1, line2

    ani = animation.FuncAnimation(fig, update, interval=update_interval, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == '__main__':
    main()


