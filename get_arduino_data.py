import serial
import sys

def main():
    try:
        # 替换为你的串口端口和波特率
        ser = serial.Serial('/dev/ttyACM0', 921600)  
        ser.flush()
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        sys.exit(1)
    
    print("Starting to read from serial port...")
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                print(line)
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == '__main__':
    main()
