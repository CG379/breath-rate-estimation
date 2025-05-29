import serial
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Configuration for the serial port
SERIAL_PORT = 'COM3'  # Replace with your serial port (e.g., COM3, /dev/ttyUSB0)
BAUD_RATE = 115200      # Match the baud rate of your device
TIMEOUT = 1           # Timeout in seconds for serial read

# Generate a unique filename for the CSV file with a timestamp
CSV_FILE = f"serial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def read_serial_data(serial_port, baud_rate, timeout=1):
    """Read data continuously from the serial port."""
    ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
    print(f"Connected to {serial_port} at {baud_rate} baud.")
    return ser

def save_data_to_csv(data, filename):
    """Save a list of data to a CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def plot_data_from_csv(filename):
    """Plot data from a CSV file."""
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)
    plt.figure(figsize=(10, 6))

    # Assuming the CSV has two columns: time and value
    plt.plot(df['time'], df['value'], label='Serial Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Serial Data Plot')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Initialize the serial connection
    ser = read_serial_data(SERIAL_PORT, BAUD_RATE, TIMEOUT)

    # Save header to the new CSV file
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'value'])
    print(f"Data will be saved to {CSV_FILE}")
    # Start reading data from the serial port
    print("Starting data collection. Press Ctrl+C to stop.")
    try:
        start_time = time.time()
        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Convert the line to a float or int if possible
                try:
                    value = float(line)
                    timestamp = time.time() - start_time
                    # print(f"Time: {timestamp:.2f}, Value: {value}")

                    # Save the data to the CSV file
                    save_data_to_csv([[timestamp, value]], CSV_FILE)
                except ValueError:
                    print(f"Invalid data received: {line}")

    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        ser.close()
        print("Serial port closed.")

    # Plot the data
    plot_data_from_csv(CSV_FILE)

if __name__ == '__main__':
    main()