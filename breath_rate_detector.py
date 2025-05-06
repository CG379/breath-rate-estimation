import serial.tools.list_ports
# Use find_peaks from scipy.signal and get breath rate from peaks
from scipy.signal import butter, lfilter, find_peaks
import numpy as np

class SerialConnection:
    def __init__(self, baudrate=115200):
        self.serialInst = serial.Serial()
        self.baudrate = baudrate
        self.port = None
    
    def list_ports(self):
        '''List all ports to connect to, change as needed'''
        print("---- PRINTING PORTS AVALIABLE ----")
        ports = serial.tools.list_ports.comports()
        # Extract names of ports
        portsList = [str(port).split()[0] for port in ports]
        for port in portsList:
            print(port)
        return portsList
    
    def select_port(self):
        """Prompt user to select a valid COM port and establish connection."""
        ports_list = self.list_ports()
        
        if not ports_list:
            return False  # No ports available

        port_number = input("Select COM Port (just the number): ").strip()
        self.port = None  # Reset port selection

        for port_desc in ports_list:
            if port_desc.startswith("COM" + port_number):
                self.port = "COM" + port_number
                print(f"Selected port: {self.port}")
                break

        if self.port is None:
            print("Error: Selected COM port not found.")
            return False  # Exit on invalid selection

        # Configure serial instance
        self.serialInst.baudrate = self.baudrate
        self.serialInst.port = self.port

        try:
            self.serialInst.open()
            print(f"Connected to {self.port}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def disconnect(self):
        '''Close the serial connection'''
        if self.serialInst.is_open:
            self.serialInst.close()
            print("Connection closed")

# Create a SerialConnection instance
serial_conn = SerialConnection(baudrate=115200)

if not serial_conn.select_port():
    print("Failed to establish connection. Exiting.")
    exit()

serialInst = serial_conn.serialInst


'''
TODO: Decide on circuit lowpass filter or software low pass filter

Initial thoughts on DSP pipeline:
1. Low pass filter to remove high frequency noise
2. Remove linear drift (if any)
3. Smooth the data (might not be needed)

Then find peaks and exstimate 

TODO: Fix for real time estimation
Use a rolling buffer to store the last N samples and apply DSP pipeline to that buffer in real time

TODO: Decide on early, mid or late data fusion
4. Early fusion: DSP each sensor data seperately, then combine, then estimate breath rate
5. Mid fusion: Data fusion somewhere in the middle of DSP pipeline, then estimate breath rate
6. Late fusion: DSP each sensor data seperately, estimate breath rate of each, then combine 


TODO: Figure out method for collection 'true' breath rate data for report analysis
'''

def low_pss_filter(data, order=5, cutoff=0.1, fs=1.0):
    """ Can do other filter if needed
    Low-pass Butterworth filter.
    data: Input data to be filtered.
    order: Order of the filter.
    cutoff: Cutoff frequency as a fraction of the Nyquist frequency.
    fs: Sampling frequency.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def remove_drift(data):
    """Remove linear drift from the data."""
    return data - np.poly1d(np.polyfit(range(len(data)), data, 1))(range(len(data)))

def smooth_data(data, window_size=5):
    """Smooth the data using a moving average filter"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')