import serial.tools.list_ports
# Use find_peaks from scipy.signal and get breath rate from peaks
from scipy.signal import butter, lfilter, find_peaks
import numpy as np
from collections import deque
import time
import os
import matplotlib.pyplot as plt

''' Assumptions:
Breathing signals are low-frequency (~0.2 - 0.5 Hz)
Any drift will be < 0.05Hz

'''
# TODO: fix breath rate for stm timer, data fusion
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


def butter_lowpass_coeffs(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='low')

def apply_lowpass(new_sample, zf, b, a):
    """Apply single-step IIR low-pass filter with filter memory (state zf)."""
    output, zf = lfilter(b, a, [new_sample], zi=zf)
    return output[0], zf



class BreathRateEstimator:
    def __init__(self, buffer_size=750, sampling_rate=50, window_size=5, drift_window=200):
        self.fs = sampling_rate
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.b, self.a = butter_lowpass_coeffs(2.0, sampling_rate)
        self.zf = np.zeros(max(len(self.a), len(self.b)) - 1)
        self.window_size = window_size
        self.buffer = deque(maxlen=buffer_size) # use singular buffer if we want to use FFT
        self.filtered_buffer = deque(maxlen=drift_window)
        self.smooth_buffer = deque(maxlen=window_size)
        self.output_buffer = deque(maxlen=sampling_rate * 15)

    def update(self, new_sample, timestamp):
        #TODO: rework DSP pipeline, include FFT somewhere? 
        # Low-pass filter
        filtered, self.zf = apply_lowpass(new_sample, self.zf, self.b, self.a)
        
        # Drift removal (rolling mean)
        # TODO: fix drift removal
        self.filtered_buffer.append(filtered)
        baseline = np.mean(self.filtered_buffer)
        detrended = filtered - baseline

        # Smoothing (moving average)
        self.smooth_buffer.append(detrended)
        smoothed = np.mean(self.smooth_buffer)

        # Save to buffers
        self.output_buffer.append(smoothed)
        self.buffer.append(smoothed)  # For peak detection
        self.timestamp_buffer.append(float(timestamp))
        return smoothed

    def estimate_breath_rate(self):
        """Estimate breath rate from buffered signal and actual timestamps.
        1. Find peaks in the signal
        2. Calculate intervals between peaks
        3. Calculate average interval time
        4. Convert to breaths per minute
        """
        if len(self.buffer) < self.fs * 5:
            return None  # Not enough data

        signal = np.array(self.buffer)
        timestamps = np.array(self.timestamp_buffer)
        
        # STM outputs in miliseconds, convert to seconds
        # 
        timestamps_seconds = timestamps * 0.1
        
        peaks, _ = find_peaks(signal, distance=self.fs * 0.5)  # Can adjust threshold
        if len(peaks) < 2:
            return None

        peak_times = timestamps_seconds[peaks]
        intervals = np.diff(peak_times)

        avg_breath_time = np.mean(intervals)
        return 60 / avg_breath_time  # breaths per minute
    
    def fft_breath_rate(self):
        # TODO: Check if full DSP pipline is needed before FFT
        # Drift removal definately needed for FFT, others?
        if len(self.buffer) < self.fs * 5:
            return None  # Not enough data
        signal = np.array(self.buffer)
        # Experiment with hamming or han window
        wavelet = signal * np.hamming(len(signal))
        #wavelet = signal * np.hanning(len(signal))
        # FFT
        fft_result = np.fft.rfft(wavelet)
        # get rid of phase information
        fft_magnitude = np.abs(fft_result)

        # Get frequency axis
        freq = np.fft.rfftfreq(len(signal), d=1/self.fs)
        
        # Find peaks in frequency domain (within breathing range)
        breathing_range_mask = (freq >= 0.1) & (freq <= 1.1)  # 6-60 BPM
        valid_freqs = freq[breathing_range_mask]
        valid_magnitudes = fft_magnitude[breathing_range_mask]
        # TODO: check if thisw is enough for not on a person 
        if len(valid_magnitudes) == 0:
            return None
        
        # Find dominant frequency
        dominant_idx = np.argmax(valid_magnitudes)
        dominant_freq = valid_freqs[dominant_idx]
        
        # Convert to BPM
        return dominant_freq * 60




class BayesFusion:
    def __init__(self, hypothesis_range=(6, 60, 0.1)):
        self.hypotheses = np.arange(*hypothesis_range)
        self.n_hyp = len(self.hypotheses)
        # Prior = expected distribution of breath rates
        self.prior = np.ones(self.n_hyp) / self.n_hyp
        
        # TODO: Find correct sigmas
        # Higher sigma = less reliable
        # Lower sigma = more reliable
        self.sigma_temp = 0.05
        self.sigma_stretch = 0.05


        # For visualization
        self.latest_fusion = None
        self.latest_rates = [None, None]

    def likelihood(self, measured_rate, hypothesis_rates, sigma):
        """Calculate likelihood of hypotheses given a measurement"""
        return np.exp(-0.5 * ((measured_rate - hypothesis_rates) / sigma) ** 2)
    
    def fuse_estimates(self, rate1, rate2):
        self.latest_rates = [rate1, rate2]

        # If either rate is None, return the other one
        if rate1 is None and rate2 is None:
            return None
        elif rate1 is None:
            return rate2
        elif rate2 is None:
            return rate1
        
        # Calculate likelihoods
        L_sensor1 = self.likelihood(rate1, self.hypotheses, self.sigma_temp)
        L_sensor2 = self.likelihood(rate2, self.hypotheses, self.sigma_stretch)

        # Posterior is probability distribution of each possible breath rate is, 
        # given the evidence from both sensors
        # Combine evidence using Bayes' rule
        unnormalized_posterior = L_sensor1 * L_sensor2 * self.prior
        # Normalize the posterior
        posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

        # Store for visualization
        self.latest_fusion = {
            'L_sensor1': L_sensor1,
            'L_sensor2': L_sensor2,
            'posterior': posterior
        }

        # MAP estimate = max probability in posterior distrobution
        map_estimate = self.hypotheses[np.argmax(posterior)]
        
        return map_estimate

    def visualize_fusion(self):
        # TODO: Figure out how do continously update the plot
        return



# Create a SerialConnection instance
serial_conn = SerialConnection(baudrate=115200)

if not serial_conn.select_port():
    print("Failed to establish connection. Exiting.")
    exit()

serialInst = serial_conn.serialInst



# Main loop for reading data
def main():
    # Buffer length (seconds) = buffer_size / sampling_rate
    # Put whatever sampeling rate the STM32 is using
    sensor1_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=50)
    sensor2_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=50)

    # For fusion, assume everything will fall in 6-60 BPM range
    fusion = BayesFusion(hypothesis_range=(6, 60, 0.1))

    running = True
    
    # Signal STM32 to prepare
    serialInst.write("SX".encode('utf-8'))
    serialInst.flush()

    # Save sensor data with timestamp in the filename
    current_time = time.strftime("%H-%M")

    # Add a wait to arm device if we decide to do that here
    # Wait for the STM32 to send "START"
    print("Waiting for STM32 to be ARMED (press the button)...")
    while True:
        if serialInst.in_waiting:
            line = serialInst.readline().decode('utf-8').strip()
            if line == "ARMED":
                print("STM32 started. Beginning data collection.")
                break
    print("Press ESC to stop the program.")
    
    os.makedirs("data", exist_ok=True)
    sensor_data_filename = f"./data/sensor_data_{current_time}.csv"
    with open(sensor_data_filename, "w", buffering=1) as f:
        f.write("Timestamp,Sensor1,Sensor2,BreathRate1,BreathRate2,FusedBreathRate\n")
        # TODO: might do Caibration phase if time allows
        try:
            while running:
                line = serialInst.readline().decode('utf-8').strip()
                if line:
                    try:
                        timestamp, value1, value2 = line.split(",")
                        timestamp = float(timestamp)
                        # For debugging
                        # DPS pipeline
                        # TODO: campture sample data of each DSP layer for report
                        smoothed1 = sensor1_estimator.update(float(value1), timestamp)
                        smoothed2 = sensor2_estimator.update(float(value2), timestamp)

                        # TODO: Check if thermistor setup works with FFT estimation
                        rate1 = sensor1_estimator.fft_breath_rate()
                        rate2 = sensor2_estimator.fft_breath_rate()

                        if rate1 and rate2:
                            fused_rate = fusion.fuse_estimates(rate1, rate2)
                            # TODO: Use better data fusion if if it is discussed in workshop
                            # One will have more confidence than the other
                            # Week 11 workshop
                            if fused_rate is not None:
                                print(f"Fused Breath Rate: {fused_rate:.2f} bpm")
                                f.write(f"{timestamp},{value1},{value2},{rate1},{rate2},{fused_rate}\n")

                                # TODO: Add visualisation here
                                # Update every 100 samples? 

                    except KeyboardInterrupt:
                        print("Exiting...")
                        running = False
                    except ValueError:
                        if line == "DISARMED":
                            print("STM32 stopped. Exiting...")
                            running = False
                        else:
                            print(f"tf is this sh: {line}")
                        continue
        except Exception as e:
            print(f"Error: {e}")

    serial_conn.disconnect()
            
    print(f"Sensor data saved to {sensor_data_filename}")

if __name__ == "__main__":
    main()




'''

Initial thoughts on DSP pipeline:
1. Low pass filter to remove high frequency noise
2. Remove linear drift (if any)
3. Smooth the data (might not be needed)

Decided on Late fusion: DSP each sensor data seperately, estimate breath rate of each, then combine 
Reasons:
1. Each sensor is different, merging raw values might obscure individual signal characteristics
2. Improved robustness to noise if one signal drops out/is noisy
3. Easier to make, debug and expand


TODO: Figure out method for collection 'true' breath rate data for report analysis
'''
