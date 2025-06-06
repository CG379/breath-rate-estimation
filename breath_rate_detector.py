import serial.tools.list_ports
# Use find_peaks from scipy.signal and get breath rate from peaks
from scipy.signal import butter, sosfilt_zi, sosfilt, find_peaks, welch, savgol_filter
import numpy as np
from arraydeque import ArrayDeque as deque
import time
import os
import matplotlib.pyplot as plt

''' Assumptions:
Breathing signals are low-frequency (~0.2 - 0.5 Hz)
Any drift will be < 0.05Hz

'''

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


def butter_lowpass_sos(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='low', output='sos')

def butter_highpass_sos(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='high', output='sos')


class BreathRateEstimator:
    def __init__(self, sampling_rate=10, min_duration=5, max_duration=30):
        self.fs = sampling_rate
        self.min_duration = min_duration  # minimum seconds needed for analysis
        self.max_duration = max_duration  # maximum seconds to keep
        
        # Initialize filters
        self.sos_lp = butter_lowpass_sos(2, sampling_rate)
        self.sos_hp = butter_highpass_sos(0.1, sampling_rate)
        self.zi_lp = None
        self.zi_hp = None
        
        # Single buffer for processed signal
        max_samples = int(self.fs * self.max_duration)
        self.signal = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        
        # Rate estimation history for stability
        self.rate_history = deque(maxlen=5)

    def update(self, value, timestamp):
        """Process and store a new sample"""
        
        # Initialize filter states on first sample
        if self.zi_lp is None:
            self.zi_lp = sosfilt_zi(self.sos_lp) * value
            self.zi_hp = sosfilt_zi(self.sos_hp) * value
        
        # Apply filters in sequence
        filtered, self.zi_lp = sosfilt(self.sos_lp, [value], zi=self.zi_lp)
        detrended, self.zi_hp = sosfilt(self.sos_hp, filtered, zi=self.zi_hp)
        
        # Store processed signal
        self.signal.append(detrended[0])
        self.timestamps.append(timestamp)
        
        return detrended[0]

    def get_signal_array(self):
        """Get signal as numpy array with preprocessing"""
        if len(self.signal) < max(self.fs * self.min_duration, self.fs * 60 / 6 * 3):
            return None
            
        signal = np.array(self.signal)
        
        # Normalize
        signal = signal - np.mean(signal)
        std = np.std(signal)
        if std > 1e-6:
            signal = signal / std
            
        return signal


    def peak_breath_rate(self):
        """Estimate breathing rate using peak detection"""
        signal = self.get_signal_array()
        if signal is None:
            return None, None
        
        # Smooth signal while preserving peaks
        win_len = min(len(signal) // 5, 25)
        if win_len % 2 == 0:
            win_len += 1
        if win_len >= 5:
            signal = savgol_filter(signal, window_length=win_len, polyorder=3)
        
        # Adaptive peak detection parameters
        noise_level = np.std(np.diff(signal)) / np.sqrt(2)
        min_prominence = max(0.3, 0.5 * noise_level)

        if self.rate_history and np.median(self.rate_history) < 9:
            min_prominence *= 0.8

        # Expected breathing rate constraints
        if self.rate_history:
            expected_rate = np.median(self.rate_history)
            min_distance = int(self.fs * 60 / (expected_rate * 1.5))
            max_distance = int(self.fs * 60 / (expected_rate * 0.5))
        else:
            min_distance = int(self.fs * 2)    # 30 BPM max
            max_distance = int(self.fs * 10)   # 6 BPM min
        
        # Find peaks
        peaks, _ = find_peaks(
            signal,
            distance=min_distance,
            prominence=min_prominence
            #height=np.percentile(signal, 60)
        )
        
        if len(peaks) < 3:
            return None, None
        
        # Calculate intervals
        intervals = np.diff(peaks) / self.fs
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        valid = (intervals >= q1 - 1.5*iqr) & (intervals <= q3 + 1.5*iqr)
        
        if np.sum(valid) < 2:
            return None, None
            
        clean_intervals = intervals[valid]
        
        # Calculate rate
        median_interval = np.median(clean_intervals)
        rate = 60.0 / median_interval
        
        # Confidence based on consistency
        if len(clean_intervals) > 2:
            cv = np.std(clean_intervals) / median_interval
            confidence = 1.0 / (1.0 + cv * 5)
        else:
            confidence = 0.2
        
        return np.clip(rate, 6, 30), confidence
        
    def fft_breath_rate(self):
        """Estimate breathing rate using frequency analysis"""
        signal = self.get_signal_array()
        if signal is None or len(signal) < self.fs * 10:
            return None, None
        
        # Use Welch's method for robust spectral estimation
        nperseg = min(len(signal) // 4, int(self.fs * 5))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg, 
                          detrend='linear', scaling='density')
        
        # Focus on breathing frequency range
        mask = (freqs >= 0.1) & (freqs <= 0.5)  # 6-30 BPM
        breath_freqs = freqs[mask]
        breath_psd = psd[mask]
        
        if len(breath_psd) == 0:
            return None, None
        
        # Find dominant frequency
        peak_idx = np.argmax(breath_psd)
        peak_freq = breath_freqs[peak_idx]
        peak_power = breath_psd[peak_idx]
        
        # Estimate SNR for confidence
        noise_power = np.median(breath_psd)
        snr = peak_power / (noise_power + 1e-10)
        confidence = np.tanh(snr / 10)  # Maps SNR to 0-1 range
        
        rate = peak_freq * 60
        return np.clip(rate, 6, 30), confidence

    def combined_breath_rate(self):
        """Combined rate estimation with intelligent fusion"""
        # Try both methods
        rate_peaks, conf_peaks = self.peak_breath_rate()
        rate_fft, conf_fft = self.fft_breath_rate()
        
        # Handle cases where one method fails
        if rate_peaks is None and rate_fft is None:
            return None, None
        if rate_peaks is None:
            result = rate_fft, conf_fft
        elif rate_fft is None:
            result = rate_peaks, conf_peaks
        else:
            # Both methods succeeded - weighted average
            if abs(rate_peaks - rate_fft) > 3:
                # Methods disagree - trust higher confidence
                if conf_peaks > conf_fft:
                    result = rate_peaks, conf_peaks
                else:
                    result = rate_fft, conf_fft
            else:
                # Methods agree - combine them
                total_conf = conf_peaks + conf_fft
                rate = (rate_peaks * conf_peaks + rate_fft * conf_fft) / total_conf
                confidence = min(1.0, total_conf / 1.5)
                result = rate, confidence
        
        # Update history
        if result[0] is not None:
            self.rate_history.append(result[0])
            
        return result
    
    def get_buffer_duration(self):
        """Get current buffer duration in seconds"""
        return len(self.signal) / self.fs


class BayesFusion:
    def __init__(self, min_rate=5, max_rate=25, resolution=0.1):
        self.hypotheses = np.arange(min_rate, max_rate, resolution)
        self.prior = np.ones(len(self.hypotheses)) / len(self.hypotheses)
        self.current_posterior = self.prior.copy()
        self.latest_rate = None
        self.latest_sigma = None

    def update(self, rate1, sigma1, rate2, sigma2):
        posterior = self.current_posterior.copy()
        log_L_total = np.zeros_like(posterior)
        log_weights = 0

        if rate1 is not None and sigma1 is not None:
            sigma1 = max(sigma1, 0.1)
            log_L1 = -0.5 * ((rate1 - self.hypotheses) / sigma1) ** 2
            log_L1 -= np.max(log_L1)  # for numerical stability
            log_L_total += log_L1
            log_weights += 1

        if rate2 is not None and sigma2 is not None:
            sigma2 = max(sigma2, 0.1)
            log_L2 = -0.5 * ((rate2 - self.hypotheses) / sigma2) ** 2
            log_L2 -= np.max(log_L2)
            log_L_total += log_L2
            log_weights += 1

        if log_weights > 0:
            log_prior = np.log(self.current_posterior + 1e-12)
            log_combined = log_L_total + log_prior
            log_combined -= np.max(log_combined)
            posterior = np.exp(log_combined)
            posterior /= np.sum(posterior)
            self.current_posterior = posterior

            self.latest_rate = self.hypotheses[np.argmax(posterior)]
            self.latest_sigma = np.sqrt(np.sum(posterior * 
                                               (self.hypotheses - self.latest_rate) ** 2))
        return self.latest_rate

    def reset_prior(self, center=None):
        if center is None:
            self.current_posterior = np.ones(len(self.hypotheses)) / len(self.hypotheses)
        else:
            prior_sigma = 2.0
            self.current_posterior = np.exp(-0.5 * ((self.hypotheses - center) / prior_sigma) ** 2)
            self.current_posterior /= np.sum(self.current_posterior)

    def get_distribution(self):
        return self.hypotheses, self.current_posterior

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
    sensor1_estimator = BreathRateEstimator(sampling_rate=10)
    sensor2_estimator = BreathRateEstimator(sampling_rate=10)

    # For fusion, assume everything will fall in 6-60 BPM range
    fusion = BayesFusion(min_rate=6, max_rate=25, resolution=0.1)

    running = True
    
    # Signal STM32 to prepare
    serialInst.write("SX".encode('utf-8'))
    serialInst.flush()

    # Save sensor data with timestamp in the filename
    current_time = time.strftime("%H-%M")

    # Add a wait to arm device if we decide to do that here
    # Wait for the STM32 to send "START"
    print("Waiting for STM32 to be ARMED...")
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

        try:
            while running:
                line = serialInst.readline().decode('utf-8').strip()
                if line:
                    try:
                        # time, conductive band, tempo sensor
                        timestamp, value1, value2 = line.split(",")
                        timestamp = float(timestamp)/10
                        # For debugging
                        # DPS pipeline

                        smoothed1 = sensor1_estimator.update(float(value1), timestamp)
                        smoothed2 = sensor2_estimator.update(float(value2), timestamp)

                        rate1, sigma1 = sensor1_estimator.combined_breath_rate()
                        rate2, sigma2 = sensor2_estimator.combined_breath_rate()

                        if rate1 and rate2:
                            fused_rate = fusion.update(rate1, sigma1, rate2, sigma2)
                            if fused_rate is not None:
                                print(f"Fused Breath Rate: {fused_rate:.2f} bpm")
                                f.write(f"{timestamp},{value1},{value2},{rate1},{rate2},{fused_rate}\n")

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

    # os.makedirs("data/figs", exist_ok=True)
    # sensor1_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor1_{timestamp}.png")
    # sensor2_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor2_{timestamp}.png")
    # print("DSP pipeline visualizations saved.")

if __name__ == "__main__":
    main()



