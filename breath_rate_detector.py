import serial.tools.list_ports
# Use find_peaks from scipy.signal and get breath rate from peaks
from scipy.signal import butter, sosfilt_zi, sosfilt, find_peaks, welch, savgol_filter
import numpy as np
from collections import deque
import time
import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime

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
    def __init__(self, buffer_size=750, sampling_rate=10, window_size=3):
        self.fs = sampling_rate
        self.window_size = window_size
        
        # Use SOS format for better numerical stability
        self.sos_lp = butter_lowpass_sos(2, sampling_rate)
        self.sos_hp = butter_highpass_sos(0.1, sampling_rate)  # Increased cutoff
        
        # Get steady-state initial conditions
        self.zi_lp_unit = sosfilt_zi(self.sos_lp)
        self.zi_hp_unit = sosfilt_zi(self.sos_hp)
        
        # Will be initialized on first sample
        self.zi_lp = None
        self.zi_hp = None
        self.first_sample = True
        
        # Buffers
        self.raw_buffer = deque(maxlen=buffer_size) 
        self.filtered_buffer = deque(maxlen=buffer_size)
        self.no_drift_buffer = deque(maxlen=buffer_size)
        self.buffer = deque(maxlen=buffer_size) 
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.last_rate = None

    def update(self, new_sample, timestamp):
        # Initialize filter states on first sample
        if self.first_sample:
            self.zi_lp = self.zi_lp_unit * new_sample
            self.zi_hp = self.zi_hp_unit * new_sample
            self.first_sample = False

        # Stage 0: Raw
        self.raw_buffer.append(new_sample)

        # Stage 1: Low-pass filter (using SOS)
        filtered, self.zi_lp = sosfilt(self.sos_lp, [new_sample], zi=self.zi_lp)
        self.filtered_buffer.append(filtered[0])

        # Stage 2: High-pass filter (using SOS)
        detrended, self.zi_hp = sosfilt(self.sos_hp, [filtered[0]], zi=self.zi_hp)
        self.no_drift_buffer.append(detrended[0])

        # Stage 3: Rolling average smoothing
        if len(self.buffer) > 0:
            alpha = 0.4
            smoothed = alpha * detrended[0] + (1 - alpha) * self.buffer[-1]
        else:
            smoothed = detrended[0]
        
        self.buffer.append(smoothed)
        self.timestamp_buffer.append(float(timestamp))

        return smoothed

    def peak_breath_rate(self):
         # Reduce minimum data requirement
        min_samples = int(self.fs * 5)  # 5 seconds
        if len(self.buffer) < min_samples:
            return None, None
        
        # Use adaptive window
        max_window = int(self.fs * 30)  # 30 seconds max
        window_size = min(len(self.buffer), max_window)
        signal = np.array(list(self.buffer)[-window_size:])
        
        # Better preprocessing
        #signal = detrend(signal, type='linear')
        
        # Apply Savitzky-Golay filter for smoothing while preserving peaks
        win_len = min(len(signal) // 5 * 2 + 1, 51)
        if win_len >= 5:
            signal = savgol_filter(signal, window_length=win_len, polyorder=3)
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        # Dynamic peak detection parameters based on signal characteristics
        # Estimate noise level
        noise_std = np.std(np.diff(signal)) / np.sqrt(2)
        
        # Adaptive parameters
        min_prominence = max(0.5 * noise_std, 0.2)
        min_height = np.percentile(signal, 60)

        if self.last_rate:
            expected_period = self.fs * 60 / self.last_rate
            expected_period = np.clip(expected_period, self.fs*2, self.fs*10)
        
        # Expected distance between peaks (with wider tolerance)
        if hasattr(self, 'last_rate') and self.last_rate:
            expected_period = self.fs * 60 / self.last_rate
            min_distance = int(expected_period * 0.5)
            max_distance = int(expected_period * 1.5)
        else:
            min_distance = int(self.fs * 60 / 30)  # 30 BPM max
            max_distance = int(self.fs * 60 / 4)   # 6 BPM min
        
        # Find peaks
        peaks, properties = find_peaks(
            signal,
            distance=min_distance,
            prominence=min_prominence,
            height=min_height
        )
        
        # Filter peaks that are too far apart
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            valid_mask = intervals <= max_distance
            # Keep peaks that form valid intervals
            valid_peaks = [peaks[0]]
            for i in range(len(intervals)):
                if valid_mask[i]:
                    valid_peaks.append(peaks[i+1])
            peaks = np.array(valid_peaks)
        
        if len(peaks) < 3:
            return None, None
        
        # Calculate intervals with outlier rejection
        intervals = np.diff(peaks) / self.fs
        
        # Use median absolute deviation for robust outlier detection
        median_interval = np.median(intervals)
        mad = np.median(np.abs(intervals - median_interval))
        threshold = 3 * mad
        
        if mad > 0:
            mask = np.abs(intervals - median_interval) <= threshold
            clean_intervals = intervals[mask]
        else:
            clean_intervals = intervals
        
        if len(clean_intervals) < 2:
            return None, None
        
        # Use median for robustness
        avg_period = np.median(clean_intervals)
        
        # Calculate rate
        rate = 60.0 / avg_period
        
        # Constrain to reasonable bounds
        rate = np.clip(rate, 6, 30)
        
        # Store for next iteration
        self.last_rate = rate
        
        # Calculate uncertainty based on interval consistency
        if len(clean_intervals) > 2:
            cv = np.std(clean_intervals) / (avg_period + 1e-6)  # Coefficient of variation
            sigma = max(0.5, min(2.0, cv * 5))  # Scale CV to uncertainty
        else:
            sigma = 1.0  # Higher uncertainty with few intervals
        
        return rate, sigma
        
    def fft_breath_rate(self):
        if len(self.buffer) < self.fs * 10:  # Need more data
            return None, None
        
        buffer_duration = 30  # seconds
        window_size = int(self.fs * buffer_duration)
        if len(self.buffer) > window_size:
            signal = np.array(list(self.buffer)[-window_size:])
        else:
            signal = np.array(self.buffer)
        
        #signal = detrend(np.array(self.buffer))
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        # Zero-pad for better frequency resolution
        n_fft = 2 ** int(np.ceil(np.log2(len(signal) * 4)))
        
        # Use Welch's method for more robust spectral estimation
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(len(signal)//4, 256), 
                        nfft=n_fft, detrend='constant')
        
        # Focus on breathing range
        mask = (freqs >= 0.08) & (freqs <= 0.5)  # 6-30 BPM
        valid_freqs = freqs[mask]
        valid_psd = psd[mask]
        
        if len(valid_psd) == 0:
            return None, None
        
        # Find peaks in PSD
        peaks, properties = find_peaks(valid_psd, prominence=np.max(valid_psd)*0.1)
        
        if len(peaks) == 0:
            return None, None
        
        # Select highest peak
        idx = peaks[np.argmax(valid_psd[peaks])]
        
        # Parabolic interpolation for sub-bin accuracy
        if 0 < idx < len(valid_psd) - 1:
            y1, y2, y3 = valid_psd[idx-1:idx+2]
            x0 = (y3 - y1) / (2 * (2*y2 - y1 - y3))
            freq_est = valid_freqs[idx] + x0 * (valid_freqs[1] - valid_freqs[0])
        else:
            freq_est = valid_freqs[idx]
        
        # Better SNR estimation
        signal_power = valid_psd[idx]
        noise_mask = np.ones(len(valid_psd), dtype=bool)
        noise_mask[max(0, idx-3):min(len(valid_psd), idx+4)] = False
        noise_power = np.median(valid_psd[noise_mask]) if np.any(noise_mask) else 1e-6
        
        snr = 10 * np.log10(signal_power / noise_power)
        sigma = 1.0 / (snr + 1e-6)
        sigma = np.clip(sigma, 0.1, 2.0)
  
        
        return freq_est * 60, sigma

    def combined_breath_rate(self):
        # Check how much data we have
        data_duration = len(self.buffer) / self.fs
        
        # Use different strategies based on available data
        if data_duration < 5:
            # Not enough data
            return None, None
        elif data_duration < 10:
            # Early stage - rely more on peak detection
            rate_peak, sigma_peak = self.peak_breath_rate()
            return rate_peak, sigma_peak if rate_peak else (None, None)
        else:
            # Enough data - use both methods
            rate_fft, sigma_fft = self.fft_breath_rate()
            rate_peak, sigma_peak = self.peak_breath_rate()
            
            if rate_fft is None and rate_peak is None:
                return None, None
            if rate_fft is None:
                return rate_peak, sigma_peak
            if rate_peak is None:
                return rate_fft, sigma_fft
            
            # Check for agreement
            if abs(rate_fft - rate_peak) > 5:  # Disagreement > 5 BPM
                # Trust the one with lower uncertainty
                if sigma_fft < sigma_peak:
                    return rate_fft, sigma_fft
                else:
                    return rate_peak, sigma_peak
            
            # Normal weighted average
            sigma_fft = max(0.1, sigma_fft)
            sigma_peak = max(0.1, sigma_peak)
            
            w_fft = 1.0 / (sigma_fft ** 2)
            w_peak = 1.0 / (sigma_peak ** 2)
            rate = (w_fft * rate_fft + w_peak * rate_peak) / (w_fft + w_peak)
            sigma = 1.0 / np.sqrt(w_fft + w_peak)

            return rate, max(0.1, sigma)

    def show_DSP_pipeline(self, save_path=None):
        '''Graphs for report showing each step of DSP pipeline'''
        plt.figure(figsize=(12, 10))
    
        # Create time arrays for each buffer (they may have different lengths)
        time_raw = np.arange(len(self.raw_buffer)) / self.fs
        time_filtered = np.arange(len(self.filtered_buffer)) / self.fs
        time_no_drift = np.arange(len(self.no_drift_buffer)) / self.fs
        time_smoothed = np.arange(len(self.buffer)) / self.fs
        
        # Plot 1: Raw Signal
        plt.subplot(4, 1, 1)
        plt.plot(time_raw, self.raw_buffer, label='Raw Signal', color='blue')
        plt.title('Raw Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Low-pass Filtered
        plt.subplot(4, 1, 2)
        plt.plot(time_filtered, self.filtered_buffer, label='Low-pass Filtered', color='orange')
        plt.title('Low-pass Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Drift Removed
        plt.subplot(4, 1, 3)
        plt.plot(time_no_drift, self.no_drift_buffer, label='Drift Removed', color='green')
        plt.title('Drift Removed Signal (High-pass Filtered)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Final Smoothed
        plt.subplot(4, 1, 4)
        plt.plot(time_smoothed, self.buffer, label='Smoothed Signal', color='red')
        plt.title('Final Smoothed Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DSP pipeline visualization saved to {save_path}")
        
        # Close the figure to free memory
        plt.close()


class BayesFusion:
    def __init__(self, hypothesis_range=(6, 30, 0.1)):
        self.hypotheses = np.arange(*hypothesis_range)
        self.n_hyp = len(self.hypotheses)
        self.fusions = []
        # Informative prior based on typical breathing rates
        mean_rate = 10
        std_rate = 3
        self.prior = np.exp(-0.5 * ((self.hypotheses - mean_rate) / std_rate) ** 2)
        self.prior /= np.sum(self.prior)
        
        # Kalman filter for temporal tracking
        self.kf_state = None
        self.kf_covariance = 1.0
        self.process_noise = 0.1
        
        # Sensor reliability tracking
        self.sensor1_reliability = 1.0
        self.sensor2_reliability = 1.0
        self.error_history = {'sensor1': deque(maxlen=5), 
                              'sensor2': deque(maxlen=5)}

    def likelihood(self, measured_rate, hypothesis_rates, sigma):
        """Calculate likelihood of hypotheses given a measurement"""
        # Prevent division by zero
        sigma = max(sigma, 0.1)  # Minimum sigma to prevent numerical issues
        
        likelihoods = np.exp(-0.5 * ((measured_rate - hypothesis_rates) / sigma) ** 2)
        likelihoods = np.clip(likelihoods, 1e-10, 1.0)
        return likelihoods
    
    def update_reliability(self, rate1, rate2, fused_estimate):
        """
        Update sensor reliability scores based on consistency with fused estimate
        and historical performance
        """
        # Check if fused_estimate is None
        if fused_estimate is None:
            return
        
        # Calculate errors if measurements exist
        if rate1 is not None:
            error1 = abs(rate1 - fused_estimate)
            self.error_history['sensor1'].append(error1)
        
        if rate2 is not None:
            error2 = abs(rate2 - fused_estimate)
            self.error_history['sensor2'].append(error2)
        
        # Update reliability scores based on recent error history
        if len(self.error_history['sensor1']) >= 5:
            # Use inverse of mean absolute error as reliability
            mean_error1 = np.mean(self.error_history['sensor1'])
            self.sensor1_reliability = 1.0 / (1.0 + mean_error1)
        
        if len(self.error_history['sensor2']) >= 5:
            mean_error2 = np.mean(self.error_history['sensor2'])
            self.sensor2_reliability = 1.0 / (1.0 + mean_error2)
        
        # Ensure reliability stays within reasonable bounds
        self.sensor1_reliability = np.clip(self.sensor1_reliability, 0.1, 1.0)
        self.sensor2_reliability = np.clip(self.sensor2_reliability, 0.1, 1.0)
    
    def predict_from_kalman(self):
        """Return prediction from Kalman filter when no measurements available"""
        if self.kf_state is not None:
            return self.kf_state
        return None

    def fuse_estimates(self, rate1, rate2, sigma1, sigma2):
        # Handle missing measurements
        if rate1 is None and rate2 is None:
            prediction = self.predict_from_kalman() if self.kf_state else None
            return prediction
        
        # Outlier detection
        if self.kf_state is not None:
            if rate1 is not None and abs(rate1 - self.kf_state) > 3 * np.sqrt(self.kf_covariance):
                rate1 = None
            if rate2 is not None and abs(rate2 - self.kf_state) > 3 * np.sqrt(self.kf_covariance):
                rate2 = None
        
        # Single sensor fallback
        if rate1 is None and rate2 is None:
            return self.predict_from_kalman() if self.kf_state else None
        if rate1 is None:
            result = self.kalman_update(rate2, sigma2**2)
            return result
        if rate2 is None:
            result = self.kalman_update(rate1, sigma1**2)
            return result
        
        # Fix: Use correct reliability for each sensor
        sigma1_adj = np.clip(sigma1 / self.sensor1_reliability, 0.1, 5.0)
        sigma2_adj = np.clip(sigma2 / self.sensor2_reliability, 0.1, 5.0)
        
        # Bayesian fusion
        L_sensor1 = self.likelihood(rate1, self.hypotheses, sigma1_adj)
        L_sensor2 = self.likelihood(rate2, self.hypotheses, sigma2_adj)
        
        # Use static prior or reduce temporal prior influence
        if self.kf_state is not None:
            temporal_prior = np.exp(-0.5 * ((self.hypotheses - self.kf_state) / 
                                np.sqrt(self.kf_covariance + self.process_noise)) ** 2)
            temporal_prior /= np.sum(temporal_prior)
            alpha = 0.3  # Reduced weight on temporal prior
            combined_prior = (self.prior ** (1 - alpha)) * (temporal_prior ** alpha)
            combined_prior /= np.sum(combined_prior)
        else:
            combined_prior = self.prior
        
        posterior = L_sensor1 * L_sensor2 * combined_prior
        posterior /= np.sum(posterior)
        
        # Use the full posterior distribution, not just MAP
        # Weighted mean gives a smoother estimate
        fused_estimate = np.sum(self.hypotheses * posterior)
        
        # Estimate uncertainty from posterior spread
        posterior_variance = np.sum((self.hypotheses - fused_estimate)**2 * posterior)
        
        # Update Kalman with the Bayesian estimate and its uncertainty
        self.kalman_update(fused_estimate, posterior_variance)
        
        # Update reliability based on consistency between sensors
        sensor_agreement = abs(rate1 - rate2)
        if sensor_agreement < 2.0:  # Good agreement
            self.sensor1_reliability = min(1.0, self.sensor1_reliability * 1.05)
            self.sensor2_reliability = min(1.0, self.sensor2_reliability * 1.05)
        else:  # Poor agreement - reduce reliability of the outlier
            if abs(rate1 - fused_estimate) > abs(rate2 - fused_estimate):
                self.sensor1_reliability *= 0.95
            else:
                self.sensor2_reliability *= 0.95
        self.fusions.append(fused_estimate)
        return fused_estimate
    
    def kalman_update(self, measurement, measurement_variance):
        measurement_variance = max(measurement_variance, 0.01)
        if self.kf_state is None:
            self.kf_state = measurement
            self.kf_covariance = measurement_variance
            return measurement
        
        # Prediction step
        predicted_state = self.kf_state
        predicted_covariance = self.kf_covariance + self.process_noise
        
        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance)
        self.kf_state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.kf_covariance = (1 - kalman_gain) * predicted_covariance
        
        return self.kf_state

# Create a SerialConnection instance
serial_conn = SerialConnection(baudrate=115200)

if not serial_conn.select_port():
    print("Failed to establish connection. Exiting.")
    exit()

serialInst = serial_conn.serialInst

def clear():
    '''
    Clears the terminal screen and scroll back to present
    the user with a nice clean, new screen. Useful for managing
    menu screens in terminal applications.
    '''
    os.system('cls||echo -e \\\\033c')

class BreathRateSession:
    def __init__(self, sensor1, sensor2, fusion, serialInst):
        self.sensor1 = sensor1
        self.sensor2 = sensor2
        self.fusion = fusion
        self.serialInst = serialInst

    def start_sensor(self, no_breaths: int):   
        while no_breaths != 0:
            signal_go = "BG"
            self.serialInst.write(signal_go.encode('utf-8'))
            self.serialInst.flush()

            for j in range(6):
                clear()
                bar = 'LOADING BREATH RATE SENSOR GET READY USER' + '-'*j
                print(bar)
                time.sleep(1)

            time_interval = 0
            for i in range(4,0, -1):
                clear()
                print(">>> Breathe in <<<")
                print(f">>>    {i}    <<<")
                start_time = time.time()
                while self.serialInst.in_waiting and time_interval <= 1:
                    sensor_data = self.serialInst.readline().decode('utf-8').strip()
                    timestamp, value1, value2 = sensor_data.split(",")
                    timestamp = float(timestamp)
                    smoothed1 = self.sensor1.update(float(value1), timestamp)
                    smoothed2 = self.sensor2.update(float(value2), timestamp)
                    rate1, sigma1 = self.sensor1.combined_breath_rate()
                    rate2, sigma2 = self.sensor2.combined_breath_rate()
                    if rate1 and rate2:
                        fused_rate = self.fusion.fuse_estimates(rate1, rate2, sigma1, sigma2)
                        if fused_rate is not None:
                            print(f"Fused Breath Rate: {fused_rate:.2f} bpm")
                    end_time = time.time()
                    time_interval = end_time - start_time
                time_interval = 0

            for k in range(6, 0, -1):
                clear()
                print("<<< Exhale >>>")
                print(f">>>  {k}  <<<")
                start_time = time.time()
                while self.serialInst.in_waiting and time_interval <= 1:
                    sensor_data = self.serialInst.readline().decode('utf-8').strip()
                    timestamp, value1, value2 = sensor_data.split(",")
                    timestamp = float(timestamp)
                    smoothed1 = self.sensor1.update(float(value1), timestamp)
                    smoothed2 = self.sensor2.update(float(value2), timestamp)
                    rate1, sigma1 = self.sensor1.combined_breath_rate()
                    rate2, sigma2 = self.sensor2.combined_breath_rate()
                    if rate1 and rate2:
                        fused_rate = self.fusion.fuse_estimates(rate1, rate2, sigma1, sigma2)
                        if fused_rate is not None:
                            print(f"Fused Breath Rate: {fused_rate:.2f} bpm")
                    end_time = time.time()
                    time_interval = end_time - start_time
                time_interval = 0

            no_breaths -= 1

        clear()
        print("Breath recorded, now returning.....")
        time.sleep(1)
        self.activation_menu()

    def time_selection(self):
        clear()
        print(">>>>>>> List of Commands <<<<<<<")
        print("Input time in seconds for how many breaths to record")
        print("----------------------------------")
        record_value = int(input("Input: "))
        self.start_sensor(record_value)

    def export_data(self):
        clear()
        print(">>>>>>> List of Commands <<<<<<<")
        command = input("Export data (Y/N): ")
        if command.lower() == "y":
            clear()
            print("Writing CSV file......")
            with open(f"breathrate_data_{datetime.now()}.csv", mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Sensor1", "Sensor2", "BreathRate1", "BreathRate2", "FusedBreathRate"])
                for i in range(len(self.sensor1.timestamp_buffer)):
                    timestamp = self.sensor1.timestamp_buffer[i]
                    value1 = self.sensor1.raw_buffer[i] 
                    value2 = self.sensor2.raw_buffer[i] 
                    rate1 = self.sensor1.buffer[i]
                    rate2 = self.sensor2.buffer[i]
                    fused_rate = self.fusion.fusions[i]
                    writer.writerow([timestamp, value1, value2, rate1, rate2, fused_rate])
            clear()
            print("CSV file done.")
            time.sleep(2)
            self.activation_menu()
        elif command.lower() == "n":
            self.activation_menu()

    def activation_menu(self):
        while True:
            clear()
            print("------- BREATH RATE SENSOR ------")
            print(">>>>>>> List of Commands <<<<<<<")
            print("1. Information")
            print("2. Begin")
            print("3. Transcript")
            print("4. Exit")
            print("----------------------------------")
            command = input("Input: ")
            if command.lower() in ("begin", "b", "2"):
                self.time_selection()
            elif command.lower() in ("transcript", "t", "3"):
                self.export_data()
            elif command.lower() == "exit":
                print("exit has been issued... ")
                print("---- TERMINATING ----")
                self.serialInst.close()
                break

# Main loop for reading data
def main():
    # Buffer length (seconds) = buffer_size / sampling_rate
    # Put whatever sampeling rate the STM32 is using
    sensor1_estimator = BreathRateEstimator(buffer_size=500, sampling_rate=10)
    sensor2_estimator = BreathRateEstimator(buffer_size=500, sampling_rate=10)

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
    print("Waiting for STM32 to be ARMED...")
    while True:
        if serialInst.in_waiting:
            line = serialInst.readline().decode('utf-8').strip()
            if line == "ARMED":
                print("STM32 started. Beginning data collection.")
                break
    print("Press ESC to stop the program.")
    session = BreathRateSession(sensor1_estimator, sensor2_estimator, fusion, serialInst)

    session.activation_menu()

    # os.makedirs("data", exist_ok=True)
    # sensor_data_filename = f"./data/sensor_data_{current_time}.csv"
    # with open(sensor_data_filename, "w", buffering=1) as f:
    #     f.write("Timestamp,Sensor1,Sensor2,BreathRate1,BreathRate2,FusedBreathRate\n")

    #     try:
    #         while running:
    #             line = serialInst.readline().decode('utf-8').strip()
    #             if line:
    #                 try:
    #                     # TODO: Check if thermistor and band need same DSP pipeline
    #                     timestamp, value1, value2 = line.split(",")
    #                     timestamp = float(timestamp)
    #                     # For debugging
    #                     # DPS pipeline

    #                     smoothed1 = sensor1_estimator.update(float(value1), timestamp)
    #                     smoothed2 = sensor2_estimator.update(float(value2), timestamp)

    #                     rate1, sigma1 = sensor1_estimator.combined_breath_rate()
    #                     rate2, sigma2 = sensor2_estimator.combined_breath_rate()

    #                     if rate1 and rate2:
    #                         fused_rate = fusion.fuse_estimates(rate1, rate2, sigma1, sigma2)
    #                         if fused_rate is not None:
    #                             print(f"Fused Breath Rate: {fused_rate:.2f} bpm")
    #                             f.write(f"{timestamp},{value1},{value2},{rate1},{rate2},{fused_rate}\n")

    #                 except KeyboardInterrupt:
    #                     print("Exiting...")
    #                     running = False
    #                 except ValueError:
    #                     if line == "DISARMED":
    #                         print("STM32 stopped. Exiting...")
    #                         running = False
    #                     else:
    #                         print(f"tf is this sh: {line}")
    #                     continue
    #     except Exception as e:
    #         print(f"Error: {e}")

    serial_conn.disconnect()
            
    # print(f"Sensor data saved to {sensor_data_filename}")

    os.makedirs("data/figs", exist_ok=True)
    sensor1_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor1_{current_time}.png")
    sensor2_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor2_{current_time}.png")
    print("DSP pipeline visualizations saved.")

if __name__ == "__main__":
    main()



