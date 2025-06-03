import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, detrend, find_peaks, welch, savgol_filter
import os
import threading
import queue

# Keep all your existing filter and class definitions as they are
def butter_lowpass_coeffs(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='low')

def apply_lowpass(new_sample, zf, b, a):
    """Apply single-step IIR low-pass filter with filter memory (state zf)."""
    output, zf = lfilter(b, a, [new_sample], zi=zf)
    return output[0], zf

def butter_highpass_coeffs(cutoff, fs, order=4):
    """Design a highpass Butterworth filter."""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='high')

def apply_highpass(new_sample, zf, b, a):
    """Apply single-step IIR high-pass filter with filter memory (state zf)."""
    output, zf = lfilter(b, a, [new_sample], zi=zf)
    return output[0], zf

class BreathRateEstimator:
    def __init__(self, buffer_size=750, sampling_rate=10, window_size=3):
        self.fs = sampling_rate
        self.window_size = window_size
        # Low pass
        self.b, self.a = butter_lowpass_coeffs(1.5, sampling_rate)
        self.zf = np.zeros(max(len(self.a), len(self.b)) - 1)
        # High pass
        self.b_hp, self.a_hp = butter_highpass_coeffs(0.06, sampling_rate)
        self.zf_hp = np.zeros(max(len(self.a_hp), len(self.b_hp)) - 1)
        
        # Raw signal buffer for DSP comparison
        self.raw_buffer = deque(maxlen=buffer_size) 
        # Low pass filtered buffer
        self.filtered_buffer = deque(maxlen=buffer_size)
        # Removed drift buffer (rolling mean)
        self.no_drift_buffer = deque(maxlen=buffer_size)

        # Final buffer for peak detection and FFT
        # Smoothed signal buffer
        self.buffer = deque(maxlen=buffer_size) 

        self.timestamp_buffer = deque(maxlen=buffer_size)

        self.last_rate = None  # For adaptive peak detection

    def update(self, new_sample, timestamp):
        """
        Process a new input sample through:
        1. Low-pass filter
        2. High-pass filter (for drift removal)
        3. Rolling average (for smoothing)

        Each stage is buffered for visualization.
        """

        # Stage 0: Raw
        self.raw_buffer.append(new_sample)

        # Stage 1: Low-pass filter
        filtered, self.zf = apply_lowpass(new_sample, self.zf, self.b, self.a)
        self.filtered_buffer.append(filtered)

        # Stage 2: High-pass filter on low-passed signal
        # Remove drift by subtracting a rolling mean (removes slow drift)
        detrended, self.zf_hp = apply_highpass(filtered, self.zf_hp, self.b_hp, self.a_hp)
        self.no_drift_buffer.append(detrended)

        # Stage 3: Rolling average smoothing (on drift-removed signal)
        if len(self.buffer) > 0:
            alpha = 0.4  # Higher alpha = less smoothing, faster response
            smoothed = alpha * detrended + (1 - alpha) * self.buffer[-1]
        else:
            smoothed = detrended
        
        self.buffer.append(smoothed)

        # Timestamps for all stages (assumed same for alignment)
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
        signal = detrend(signal, type='linear')
        
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
        min_height = np.percentile(signal, 70)

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
            max_distance = int(self.fs * 60 / 6)   # 6 BPM min
        
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
        
        signal = detrend(np.array(self.buffer))
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        # Zero-pad for better frequency resolution
        n_fft = 2 ** int(np.ceil(np.log2(len(signal) * 4)))
        
        # Use Welch's method for more robust spectral estimation
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(len(signal)//4, 256), 
                        nfft=n_fft, detrend='constant')
        
        # Focus on breathing range
        mask = (freqs >= 0.1) & (freqs <= 0.5)  # 6-30 BPM
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
        
        # Informative prior based on typical breathing rates
        mean_rate = 10
        std_rate = 5
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
        
        # Outlier detection - make sure kf_state exists
        if self.kf_state is not None:
            if rate1 is not None and abs(rate1 - self.kf_state) > 3 * np.sqrt(self.kf_covariance):
                rate1 = None  # Reject outlier
            if rate2 is not None and abs(rate2 - self.kf_state) > 3 * np.sqrt(self.kf_covariance):
                rate2 = None
        
        # Check if both were rejected as outliers
        if rate1 is None and rate2 is None:
            return self.predict_from_kalman() if self.kf_state else None
        
        # Single sensor fallback
        if rate1 is None:
            result = self.kalman_update(rate2, sigma2**2)
            if result is not None:
                self.update_reliability(None, rate2, result)
            return result
        if rate2 is None:
            result = self.kalman_update(rate1, sigma1**2)
            if result is not None:
                self.update_reliability(rate1, None, result)
            return result
        
        # Adaptive sigma based on sensor reliability
        sigma1_adj = sigma1 / self.sensor1_reliability
        sigma2_adj = sigma2 / self.sensor2_reliability
        
        # Bayesian fusion
        L_sensor1 = self.likelihood(rate1, self.hypotheses, sigma1_adj)
        L_sensor2 = self.likelihood(rate2, self.hypotheses, sigma2_adj)
        
        # Include temporal prior from Kalman filter
        if self.kf_state is not None:
            temporal_prior = np.exp(-0.5 * ((self.hypotheses - self.kf_state) / 
                                   np.sqrt(self.kf_covariance)) ** 2)
            temporal_prior /= np.sum(temporal_prior)
            combined_prior = self.prior * temporal_prior
            combined_prior /= np.sum(combined_prior)
        else:
            combined_prior = self.prior
        
        posterior = L_sensor1 * L_sensor2 * combined_prior
        posterior /= np.sum(posterior)
        
        # MAP estimate
        map_estimate = self.hypotheses[np.argmax(posterior)]
        
        # Update Kalman filter
        fused_estimate = self.kalman_update(map_estimate, 
                                           1.0/np.sqrt(1/sigma1_adj**2 + 1/sigma2_adj**2))
        
        if fused_estimate is not None:
            self.update_reliability(rate1, rate2, fused_estimate)
        
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



# NEW: Simulated data generator class
class SimulatedSensorData:
    def __init__(self, sampling_rate=50, base_breath_rate=15):
        self.sampling_rate = sampling_rate
        self.base_breath_rate = base_breath_rate  # breaths per minute
        self.time_start = time.time()
        self.sample_count = 0
        
        # Sensor characteristics
        self.sensor1_amplitude = 1.0  # Thermistor
        self.sensor2_amplitude = 0.8  # Stretch band
        
        # Noise levels
        self.sensor1_noise = 0.15
        self.sensor2_noise = 0.1
        
        # Drift parameters
        self.sensor1_drift_rate = 0.002
        self.sensor2_drift_rate = 0.001
        
        # Variable breath rate parameters
        self.breath_rate_variation = 2.0  # +/- BPM variation
        self.breath_rate_change_period = 30  # seconds
        
    def generate_sample(self):
        """Generate a single sample of simulated sensor data"""
        current_time = time.time() - self.time_start
        timestamp = current_time
        
        # Vary breath rate slowly over time
        breath_rate_offset = self.breath_rate_variation * np.sin(2 * np.pi * current_time / self.breath_rate_change_period)
        current_breath_rate = self.base_breath_rate + breath_rate_offset
        # accracy testing later
        self.current_actual_breath_rate = current_breath_rate
        # Convert BPM to Hz
        breath_freq = current_breath_rate / 60.0
        
        # Generate base breathing signal
        breath_signal = np.sin(2 * np.pi * breath_freq * current_time)
        
        # Add harmonics for more realistic waveform
        breath_signal += 0.3 * np.sin(4 * np.pi * breath_freq * current_time)
        breath_signal += 0.1 * np.sin(6 * np.pi * breath_freq * current_time)
        
        # Sensor 1 (Thermistor) - typically has cleaner signal
        sensor1_drift = self.sensor1_drift_rate * current_time
        sensor1_noise = np.random.normal(0, self.sensor1_noise)
        sensor1_value = self.sensor1_amplitude * breath_signal + sensor1_drift + sensor1_noise
        
        # Sensor 2 (Stretch band) - typically has more noise and different response
        sensor2_drift = self.sensor2_drift_rate * current_time * np.sin(0.1 * current_time)
        sensor2_noise = np.random.normal(0, self.sensor2_noise)
        # Add phase shift and amplitude difference
        sensor2_signal = np.sin(2 * np.pi * breath_freq * current_time - np.pi/6)
        sensor2_value = self.sensor2_amplitude * sensor2_signal + sensor2_drift + sensor2_noise
        
        self.sample_count += 1
        
        return timestamp, sensor1_value, sensor2_value
    
    def get_data_stream(self):
        """Generator that yields data at the specified sampling rate"""
        sample_period = 1.0 / self.sampling_rate
        next_sample_time = time.time()
        
        while True:
            current_time = time.time()
            if current_time >= next_sample_time:
                yield self.generate_sample()
                next_sample_time += sample_period
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting


# Modified main function for simulation
def main_simulation():
    # Buffer length (seconds) = buffer_size / sampling_rate
    sensor1_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=10)
    sensor2_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=10)

    # For fusion, assume everything will fall in 6-30 BPM range
    fusion = BayesFusion(hypothesis_range=(6, 30, 0.1))

    # Create simulated data generator
    sim_data = SimulatedSensorData(sampling_rate=10, base_breath_rate=15)
    
    running = True
    
    # Save sensor data with timestamp in the filename
    current_time = time.strftime("%H-%M")
    
    print("Starting simulated data collection...")
    print("Press Ctrl+C to stop the program.")
    
    os.makedirs("data", exist_ok=True)
    sensor_data_filename = f"./data/simulated_sensor_data_{current_time}.csv"
    
    with open(sensor_data_filename, "w", buffering=1) as f:
        f.write("Timestamp,Sensor1,Sensor2,BreathRate1,BreathRate2,FusedBreathRate,ActualBreathRate\n")
        
        try:
            data_generator = sim_data.get_data_stream()
            sample_count = 0
            
            for timestamp, value1, value2 in data_generator:
                if not running:
                    break
                    
                sample_count += 1
                
                # DSP pipeline
                smoothed1 = sensor1_estimator.update(value1, timestamp)
                smoothed2 = sensor2_estimator.update(value2, timestamp)

                # Estimate breath rates
                rate1, sigma1 = sensor1_estimator.combined_breath_rate()
                rate2, sigma2 = sensor2_estimator.combined_breath_rate()

                if rate1 and rate2:
                    fused_rate = fusion.fuse_estimates(rate1, rate2, sigma1, sigma2)
                    if fused_rate is not None:
                        actual_rate = sim_data.current_actual_breath_rate
                        print(f"Sample {sample_count}: Actual: {actual_rate:.1f} bpm, Fused: {fused_rate:.1f} bpm, Error: {abs(actual_rate - fused_rate):.1f} bpm")
                        f.write(f"{timestamp:.3f},{value1:.4f},{value2:.4f},{rate1:.2f},{rate2:.2f},{fused_rate:.2f},{actual_rate:.2f}\n")
                                # Optional: Stop after a certain duration or number of samples
                        if sample_count >= 1000:  # Stop after 1000 samples (20 seconds at 50Hz)
                            print("Reached sample limit. Stopping...")
                            running = False
                    
        except KeyboardInterrupt:
            print("\nStopping data collection...")
            running = False
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Sensor data saved to {sensor_data_filename}")

    # Save DSP pipeline visualizations
    os.makedirs("data/figs", exist_ok=True)
    sensor1_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor1_sim_{current_time}.png")
    sensor2_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor2_sim_{current_time}.png")
    print("DSP pipeline visualizations saved.")


# Alternative: Load and replay existing data
def replay_recorded_data(data_file, realtime=True):
    """
    Replay recorded sensor data from a CSV file
    
    Args:
        data_file: Path to CSV file with columns: Timestamp,Sensor1,Sensor2,...
        realtime: If True, replay at original timing. If False, process as fast as possible.
    """
    import pandas as pd
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # Initialize estimators and fusion
    sensor1_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=50)
    sensor2_estimator = BreathRateEstimator(buffer_size=750, sampling_rate=50)
    fusion = BayesFusion(hypothesis_range=(6, 60, 0.1))
    
    # Save results
    current_time = time.strftime("%H-%M")
    output_filename = f"./data/replayed_results_{current_time}.csv"
    
    with open(output_filename, "w", buffering=1) as f:
        f.write("Timestamp,Sensor1,Sensor2,BreathRate1,BreathRate2,FusedBreathRate\n")
        
        start_time = time.time()
        first_timestamp = df['Timestamp'].iloc[0]
        
        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            value1 = row['Sensor1']
            value2 = row['Sensor2']
            
            # If realtime mode, wait to maintain original timing
            if realtime and idx > 0:
                elapsed = time.time() - start_time
                data_elapsed = timestamp - first_timestamp
                if data_elapsed > elapsed:
                    time.sleep(data_elapsed - elapsed)
            
            # Process the data
            smoothed1 = sensor1_estimator.update(value1, timestamp)
            smoothed2 = sensor2_estimator.update(value2, timestamp)
            
            rate1 = sensor1_estimator.combined_breath_rate()
            rate2 = sensor2_estimator.combined_breath_rate()
            
            if rate1 and rate2:
                fused_rate = fusion.fuse_estimates(rate1, rate2)
                if fused_rate is not None:
                    print(f"Timestamp {timestamp:.2f}: Fused rate: {fused_rate:.1f} bpm")
                    f.write(f"{timestamp:.3f},{value1:.4f},{value2:.4f},{rate1:.2f},{rate2:.2f},{fused_rate:.2f}\n")
    
    print(f"Replayed results saved to {output_filename}")
    
    # Save visualizations
    os.makedirs("data/figs", exist_ok=True)
    sensor1_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor1_replay_{current_time}.png")
    sensor2_estimator.show_DSP_pipeline(f"./data/figs/dsp_pipeline_sensor2_replay_{current_time}.png")


# Test data generator with various scenarios
class ScenarioSimulator:
    """Generate specific test scenarios for the breath rate sensor"""
    
    def __init__(self, sampling_rate=10):
        self.sampling_rate = sampling_rate
        self.scenarios = {
            'normal': self.normal_breathing,
            'exercise': self.exercise_breathing,
            'sleep': self.sleep_breathing,
            'irregular': self.irregular_breathing,
            'sensor_failure': self.sensor_failure,
            'high_noise': self.high_noise_scenario
        }
    
    def normal_breathing(self, duration=60):
        """Normal breathing at rest (12-18 BPM)"""
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=15)
        sim.breath_rate_variation = 1.5
        return self._run_scenario(sim, duration)
    
    def exercise_breathing(self, duration=60):
        """Elevated breathing during exercise (25-40 BPM)"""
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=32)
        sim.breath_rate_variation = 4.0
        sim.sensor2_noise = 0.25  # More noise from movement
        return self._run_scenario(sim, duration)
    
    def sleep_breathing(self, duration=60):
        """Slow breathing during sleep (8-12 BPM)"""
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=10)
        sim.breath_rate_variation = 1.0
        sim.sensor1_noise = 0.05  # Very low noise
        sim.sensor2_noise = 0.08
        return self._run_scenario(sim, duration)
    
    def irregular_breathing(self, duration=60):
        """Irregular breathing pattern"""
        data = []
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=15)
        
        for i, (timestamp, v1, v2) in enumerate(sim.get_data_stream()):
            # Add random pauses and rate changes
            if i % 200 == 0:  # Every 4 seconds at 50Hz
                sim.base_breath_rate = np.random.uniform(10, 25)
            
            data.append([timestamp, v1, v2])
            
            if len(data) >= duration * self.sampling_rate:
                break
                
        return np.array(data)
    
    def sensor_failure(self, duration=60):
        """Simulate sensor 2 failing halfway through"""
        data = []
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=15)
        
        for i, (timestamp, v1, v2) in enumerate(sim.get_data_stream()):
            # Sensor 2 fails after 30 seconds
            if timestamp > 30:
                v2 = np.random.normal(0, 0.01)  # Just noise
            
            data.append([timestamp, v1, v2])
            
            if len(data) >= duration * self.sampling_rate:
                break
                
        return np.array(data)
    
    def high_noise_scenario(self, duration=60):
        """High noise environment"""
        sim = SimulatedSensorData(sampling_rate=self.sampling_rate, base_breath_rate=15)
        sim.sensor1_noise = 0.3
        sim.sensor2_noise = 0.4
        return self._run_scenario(sim, duration)
    
    def _run_scenario(self, sim, duration):
        """Run a scenario for specified duration"""
        data = []
        for timestamp, v1, v2 in sim.get_data_stream():
            data.append([timestamp, v1, v2])
            if len(data) >= duration * self.sampling_rate:
                break
        return np.array(data)
    
    def save_scenario(self, scenario_name, duration=60):
        """Save a specific scenario to file"""
        if scenario_name not in self.scenarios:
            print(f"Unknown scenario: {scenario_name}")
            return
        
        print(f"Generating {scenario_name} scenario for {duration} seconds...")
        data = self.scenarios[scenario_name](duration)
        
        filename = f"./data/scenario_{scenario_name}_{time.strftime('%H-%M')}.csv"
        np.savetxt(filename, data, delimiter=',', 
                   header='Timestamp,Sensor1,Sensor2', comments='')
        print(f"Scenario saved to {filename}")
        return filename


# Visualization helper for real-time plotting
def plot_realtime_data(sensor1_estimator, sensor2_estimator, fusion, save_path=None):
    """Create a comprehensive plot of the current state"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Plot 1: Raw signals
    ax = axes[0, 0]
    if len(sensor1_estimator.raw_buffer) > 0:
        time_raw = np.arange(len(sensor1_estimator.raw_buffer)) / sensor1_estimator.fs
        ax.plot(time_raw, sensor1_estimator.raw_buffer, 'b-', label='Sensor 1')
        ax.plot(time_raw, sensor2_estimator.raw_buffer, 'r-', label='Sensor 2')
        ax.set_title('Raw Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
    
    # Plot 2: Processed signals
    ax = axes[0, 1]
    if len(sensor1_estimator.buffer) > 0:
        time_proc = np.arange(len(sensor1_estimator.buffer)) / sensor1_estimator.fs
        ax.plot(time_proc, sensor1_estimator.buffer, 'b-', label='Sensor 1')
        ax.plot(time_proc, sensor2_estimator.buffer, 'r-', label='Sensor 2')
        ax.set_title('Processed Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
    
    # Plot 3 & 4: FFT for each sensor
    for idx, (estimator, ax) in enumerate([(sensor1_estimator, axes[1, 0]), 
                                           (sensor2_estimator, axes[1, 1])]):
        if len(estimator.buffer) >= estimator.fs * 5:
            signal = np.array(estimator.buffer)
            wavelet = signal * np.hamming(len(signal))
            fft_result = np.fft.rfft(wavelet)
            fft_magnitude = np.abs(fft_result)
            freq = np.fft.rfftfreq(len(signal), d=1/estimator.fs)
            
            # Plot only breathing range
            mask = (freq >= 0.1) & (freq <= 1.1)
            ax.plot(freq[mask] * 60, fft_magnitude[mask])
            ax.set_title(f'Sensor {idx+1} FFT')
            ax.set_xlabel('Frequency (BPM)')
            ax.set_ylabel('Magnitude')
            ax.grid(True)
    
    # Plot 5: Bayesian fusion
    ax = axes[2, 0]
    if fusion.latest_fusion is not None:
        ax.plot(fusion.hypotheses, fusion.latest_fusion['L_sensor1'], 'b-', 
                label=f'Sensor 1 ({fusion.latest_rates[0]:.1f} BPM)' if fusion.latest_rates[0] else 'Sensor 1')
        ax.plot(fusion.hypotheses, fusion.latest_fusion['L_sensor2'], 'r-', 
                label=f'Sensor 2 ({fusion.latest_rates[1]:.1f} BPM)' if fusion.latest_rates[1] else 'Sensor 2')
        ax.set_title('Likelihood Functions')
        ax.set_xlabel('Breath Rate (BPM)')
        ax.set_ylabel('Likelihood')
        ax.legend()
        ax.grid(True)
    
    # Plot 6: Posterior distribution
    ax = axes[2, 1]
    if fusion.latest_fusion is not None:
        ax.plot(fusion.hypotheses, fusion.latest_fusion['posterior'], 'g-', linewidth=2)
        map_idx = np.argmax(fusion.latest_fusion['posterior'])
        map_estimate = fusion.hypotheses[map_idx]
        ax.axvline(map_estimate, color='k', linestyle='--', 
                   label=f'MAP: {map_estimate:.1f} BPM')
        ax.set_title('Posterior Distribution')
        ax.set_xlabel('Breath Rate (BPM)')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Choose which mode to run
    mode = "simulation"  # Options: "simulation", "replay", "scenarios"
    
    if mode == "simulation":
        # Run live simulation
        main_simulation()
        
    elif mode == "replay":
        # Replay existing data
        data_file = "./data/scenario_high_noise_11-07.csv"  # Replace with your file
        replay_recorded_data(data_file, realtime=True)
        
    elif mode == "scenarios":
        # Generate test scenarios
        scenario_gen = ScenarioSimulator()
        
        # Generate all scenarios
        for scenario in ['normal', 'exercise', 'sleep', 'irregular', 'sensor_failure', 'high_noise']:
            scenario_gen.save_scenario(scenario, duration=120)  # 2 minutes each
