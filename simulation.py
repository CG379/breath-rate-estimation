import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, detrend
import os
import threading
import queue

# Keep all your existing filter and class definitions as they are
def butter_lowpass_coeffs(cutoff, fs, order=2):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    return butter(order, norm_cutoff, btype='low')

def apply_lowpass(new_sample, zf, b, a):
    """Apply single-step IIR low-pass filter with filter memory (state zf)."""
    output, zf = lfilter(b, a, [new_sample], zi=zf)
    return output[0], zf

def butter_highpass_coeffs(cutoff, fs, order=2):
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
        window = list(self.filtered_buffer)[-self.window_size*10:]
        if len(window) > 0:
            mean_val = np.median(window)
        else:
            mean_val = 0
        detrended = filtered - mean_val
        self.no_drift_buffer.append(detrended)

        # Stage 3: Rolling average smoothing (on drift-removed signal)
        window = list(self.no_drift_buffer)[-self.window_size:]
        smoothed = np.mean(window)
        self.buffer.append(smoothed)

        # Timestamps for all stages (assumed same for alignment)
        self.timestamp_buffer.append(float(timestamp))

        return smoothed

    
    def fft_breath_rate(self):
        # TODO: Check if full DSP pipline is needed before FFT
        # Drift removal definately needed for FFT, others?
        if len(self.buffer) < self.fs * 10:
            return None, None  # Not enough data
        signal = np.array(self.buffer)
        # Experiment with hamming or han window
        #wavelet = signal * np.hamming(len(signal))
        wavelet = signal * np.hanning(len(signal))
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
        # TODO: check if this is enough for no breathing
        if len(valid_magnitudes) == 0:
            return None, None

        # Find dominant frequency
        dominant_idx = np.argmax(valid_magnitudes)
        dominant_freq = valid_freqs[dominant_idx]

                
        peak_magnitude = fft_magnitude[dominant_idx]
        background = np.mean(fft_magnitude)  # or use sideband avg
        snr = peak_magnitude / (background + 1e-6)

        sigma = 1.0 / (snr + 1e-6)
        rate = dominant_freq * 60  # Convert to BPM


        # Convert to BPM
        return rate, sigma
    
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
        # Prior = expected distribution of breath rates
        self.prior = np.ones(self.n_hyp) / self.n_hyp 
        # TODO: Find correct sigmas
        # Higher sigma = less reliable
        # Lower sigma = more reliable
        self.sigma_temp = 6
        self.sigma_stretch = 3
        # Dynamic sigma based on measurement confidence
        self.base_sigma = 1.5
        
        # History for adaptive filtering
        self.estimate_history = deque(maxlen=5)

        # Add temporal smoothing
        self.previous_estimate = None
        self.smoothing_factor = 0.7  # 0-1, higher = more smoothing

        # For visualization
        self.latest_fusion = None
        self.latest_rates = [None, None]

    def likelihood(self, measured_rate, hypothesis_rates, sigma):
        """Calculate likelihood of hypotheses given a measurement"""
        likelihoods = np.exp(-0.5 * ((measured_rate - hypothesis_rates) / sigma) ** 2)
        likelihoods = np.clip(likelihoods, 1e-10, 1.0)
        return likelihoods
    
    def fuse_estimates(self, rate1, rate2, sigma_temp, sigma_stretch):
        self.latest_rates = [rate1, rate2]

        # If either rate is None, return the other one
        if rate1 is None and rate2 is None:
            return None
        elif rate1 is None:
            return rate2
        elif rate2 is None:
            return rate1
        
        # Calculate likelihoods
        L_sensor1 = self.likelihood(rate1, self.hypotheses, sigma_temp)
        L_sensor2 = self.likelihood(rate2, self.hypotheses, sigma_stretch)

        # Posterior is probability distribution of each possible breath rate is, 
        # given the evidence from both sensors
        # Combine evidence using Bayes' rule
        unnormalized_posterior = L_sensor1 * L_sensor2 * self.prior
        # Normalize the posterior
        posterior_sum = np.sum(unnormalized_posterior)
        if posterior_sum < 1e-6 or np.isnan(posterior_sum):
            return (rate1 + rate2) / 2
        # Normalize the posterior
        posterior = unnormalized_posterior / posterior_sum
        

        map_estimate = self.hypotheses[np.argmax(posterior)]
        # Temporal smoothing
        if self.previous_estimate is not None:
            map_estimate = (self.smoothing_factor * self.previous_estimate + 
                        (1 - self.smoothing_factor) * map_estimate)
        
        self.previous_estimate = map_estimate
        return map_estimate



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
                rate1, sigma1 = sensor1_estimator.fft_breath_rate()
                rate2, sigma2 = sensor2_estimator.fft_breath_rate()

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
            
            rate1 = sensor1_estimator.fft_breath_rate()
            rate2 = sensor2_estimator.fft_breath_rate()
            
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
