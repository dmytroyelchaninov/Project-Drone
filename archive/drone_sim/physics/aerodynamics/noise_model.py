"""
Acoustic noise spectrum calculation for propellers
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import scipy.signal

class NoiseComponent(Enum):
    """Types of noise components"""
    THICKNESS = "thickness"      # Ffowcs Williams-Hawkings thickness noise
    LOADING = "loading"          # Loading noise
    BROADBAND = "broadband"      # Turbulence broadband noise
    TOTAL = "total"

@dataclass
class ObserverPosition:
    """Position of noise measurement point"""
    x: float  # meters from propeller center
    y: float  # meters from propeller center  
    z: float  # meters from propeller center (positive upward)
    
    def to_spherical(self) -> Tuple[float, float, float]:
        """Convert to spherical coordinates (r, theta, phi)"""
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z / r) if r > 0 else 0.0  # Polar angle from z-axis
        phi = np.arctan2(self.y, self.x)  # Azimuthal angle
        return r, theta, phi
        
    def distance(self) -> float:
        """Get distance from propeller"""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

@dataclass
class AcousticEnvironment:
    """Environmental parameters for acoustic modeling"""
    temperature: float = 288.15  # K (15°C)
    pressure: float = 101325.0   # Pa
    humidity: float = 0.5        # Relative humidity (0-1)
    
    def speed_of_sound(self) -> float:
        """Calculate speed of sound in current conditions"""
        # Simplified formula
        return 331.3 * np.sqrt(self.temperature / 273.15)
        
    def air_density(self) -> float:
        """Calculate air density"""
        R_specific = 287.0  # J/(kg⋅K)
        return self.pressure / (R_specific * self.temperature)

class PropellerNoiseModel:
    """
    Propeller noise model implementing various acoustic theories
    """
    
    def __init__(self, propeller_config: Dict[str, Any], environment: AcousticEnvironment = None):
        self.config = propeller_config
        self.environment = environment or AcousticEnvironment()
        
        # Propeller parameters
        self.diameter = propeller_config.get('diameter', 0.24)  # meters
        self.radius = self.diameter / 2
        self.blades = propeller_config.get('blades', 2)
        self.chord = propeller_config.get('chord', 0.02)  # meters
        
        # Operating conditions
        self.rpm = 0.0
        self.thrust = 0.0
        
        # Frequency analysis parameters
        self.sample_rate = 44100  # Hz
        self.freq_bins = 1024
        self.frequencies = np.fft.fftfreq(self.freq_bins, 1/self.sample_rate)[:self.freq_bins//2]
        
        # Noise spectrum storage
        self.noise_spectrum = {}
        
    def calculate_bpf_frequency(self, rpm: float) -> float:
        """
        Calculate blade passing frequency (BPF)
        
        Args:
            rpm: Rotations per minute
            
        Returns:
            BPF in Hz
        """
        return (rpm / 60.0) * self.blades
        
    def calculate_thickness_noise(self, rpm: float, observer: ObserverPosition) -> np.ndarray:
        """
        Calculate thickness noise using Ffowcs Williams-Hawkings theory
        
        Args:
            rpm: Rotations per minute
            observer: Observer position
            
        Returns:
            Sound pressure spectrum in Pa
        """
        self.rpm = rpm
        bpf = self.calculate_bpf_frequency(rpm)
        r, theta, phi = observer.to_spherical()
        
        if r == 0:
            return np.zeros(len(self.frequencies))
            
        # Basic thickness noise model
        # P = (ρ * c * M²) / (4π * r) * blade_loading_function
        
        rho = self.environment.air_density()
        c = self.environment.speed_of_sound()
        
        # Mach number at tip
        tip_speed = (2 * np.pi * rpm / 60) * self.radius
        mach_tip = tip_speed / c
        
        # Thickness noise amplitude (simplified)
        thickness_amplitude = (rho * c * mach_tip**2) / (4 * np.pi * r)
        
        # Generate harmonics of BPF
        spectrum = np.zeros(len(self.frequencies))
        
        for harmonic in range(1, 11):  # First 10 harmonics
            freq = harmonic * bpf
            if freq > 0 and freq < self.frequencies[-1]:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(self.frequencies - freq))
                
                # Amplitude decreases with harmonic number
                amplitude = thickness_amplitude / (harmonic**1.5)
                
                # Directivity pattern (simplified)
                directivity = 1.0 + 0.5 * np.cos(theta)
                
                spectrum[freq_idx] = amplitude * directivity
                
        return spectrum
        
    def calculate_loading_noise(self, rpm: float, thrust: float, observer: ObserverPosition) -> np.ndarray:
        """
        Calculate loading noise (far-field approximation)
        
        Args:
            rpm: Rotations per minute
            thrust: Thrust in Newtons
            observer: Observer position
            
        Returns:
            Sound pressure spectrum in Pa
        """
        self.rpm = rpm
        self.thrust = thrust
        bpf = self.calculate_bpf_frequency(rpm)
        r, theta, phi = observer.to_spherical()
        
        if r == 0 or thrust == 0:
            return np.zeros(len(self.frequencies))
            
        # Loading noise model
        # P = (thrust_fluctuation * sin(theta)) / (4π * r * c)
        
        c = self.environment.speed_of_sound()
        
        # Thrust fluctuation amplitude (simplified)
        thrust_fluctuation = thrust * 0.1  # 10% fluctuation
        
        # Loading noise amplitude
        loading_amplitude = (thrust_fluctuation * np.sin(theta)) / (4 * np.pi * r * c)
        
        spectrum = np.zeros(len(self.frequencies))
        
        for harmonic in range(1, 6):  # First 5 harmonics
            freq = harmonic * bpf
            if freq > 0 and freq < self.frequencies[-1]:
                freq_idx = np.argmin(np.abs(self.frequencies - freq))
                
                # Amplitude decreases with harmonic
                amplitude = loading_amplitude / harmonic
                
                spectrum[freq_idx] = amplitude
                
        return spectrum
        
    def calculate_broadband_noise(self, rpm: float, thrust: float, observer: ObserverPosition) -> np.ndarray:
        """
        Calculate broadband turbulence noise
        
        Args:
            rpm: Rotations per minute
            thrust: Thrust in Newtons
            observer: Observer position
            
        Returns:
            Sound pressure spectrum in Pa
        """
        r, theta, phi = observer.to_spherical()
        
        if r == 0:
            return np.zeros(len(self.frequencies))
            
        # Broadband noise model (simplified)
        # Based on turbulent mixing and boundary layer noise
        
        c = self.environment.speed_of_sound()
        tip_speed = (2 * np.pi * rpm / 60) * self.radius
        
        # Reynolds number based scaling
        reynolds = tip_speed * self.chord / (1.5e-5)  # Approximate kinematic viscosity
        
        # Broadband level
        broadband_level = 1e-6 * (tip_speed**3) / (r**2)  # Basic scaling
        
        # Frequency shaping (pink noise-like spectrum)
        spectrum = np.zeros(len(self.frequencies))
        
        for i, freq in enumerate(self.frequencies):
            if freq > 0:
                # 1/f noise with high-frequency rolloff
                amplitude = broadband_level / np.sqrt(freq)
                
                # High frequency rolloff
                if freq > 1000:
                    amplitude *= np.exp(-(freq - 1000) / 5000)
                    
                spectrum[i] = amplitude
                
        return spectrum
        
    def calculate_total_noise(self, rpm: float, thrust: float, 
                            observer: ObserverPosition, 
                            components: List[NoiseComponent] = None) -> Dict[str, np.ndarray]:
        """
        Calculate total noise spectrum from all components
        
        Args:
            rpm: Rotations per minute
            thrust: Thrust in Newtons
            observer: Observer position
            components: List of noise components to include
            
        Returns:
            Dictionary with noise spectra for each component
        """
        if components is None:
            components = [NoiseComponent.THICKNESS, NoiseComponent.LOADING, NoiseComponent.BROADBAND]
            
        results = {}
        
        if NoiseComponent.THICKNESS in components:
            results['thickness'] = self.calculate_thickness_noise(rpm, observer)
            
        if NoiseComponent.LOADING in components:
            results['loading'] = self.calculate_loading_noise(rpm, thrust, observer)
            
        if NoiseComponent.BROADBAND in components:
            results['broadband'] = self.calculate_broadband_noise(rpm, thrust, observer)
            
        # Calculate total (RMS sum)
        total_spectrum = np.zeros(len(self.frequencies))
        for component_spectrum in results.values():
            total_spectrum += component_spectrum**2
        results['total'] = np.sqrt(total_spectrum)
        
        # Store for later use
        self.noise_spectrum = results
        
        return results
        
    def calculate_spl(self, pressure_spectrum: np.ndarray, reference_pressure: float = 2e-5) -> np.ndarray:
        """
        Calculate Sound Pressure Level (SPL) in dB
        
        Args:
            pressure_spectrum: Pressure spectrum in Pa
            reference_pressure: Reference pressure (20 μPa for air)
            
        Returns:
            SPL spectrum in dB
        """
        # Avoid log of zero
        pressure_spectrum = np.maximum(pressure_spectrum, reference_pressure * 1e-6)
        return 20 * np.log10(pressure_spectrum / reference_pressure)
        
    def calculate_oaspl(self, pressure_spectrum: np.ndarray, reference_pressure: float = 2e-5) -> float:
        """
        Calculate Overall Sound Pressure Level (OASPL)
        
        Args:
            pressure_spectrum: Pressure spectrum in Pa
            reference_pressure: Reference pressure
            
        Returns:
            OASPL in dB
        """
        total_pressure_squared = np.sum(pressure_spectrum**2)
        total_pressure = np.sqrt(total_pressure_squared)
        return 20 * np.log10(total_pressure / reference_pressure)
        
    def apply_a_weighting(self, spl_spectrum: np.ndarray) -> np.ndarray:
        """
        Apply A-weighting to SPL spectrum
        
        Args:
            spl_spectrum: SPL spectrum in dB
            
        Returns:
            A-weighted SPL spectrum in dBA
        """
        # A-weighting filter coefficients (simplified)
        a_weighting = np.zeros(len(self.frequencies))
        
        for i, freq in enumerate(self.frequencies):
            if freq > 0:
                # Simplified A-weighting formula
                f2 = freq**2
                a_weight = (12194**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
                a_weighting[i] = 20 * np.log10(a_weight)
                
        return spl_spectrum + a_weighting
        
    def calculate_octave_bands(self, pressure_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Calculate octave band levels
        
        Args:
            pressure_spectrum: Pressure spectrum in Pa
            
        Returns:
            Dictionary with octave band levels in dB
        """
        # Standard octave band center frequencies
        octave_centers = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        octave_bands = {}
        
        for center_freq in octave_centers:
            if center_freq > self.frequencies[-1]:
                break
                
            # Octave band limits
            f_low = center_freq / np.sqrt(2)
            f_high = center_freq * np.sqrt(2)
            
            # Find frequency indices
            indices = np.where((self.frequencies >= f_low) & (self.frequencies <= f_high))[0]
            
            if len(indices) > 0:
                band_energy = np.sum(pressure_spectrum[indices]**2)
                band_level = 20 * np.log10(np.sqrt(band_energy) / 2e-5)
                octave_bands[f"{int(center_freq)} Hz"] = band_level
            else:
                octave_bands[f"{int(center_freq)} Hz"] = -np.inf
                
        return octave_bands
        
    def apply_atmospheric_absorption(self, pressure_spectrum: np.ndarray, distance: float) -> np.ndarray:
        """
        Apply atmospheric absorption
        
        Args:
            pressure_spectrum: Pressure spectrum in Pa
            distance: Propagation distance in meters
            
        Returns:
            Attenuated pressure spectrum
        """
        absorption_coeffs = np.zeros(len(self.frequencies))
        
        for i, freq in enumerate(self.frequencies):
            if freq > 0:
                # Simplified atmospheric absorption (dB/km)
                alpha = 0.1 * (freq / 1000)**1.5  # Rough approximation
                attenuation_db = alpha * distance / 1000  # Convert to dB
                attenuation_linear = 10**(-attenuation_db / 20)
                absorption_coeffs[i] = attenuation_linear
            else:
                absorption_coeffs[i] = 1.0
                
        return pressure_spectrum * absorption_coeffs
        
    def apply_doppler_effect(self, pressure_spectrum: np.ndarray, 
                           source_velocity: np.ndarray, 
                           observer_velocity: np.ndarray,
                           observer_position: ObserverPosition) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Doppler effect to spectrum
        
        Args:
            pressure_spectrum: Original pressure spectrum
            source_velocity: Source velocity vector [vx, vy, vz]
            observer_velocity: Observer velocity vector [vx, vy, vz]
            observer_position: Observer position
            
        Returns:
            Tuple of (doppler_shifted_spectrum, new_frequencies)
        """
        c = self.environment.speed_of_sound()
        
        # Direction from source to observer
        obs_vec = np.array([observer_position.x, observer_position.y, observer_position.z])
        distance = np.linalg.norm(obs_vec)
        
        if distance == 0:
            return pressure_spectrum, self.frequencies
            
        direction = obs_vec / distance
        
        # Radial velocities
        v_source_radial = np.dot(source_velocity, direction)
        v_observer_radial = np.dot(observer_velocity, direction)
        
        # Doppler factor
        doppler_factor = (c + v_observer_radial) / (c + v_source_radial)
        
        # Shift frequencies
        new_frequencies = self.frequencies * doppler_factor
        
        # Interpolate spectrum to new frequency grid
        shifted_spectrum = np.interp(self.frequencies, new_frequencies, pressure_spectrum)
        
        return shifted_spectrum, new_frequencies
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Get noise metrics from last calculation
        
        Returns:
            Dictionary with various noise metrics
        """
        if not self.noise_spectrum:
            return {}
            
        total_spectrum = self.noise_spectrum.get('total', np.zeros(len(self.frequencies)))
        
        # Calculate various metrics
        oaspl = self.calculate_oaspl(total_spectrum)
        spl_spectrum = self.calculate_spl(total_spectrum)
        a_weighted_spl = self.apply_a_weighting(spl_spectrum)
        a_weighted_oaspl = self.calculate_oaspl(10**(a_weighted_spl/20) * 2e-5)
        
        octave_bands = self.calculate_octave_bands(total_spectrum)
        
        return {
            'oaspl_db': oaspl,
            'oaspl_dba': a_weighted_oaspl,
            'peak_frequency': self.frequencies[np.argmax(total_spectrum)],
            'peak_spl': np.max(spl_spectrum),
            'octave_bands': octave_bands
        } 