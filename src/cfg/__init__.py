"""
Configuration Module
Handles loading and merging of configuration files
"""
import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class Settings:
    """
    Singleton settings class that loads and merges configuration files
    """
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Load and merge configuration files"""
        config_dir = os.path.dirname(__file__)
        
        # Load base configurations in order of precedence
        config_files = [
            'default.yaml',    # Lowest precedence
            'settings.yaml',   # Main configuration
            'user.yaml',       # User overrides
            'emulation.yaml'   # Emulation mode (if simulation)
        ]
        
        cls._config = {}
        
        for config_file in config_files:
            config_path = os.path.join(config_dir, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            cls._merge_config(cls._config, file_config)
                            logger.debug(f"Loaded config from {config_file}")
                except Exception as e:
                    logger.error(f"Error loading {config_file}: {e}")
            else:
                logger.warning(f"Config file not found: {config_file}")
        
        # Apply emulation overrides if in simulation mode
        if cls._config.get('SIMULATION', False):
            emulation_path = os.path.join(config_dir, 'emulation.yaml')
            if os.path.exists(emulation_path):
                try:
                    with open(emulation_path, 'r') as f:
                        emulation_config = yaml.safe_load(f)
                        if emulation_config:
                            cls._merge_config(cls._config, emulation_config)
                            logger.debug("Applied emulation configuration")
                except Exception as e:
                    logger.error(f"Error loading emulation config: {e}")
    
    @classmethod
    def _merge_config(cls, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'GENERAL.POLL_FREQUENCY')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def reload(self):
        """Reload configuration from files"""
        self._load_config()
    
    @property
    def SIMULATION(self) -> bool:
        """Quick access to simulation mode"""
        return self.get('SIMULATION', False)
    
    @property
    def GENERAL(self) -> Dict[str, Any]:
        """Quick access to general settings"""
        return self.get_section('GENERAL')
    
    @property
    def CURRENT_DEVICE(self) -> Dict[str, Any]:
        """Quick access to current device settings"""
        return self.get_section('CURRENT_DEVICE')
    
    @property
    def AVAILABLE_SENSORS(self) -> Dict[str, Any]:
        """Quick access to available sensors"""
        return self.get_section('AVAILABLE_SENSORS')
    
    @property
    def DRONE(self) -> Dict[str, Any]:
        """Quick access to drone configuration"""
        return self.get_section('DRONE')
    
    @property
    def ENVIRONMENT(self) -> Dict[str, Any]:
        """Quick access to environment settings"""
        return self.get_section('ENVIRONMENT')
    
    @property
    def HUB(self) -> Dict[str, Any]:
        """Quick access to hub settings"""
        return self.get_section('HUB')
    
    @property
    def CONTROL(self) -> Dict[str, Any]:
        """Quick access to control settings"""
        return self.get_section('CONTROL')
    
    @property
    def PHYSICS(self) -> Dict[str, Any]:
        """Quick access to physics settings"""
        return self.get_section('PHYSICS')
    
    @property
    def AI(self) -> Dict[str, Any]:
        """Quick access to AI settings"""
        return self.get_section('AI')
    
    @property
    def UI(self) -> Dict[str, Any]:
        """Quick access to UI settings"""
        return self.get_section('UI')

# Global settings instance
settings = Settings()

# Convenience imports
__all__ = ['settings', 'Settings'] 