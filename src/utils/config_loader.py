import yaml
import os
from pathlib import Path

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Locate config file relative to this script or project root
        base_path = Path(__file__).resolve().parent.parent.parent
        config_path = base_path / "config" / "hyperparameters.yaml"
        
        if not config_path.exists():
            # Fallback for different execution contexts
            config_path = Path("config/hyperparameters.yaml")
            
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            print(f"Warning: Config not found at {config_path}. Using defaults.")
            self._config = {}

    def get(self, key, default=None):
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

# Global accessor
config = ConfigLoader()
