import yaml
from pathlib import Path
import hashlib

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        self.config_hash = self._calculate_config_hash()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")

    def _calculate_config_hash(self):
        with open(self.config_path, 'rb') as f:
            yaml_bytes = f.read()
            return hashlib.md5(yaml_bytes).hexdigest()[:10]  # 10-char hash


    @property
    def data(self) -> dict:
        """Provides access to the full config dictionary."""
        return self._config_data

    def get(self, *keys, default=None):
        """Safe getter for nested keys, e.g. get('create_ppnetwork', 'transformer_parameters', 'vn_hv_kv')"""
        val = self._config_data
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val
