import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
from jsonschema import validate

@dataclass
class DataConfig:
    ticker: str
    start_date: str
    end_date: str
    universe: Optional[str] = "sp500_utilities"

@dataclass
class PairsConfig:
    z_window: int = 60
    z_entry: float = 2.0
    z_exit: float = 0.5
    hedge_mode: str = "static_ols"
    coint_mode: str = "engle_granger"

@dataclass
class Config:
    data: DataConfig
    pairs: PairsConfig = field(default_factory=PairsConfig)

    @classmethod
    def load(cls, path: str) -> "Config":
        """
        @brief Loads configuration from a YAML file with schema validation.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at {path}")
            
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
            
        # Optional validation
        schema_path = "schema/config_schema.json"
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            validate(instance=raw_config, schema=schema)
            
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        @brief Creates a Config instance from a dictionary.
        """
        data_config = DataConfig(**data.get('data', {}))
        pairs_config = PairsConfig(**data.get('pairs', {}))
        return cls(data=data_config, pairs=pairs_config)

# Global config instance (optional, but convenient)
# config = Config.load("input/configuration.yaml")
