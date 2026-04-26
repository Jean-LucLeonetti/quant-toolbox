import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os

@dataclass
class DataConfig:
    ticker: str
    start_date: str
    end_date: str

@dataclass
class Config:
    data: DataConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        """
        @brief Loads configuration from a YAML file.

        @param path The path to the YAML configuration file.
        @return A Config instance populated with data from the file.
        @throws FileNotFoundError If the configuration file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at {path}")
            
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
            
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        @brief Creates a Config instance from a dictionary.

        @param data A dictionary containing configuration settings.
        @return A Config instance populated with data from the dictionary.
        """
        data_config = DataConfig(**data.get('data', {}))
        return cls(data=data_config)

# Global config instance (optional, but convenient)
# config = Config.load("input/configuration.yaml")
