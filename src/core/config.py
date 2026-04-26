import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
from jsonschema import validate

@dataclass
class DataConfig:
    ticker: str
    start_date: str
    end_date: str
    universe: Optional[str] = "sp500_utilities"
    excluded_periods: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class PairsConfig:
    z_window: int = 60
    z_entry: float = 2.0
    z_exit: float = 0.5
    hedge_mode: str = "static_ols"
    coint_mode: str = "engle_granger"
    regime_filter: bool = False
    portfolio_size: int = 10
    wf_train_months: int = 36  # Walk-forward training window (months)
    wf_test_months: int = 6    # Walk-forward out-of-sample test window (months)

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
        data_config_raw = data.get('data', {})
        data_config = DataConfig(**{k: v for k, v in data_config_raw.items() if k in DataConfig.__dataclass_fields__})
        
        pairs_config_raw = data.get('pairs', {})
        pairs_config = PairsConfig(**{k: v for k, v in pairs_config_raw.items() if k in PairsConfig.__dataclass_fields__})
        
        return cls(data=data_config, pairs=pairs_config)
