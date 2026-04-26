import pytest
import os
import yaml
from src.core.config import Config, DataConfig

@pytest.fixture
def temp_config_file(tmp_path):
    """
    @brief Fixture to create a temporary configuration file.
    @param tmp_path Pytest fixture for temp directory.
    @return Path to the temporary config file.
    """
    d = tmp_path / "input"
    d.mkdir()
    config_file = d / "configuration.yaml"
    content = {
        "data": {
            "ticker": "TSLA",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31"
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(content, f)
    return str(config_file)

def test_config_load(temp_config_file):
    """
    @brief Test loading configuration from a file.
    """
    config = Config.load(temp_config_file)
    assert config.data.ticker == "TSLA"
    assert config.data.start_date == "2022-01-01"
    assert config.data.end_date == "2022-12-31"

def test_config_from_dict():
    """
    @brief Test creating configuration from a dictionary.
    """
    data = {
        "data": {
            "ticker": "MSFT",
            "start_date": "2021-01-01",
            "end_date": "2021-06-01"
        }
    }
    config = Config.from_dict(data)
    assert isinstance(config.data, DataConfig)
    assert config.data.ticker == "MSFT"

def test_config_load_not_found():
    """
    @brief Test loading a non-existent configuration file.
    """
    with pytest.raises(FileNotFoundError):
        Config.load("non_existent_file.yaml")
