import yaml
import subprocess
import os

def update_config(universe, coint_mode, hedge_mode):
    config_path = "input/configuration.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config['data']['universe'] = universe
    config['pairs']['coint_mode'] = coint_mode
    config['pairs']['hedge_mode'] = hedge_mode
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

universes = ["etf_sector", "etf_country", "etf_commodity"]
coint_modes = ["engle_granger", "johansen"]
hedge_modes = ["static_ols", "kalman_filter"]

for uni in universes:
    print(f"\\n{'='*50}\\nRunning data_build for universe: {uni}\\n{'='*50}")
    # Set to some default for data build
    update_config(uni, "engle_granger", "static_ols")
    
    # Run data build
    subprocess.run([".venv/bin/python3", "main.py", "data_build"], check=False)
    
    for coint in coint_modes:
        for hedge in hedge_modes:
            print(f"\\n=> Experiment: {uni} | {coint} | {hedge}")
            update_config(uni, coint, hedge)
            # Run pairs
            subprocess.run([".venv/bin/python3", "main.py", "pairs"], check=False)
            
print("\\nAll experiments complete.")
