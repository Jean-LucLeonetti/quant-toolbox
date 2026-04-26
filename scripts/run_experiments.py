import yaml
import subprocess
import os

def update_config(universe, coint_mode, hedge_mode, wf_train=36, wf_test=6):
    config_path = "input/configuration.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config['data']['universe'] = universe
    config['pairs']['coint_mode'] = coint_mode
    config['pairs']['hedge_mode'] = hedge_mode
    config['pairs']['wf_train_months'] = wf_train
    config['pairs']['wf_test_months'] = wf_test
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

# All universes to sweep
universes = ["etf_sector", "etf_country", "etf_commodity", "sp500"]
coint_modes = ["engle_granger", "johansen"]
hedge_modes = ["static_ols", "kalman_filter"]

for uni in universes:
    print(f"\n{'='*50}\nRunning data_build for universe: {uni}\n{'='*50}")
    update_config(uni, "engle_granger", "static_ols")
    
    result = subprocess.run([".venv/bin/python3", "main.py", "data_build"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  data_build FAILED: {result.stderr[-300:]}")
        continue
    
    for coint in coint_modes:
        for hedge in hedge_modes:
            label = f"{uni} | {coint} | {hedge}"
            print(f"\n=> Experiment: {label}")
            update_config(uni, coint, hedge)
            result = subprocess.run([".venv/bin/python3", "main.py", "pairs"], check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[-300:]}")
            else:
                # Extract the WF Sharpe line from stdout/stderr (logger sends to stdout)
                combined_output = result.stdout + "\n" + result.stderr
                for line in combined_output.split('\n'):
                    if 'Walk-Forward' in line or 'WF fold' in line:
                        print(f"  {line.split(' - ')[-1]}")
            
print("\nAll experiments complete. Check output/runs/experiments_log.csv")
