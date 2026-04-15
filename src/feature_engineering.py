import pandas as pd
import numpy as np

SENSOR_COLUMNS = [
    'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
    'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4',
    'VS1', 'SE', 'CE'
]

def extract_features_from_cycle(cycle_data: dict) -> pd.DataFrame:
    """
    Takes one cycle of sensor summary statistics and returns
    an 80-feature dataframe ready for model prediction.
    
    cycle_data: dict with keys like PS1_mean, PS1_std, etc.
    """
    return pd.DataFrame([cycle_data])

def extract_features_from_csv(filepath: str) -> pd.DataFrame:
    """
    Takes a CSV file where each row is one hydraulic cycle
    with pre-computed sensor statistics.
    
    Expected columns: PS1_mean, PS1_std, PS1_min, PS1_max, 
    PS1_range, PS2_mean... and so on for all 16 sensors.
    Also expects a 'stable' column.
    
    Returns dataframe ready for model prediction.
    """
    df = pd.read_csv(filepath)
    
    required_features = []
    for sensor in SENSOR_COLUMNS:
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            required_features.append(f'{sensor}_{stat}')
    required_features.append('stable')
    
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded CSV: {missing}")
    
    return df[required_features]

def generate_csv_template() -> pd.DataFrame:    
    """
    Generates an empty CSV template the engineer fills in.
    One row per hydraulic cycle.
    """
    columns = []
    for sensor in SENSOR_COLUMNS:
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            columns.append(f'{sensor}_{stat}')
    columns.append('stable')
    
    template = pd.DataFrame(columns=columns)
    return template

if __name__ == "__main__":
    template = generate_csv_template()
    template.to_csv('engineer_upload_template.csv', index=False)
    print(f"Template created with {len(template.columns)} columns")
    print("Engineer fills this in and uploads to the tool")