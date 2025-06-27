from pathlib import Path
import pandas as pd
import numpy as np

ROOT_PATH = Path(__file__).parent.parent  
REPORT_PATH = ROOT_PATH / "reports"
DATA_PATH = ROOT_PATH / "data"
LOGS_PATH = ROOT_PATH / "logs"
MODEL_PATH = ROOT_PATH / "models"

MODEL_FILE = MODEL_PATH / "best_model.joblib"

# ROOT_PATH = "C:\Users\Юра\VS_Code\FastAPI_ML-project\FastAPI_ML-project"
# REPORT_PATH = Path(ROOT_PATH) / "reports"
# DATA_PATH = Path(ROOT_PATH) / "data"
# MODEL_FILE = Path(ROOT_PATH) / "models" / "best_model.joblib"
# MODEL_PATH = Path(ROOT_PATH) / "models"


def bool_to_int(x):
    """Преобразует любые булевы значения в целые числа"""
    if isinstance(x, pd.DataFrame):
        bool_cols = x.select_dtypes(include=['bool']).columns
        x[bool_cols] = x[bool_cols].astype(int)
        return x
    elif isinstance(x, (np.ndarray, pd.Series, list)):
        return x.astype(int) if str(x.dtype) == 'bool' else x
    else:
        try:
            return int(x) if isinstance(x, bool) else x
        except:
            return x
