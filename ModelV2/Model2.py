import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import deque

features = ["temperature_c","humidity_pct","pressure_hpa"]
contamination = 0.65
z_threshold = 3.0
rolling_window = 30
refit_n = 200
retrain_window = 1000

training_df = pd.read_csv("TestAnomalyUnlabled.csv")

def baseline_model(traing_df):
    model = IsolationForest(contamination=contamination,random_state=42)
    model.fit(traing_df[features])
    return model

class RollingZScore:
    def __init__(self,window=rolling_window):
        self.window = window
        self.buffers = {f: deque(maxlen=window) for f in features}

    def updatenscore(self,row:dict):
        z_scores={}
        for f in features:
            buf = self.buffers[f]
            if(len(buf)>5):
                mean = np.mean(buf)
                std = np.std(buf)
                z_scores[f] = (row[f]-mean)/std
            else:
                z_scores[f]=0.0
            buf.append(row[f])
        return z_scores

def detect_anomaly(row: dict,model,roller:RollingZScore):
    x=np.array([[row[f] for f in features]])

    iforest_pred = model.predict(x)[0]
    iforest_score = model.decision_function(x)[0]
    iforest_flag = iforest_pred == -1

    z_scores = roller.updatenscore(row)
    z_scoremax = max(abs(v) for v in z_scores.values())
    z_flag = z_scoremax > z_threshold

    is_anomaly = iforest_flag or z_flag

    return{
        "timestamp": row.get("timestamp"),
        "is_anomaly": is_anomaly,
        "iforest_score": iforest_score,
        "iforest_flag": iforest_flag,
        "zscore_max": z_scoremax,
        "z_flag": z_flag
    }