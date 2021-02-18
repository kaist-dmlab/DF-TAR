import numpy as np

def final_score(y_pred, threshold=0.5):
    y_mae = np.where(y_pred[0] < threshold, 0, np.ceil(y_pred[0])).astype(int)
    y_mse = np.where(y_pred[1] < threshold, 0, np.ceil(y_pred[1])).astype(int)

    y_pred_avg = np.average([y_mae, y_mse], axis=0)
    y_pred_avg = np.around(y_pred_avg).astype(int)

    return y_pred_avg