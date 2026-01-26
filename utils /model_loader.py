import joblib
import os

def load_stage1_model(threshold: float, model_dir: str):
    """
    threshold에 따라 다른 Stage1 모델 로드
    """
    th_map = {
        0.2: "stage1_catboost_t02.pkl",
        0.3: "stage1_catboost_t03.pkl",
        0.4: "stage1_catboost_t04.pkl",
    }

    if threshold not in th_map:
        raise ValueError("지원하지 않는 threshold")

    path = os.path.join(model_dir, th_map[threshold])
    return joblib.load(path)


def load_stage2(model_dir: str):
    m2 = joblib.load(os.path.join(model_dir, "stage2_xgboost.pkl"))
    enc = joblib.load(os.path.join(model_dir, "encoder_s2.pkl"))
    return m2, enc
