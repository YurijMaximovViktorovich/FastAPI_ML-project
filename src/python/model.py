import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import sys

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Кастомный классификатор с пороговым решением"""
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

sys.modules['__main__'].ThresholdClassifier = ThresholdClassifier
class Model:
    """Класс для работы с сохранённой моделью в FastAPI"""
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = self._load_model(model_path)
        self.threshold = threshold

    def _load_model(self, path: str):
        """Безопасная загрузка модели"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)

    def predict(self, data: pd.DataFrame) -> list:
        """Предсказание для новых данных"""
        try:
            proba = self.model.predict_proba(data)[:, 1]
            return (proba >= self.threshold).astype(int).tolist()
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")