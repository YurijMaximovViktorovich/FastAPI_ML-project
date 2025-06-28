import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import sys

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['mean_pressure'] = (2 * X['diastolic_blood_pressure'] * X['systolic_blood_pressure']) / 3
        X['lipid_ratio'] = X['triglycerides'] / (X['cholesterol'] + 1)
        X['activity_balance_score'] = X['exercise_hours_per_week'] / (X['sedentary_hours_per_day'] + 1)
        X['risk_age'] = X['age'] * (1 + X['family_history'])
        X['health_index'] = (X['bmi'] + X['blood_sugar']) / 2
        X['sleep_stress_ratio'] = X['sleep_hours_per_day'] / (X['stress_level'] + 1)

        # Удаляем исходные признаки
        X = X.drop(columns=[
            'id', 'diastolic_blood_pressure', 'systolic_blood_pressure',
            'cholesterol', 'triglycerides', 'blood_sugar', 'bmi',
            'heart_rate', 'exercise_hours_per_week', 'sedentary_hours_per_day',
            'sleep_hours_per_day', 'stress_level', 'family_history',
            'age'
        ], errors='ignore')
        return X

class ThresholdClassifierWithTransform(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.transformer = FeatureTransformer()
    
    def fit(self, X, y):
        X_transformed = self.transformer.fit_transform(X)
        X_transformed = X_transformed.dropna()
        y = y[X_transformed.index]
        self.model.fit(X_transformed, y)
        return self
    
    def predict(self, X):
        X_transformed = self.transformer.transform(X)
        return (self.model.predict_proba(X_transformed)[:, 1] > self.threshold).astype(int)

    def predict_proba(self, X):
        X_transformed = self.transformer.transform(X)
        return self.model.predict_proba(X_transformed)


sys.modules['__main__'].ThresholdClassifierWithTransform = ThresholdClassifierWithTransform
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