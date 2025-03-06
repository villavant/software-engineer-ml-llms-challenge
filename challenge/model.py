import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Union, List


class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._features = None
        self._model_path = os.path.join(os.path.dirname(__file__), "models", "delay_model.joblib")
        self._features_path = os.path.join(os.path.dirname(__file__), "models", "features.joblib")        
                    
    def _get_period_day(self, date: str) -> str:
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        if datetime.strptime("05:00", '%H:%M').time() <= date_time <= datetime.strptime("11:59", '%H:%M').time():
            return 'mañana'
        elif datetime.strptime("12:00", '%H:%M').time() <= date_time <= datetime.strptime("18:59", '%H:%M').time():
            return 'tarde'
        return 'noche'
    def _is_high_season(self, fecha: str) -> int:
        fecha_año = int(fecha.split('-')[0])
        fecha_dt = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        ranges = [
            ('15-Dec', '31-Dec'), ('1-Jan', '3-Mar'),
            ('15-Jul', '31-Jul'), ('11-Sep', '30-Sep')
        ]
        return int(any(
            datetime.strptime(start, '%d-%b').replace(year=fecha_año) <= fecha_dt <= datetime.strptime(end, '%d-%b').replace(year=fecha_año)
            for start, end in ranges
        ))

    def _get_min_diff(self, row: pd.Series) -> float:
        return ((datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')).total_seconds()) / 60
    
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        
        if 'Fecha-O' in data.columns and 'min_diff' not in data.columns:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data['delay'] = np.where(data['min_diff'] > 15, 1, 0)
        
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)
        
        if self._features is not None:
            for col in self._features:
                if col not in features.columns:
                    features[col] = 0
            features = features[self._features]
        
        if target_column and target_column in data.columns:
            return features, data[target_column]
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._features = [
            "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
            "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
        ]
        self._features = [f for f in self._features if f in X.columns]
        
        X_selected = X[self._features]
        scale = len(y[y == 0]) / max(len(y[y == 1]), 1)
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(X_selected, y)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            self.load_model()
        processed_data = self.preprocess(X)
        return self._model.predict(processed_data)
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            self.load_model()
        processed_data = self.preprocess(X)
        return self._model.predict_proba(processed_data)[:, 1]

    def save_model(self) -> None:
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        joblib.dump(self._model, self._model_path)
        joblib.dump(self._features, self._features_path)

    def load_model(self) -> bool:
        if os.path.exists(self._model_path) and os.path.exists(self._features_path):
            self._model = joblib.load(self._model_path)
            self._features = joblib.load(self._features_path)
            return True
        return False            