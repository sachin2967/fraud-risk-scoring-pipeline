"""Feature engineering pipeline for financial transaction risk scoring."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FraudFeaturePipeline:
    """Transforms raw transaction events into numerical feature vectors."""

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fits encoders and builds feature matrix."""
        y = df["is_fraud"].copy()
        X = self._transform_features(df, fit_encoder=True)
        self.is_fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new production transaction batch."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming.")
        return self._transform_features(df, fit_encoder=False)

    def _transform_features(
        self, df: pd.DataFrame, fit_encoder: bool = False
    ) -> pd.DataFrame:
        processed = pd.DataFrame(index=df.index)

        # 1. Cyclical time encoding (hour of day)
        hour = df["hour_of_day"].values
        processed["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        processed["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        # 2. Financial Amount & Deviation Features
        processed["log_amount"] = np.log1p(df["amount"].values)
        processed["amount_deviation_ratio"] = df["amount_deviation_ratio"].values
        processed["velocity_1h"] = df["velocity_1h"].values
        processed["amount_per_velocity"] = df["amount"].values / (
            df["velocity_1h"].values + 1.0
        )

        # 3. Geo & Device Trust Signals
        processed["log_time_since_last_sec"] = np.log1p(
            df["time_since_last_sec"].values
        )
        processed["geo_distance_km"] = df["geo_distance_km"].values
        processed["device_trust_score"] = df["device_trust_score"].values
        processed["is_foreign"] = df["is_foreign"].values

        # 4. Merchant Category Encoding
        merch_arr = df[["merchant_category"]].values
        if fit_encoder:
            encoded_cats = self.encoder.fit_transform(merch_arr)
        else:
            encoded_cats = self.encoder.transform(merch_arr)

        cat_names = [
            f"cat_{c.replace(' ', '_')}"
            for c in self.encoder.get_feature_names_out(["merchant_category"])
        ]
        for idx, name in enumerate(cat_names):
            processed[name] = encoded_cats[:, idx]

        self.feature_names = list(processed.columns)
        return processed
