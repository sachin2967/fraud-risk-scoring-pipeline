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
        self.upi_enabled = "txn_type" in df.columns
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

        # Keep legacy artifacts byte-for-byte compatible in feature width. New
        # training data that includes txn_type opts into the Indian payment block.
        if getattr(self, "upi_enabled", False):
            processed["balance_drain_ratio"] = pd.to_numeric(
                df.get("balance_drain_ratio", 0.0), errors="coerce"
            )
            if not isinstance(processed["balance_drain_ratio"], pd.Series):
                processed["balance_drain_ratio"] = 0.0
            processed["balance_drain_ratio"] = processed["balance_drain_ratio"].fillna(0.0)
            processed["is_zero_balance_after_txn"] = pd.to_numeric(
                df.get("is_zero_balance_after_txn", 0), errors="coerce"
            )
            if not isinstance(processed["is_zero_balance_after_txn"], pd.Series):
                processed["is_zero_balance_after_txn"] = 0
            processed["is_zero_balance_after_txn"] = processed[
                "is_zero_balance_after_txn"
            ].fillna(0).astype(int)
            processed["upi_velocity_1h"] = pd.to_numeric(
                df.get("upi_velocity_1h", 0), errors="coerce"
            )
            if not isinstance(processed["upi_velocity_1h"], pd.Series):
                processed["upi_velocity_1h"] = 0
            processed["upi_velocity_1h"] = processed["upi_velocity_1h"].fillna(0)
            processed["dest_mule_risk_score"] = pd.to_numeric(
                df.get("dest_mule_risk_score", 0.0), errors="coerce"
            )
            if not isinstance(processed["dest_mule_risk_score"], pd.Series):
                processed["dest_mule_risk_score"] = 0.0
            processed["dest_mule_risk_score"] = processed[
                "dest_mule_risk_score"
            ].fillna(0.0)
            txn_types = df.get("txn_type", pd.Series("UPI_P2M", index=df.index))
            txn_arr = txn_types.fillna("UPI_P2M").astype(str).to_numpy().reshape(-1, 1)
            if fit_encoder:
                self.txn_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoded_txn = self.txn_encoder.fit_transform(txn_arr)
            else:
                encoded_txn = self.txn_encoder.transform(txn_arr)
            for idx, name in enumerate(self.txn_encoder.get_feature_names_out(["txn_type"])):
                processed[f"{name}"] = encoded_txn[:, idx]

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
