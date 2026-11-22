"""Feature adapter for Indian UPI and digital wallet transactions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd


class IndianUPIPaymentsAdapter:
    """Normalize Indian payment records and derive low-cost risk features.

    Rows are enriched in input order. Velocity is computed per account over the
    trailing 60 minutes, including the current UPI attempt.
    """

    TXN_TYPES: ClassVar[set[str]] = {
        "UPI_P2P", "UPI_P2M", "WALLET_CASH_OUT", "NET_BANKING_IMPS"
    }

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``transactions`` with Indian-payment features."""
        result = transactions.copy()
        source_amount = pd.to_numeric(
            result.get("amount", pd.Series(np.nan, index=result.index)), errors="coerce"
        )
        amount_inr = pd.to_numeric(
            result.get("amount_inr", pd.Series(np.nan, index=result.index)), errors="coerce"
        )
        normalized_amount = amount_inr.fillna(source_amount).fillna(0.0).clip(lower=0)
        result["amount_inr"] = normalized_amount
        result["amount"] = normalized_amount
        if "txn_type" not in result:
            result["txn_type"] = "UPI_P2M"
        result["txn_type"] = result["txn_type"].fillna("UPI_P2M").astype(str).str.upper()
        result.loc[~result["txn_type"].isin(self.TXN_TYPES), "txn_type"] = "UPI_P2M"

        old = pd.to_numeric(result.get("old_balance", pd.Series(0.0, index=result.index)), errors="coerce").fillna(0.0)
        raw_new = pd.to_numeric(
            result.get("new_balance", pd.Series(np.nan, index=result.index)), errors="coerce"
        )
        new = raw_new.fillna(0.0)
        result["balance_drain_ratio"] = ((old - new) / old.clip(lower=1.0)).clip(0.0, 1.0)
        result["is_zero_balance_after_txn"] = (raw_new.notna() & (new == 0)).astype(int)

        if "timestamp" in result:
            timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
            hours = timestamps.dt.hour
        else:
            hours = pd.to_numeric(result.get("hour_of_day", pd.Series(12, index=result.index)), errors="coerce").fillna(12)
            timestamps = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
        result["hour_of_day"] = hours.astype(int).clip(0, 23)
        result["sin_hour"] = np.sin(2 * np.pi * result["hour_of_day"] / 24.0)
        result["cos_hour"] = np.cos(2 * np.pi * result["hour_of_day"] / 24.0)

        is_upi = result["txn_type"].str.startswith("UPI_")
        account = result.get("account_id", pd.Series("unknown", index=result.index)).astype(str)
        velocity = pd.Series(0, index=result.index, dtype=int)
        if timestamps.notna().any():
            work = pd.DataFrame({"account": account, "timestamp": timestamps, "is_upi": is_upi}, index=result.index)
            for _account, group in work[work.timestamp.notna()].groupby("account", sort=False):
                ordered = group.sort_values("timestamp", kind="stable")
                values = ordered.timestamp.astype("int64").to_numpy()
                flags = ordered.is_upi.to_numpy(dtype=int)
                counts = np.empty(len(values), dtype=int)
                left = 0
                total = 0
                for pos, stamp in enumerate(values):
                    total += flags[pos]
                    while values[left] < stamp - 3_600_000_000_000:
                        total -= flags[left]
                        left += 1
                    counts[pos] = total
                velocity.loc[ordered.index] = counts
        else:
            velocity = is_upi.groupby(account).cumsum().astype(int)
            if "upi_velocity_1h" in result:
                reported_velocity = pd.to_numeric(
                    result["upi_velocity_1h"], errors="coerce"
                )
                velocity = reported_velocity.fillna(velocity).clip(lower=0).astype(int)
        result["upi_velocity_1h"] = velocity

        recipient = result.get("upi_id", result.get("recipient_upi_id", pd.Series("", index=result.index))).fillna("").astype(str).str.lower()
        recipient_key = recipient.where(recipient.ne(""), pd.NA)
        recipient_counts = pd.Series(0, index=result.index, dtype=int)
        valid_recipient = recipient_key.notna()
        if valid_recipient.any():
            recipient_counts.loc[valid_recipient] = (
                pd.DataFrame({"account": account, "recipient": recipient_key})
                .loc[valid_recipient]
                .groupby(["account", "recipient"], dropna=True)["recipient"]
                .transform("count")
                .to_numpy()
            )
        suspicious_handle = recipient.str.contains(r"(?:cashout|loan|refund|support|verify|helpdesk|prize)", regex=True)
        recipient_speed = (velocity / 8.0).clip(0.0, 1.0)
        heuristic_mule_risk = (
            0.65 * suspicious_handle.astype(float)
            + 0.35 * recipient_speed
            + 0.1 * (recipient_counts > 3)
        ).clip(0.0, 1.0)
        supplied_mule_risk = pd.to_numeric(
            result.get("dest_mule_risk_score", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
        result["dest_mule_risk_score"] = supplied_mule_risk.fillna(
            heuristic_mule_risk
        ).clip(0.0, 1.0)
        result["device_trust_score"] = pd.to_numeric(result.get("device_trust_score", pd.Series(0.5, index=result.index)), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        result["amount_inr_formatted"] = result["amount_inr"].map(lambda amount: f"₹{amount:,.2f}")
        return result

    def adapt(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Alias for ``transform`` for callers that use adapter terminology."""
        return self.transform(transactions)
