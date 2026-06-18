"""XGBoost classifier with cost-aware execution filter.

Architecture:
    1. XGBoost classifier predicts next-bar direction probability
    2. Cost-aware threshold: only trade when |prob - 0.5| > lambda * cost
    3. Momentum confirmation: EMA slope direction must align with prediction
    4. Walk-forward retraining: retrain each fold on expanding training window

Based on: Slepaczuk (2026) — cost-aware ML execution framework.

Config structure (config["strategies"]["xgboost_cost_aware"]):
    model_params: {n_estimators, max_depth, learning_rate, subsample,
                   colsample_bytree, reg_alpha, reg_lambda, min_child_weight,
                   random_state, eval_metric}
    training: {retrain_every_candles, min_train_samples, validation_fraction,
               early_stopping_rounds}
    cost_filter: {lambda, transaction_cost_bps}
    trading: {confidence_threshold, allow_shorts}
    target: {horizon, dead_zone_pct}
"""

import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal


class XGBoostCostAware(BaseStrategy):
    """XGBoost classifier with cost-aware execution and momentum confirmation.

    Predicts next-bar direction using 65+ engineered features. Only trades
    when predicted edge exceeds round-trip transaction cost by a configurable
    lambda multiplier. Confirms direction with EMA20 slope.
    """

    # Features excluded from ML training (patterns, distance-based, raw price)
    _EXCLUDE_PREFIXES = ("pattern_", "dist_sma_", "dist_high_", "dist_low_",
                         "return_5", "return_10", "return_20", "vs_4h_", "vs_1d_")
    _EXCLUDE_COLUMNS = {"price", "high_20", "low_20", "close_position",
                        "hl_ratio", "oc_ratio", "hl_range", "gap", "trend_strength"}

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategies", {}).get("xgboost_cost_aware", {})

        # Model parameters → ml/trainer.py
        model_cfg = cfg.get("model_params", {})
        self.model_params = {
            "n_estimators": model_cfg.get("n_estimators", 200),
            "max_depth": model_cfg.get("max_depth", 5),
            "learning_rate": model_cfg.get("learning_rate", 0.05),
            "subsample": model_cfg.get("subsample", 0.7),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.6),
            "reg_alpha": model_cfg.get("reg_alpha", 2.0),
            "reg_lambda": model_cfg.get("reg_lambda", 3.0),
            "min_child_weight": model_cfg.get("min_child_weight", 1),
            "random_state": model_cfg.get("random_state", 42),
            "eval_metric": model_cfg.get("eval_metric", "logloss"),
        }

        # Training settings
        training_cfg = cfg.get("training", {})
        self.min_train_samples = training_cfg.get("min_train_samples", 1000)
        self.val_fraction = training_cfg.get("validation_fraction", 0.2)
        self.early_stopping_rounds = training_cfg.get("early_stopping_rounds", 20)
        self.retrain_every_candles = training_cfg.get("retrain_every_candles", 500)

        # Cost filter
        cost_cfg = cfg.get("cost_filter", {})
        self.lambda_cost = cost_cfg.get("lambda", 4.0)
        tx_cost_bps = cost_cfg.get("transaction_cost_bps", 30)
        self.tx_cost = tx_cost_bps / 10000  # bps → decimal

        # Trading rules
        trading_cfg = cfg.get("trading", {})
        self.confidence_threshold = trading_cfg.get("confidence_threshold", 0.55)
        self.allow_shorts = trading_cfg.get("allow_shorts", True)

        # Target construction
        target_cfg = cfg.get("target", {})
        self.horizon = target_cfg.get("horizon", 4)
        self.dead_zone_pct = target_cfg.get("dead_zone_pct", 0.001)

        # State
        self.model = None
        self._feature_names: list[str] = []
        self._last_forecast = 0.5
        self._candles_since_retrain = 0
        self._last_train_n = 0

        # WalkForwardTrainer — used for target building and prediction (not training directly)
        from ml.trainer import WalkForwardTrainer
        trainer_cfg = {"strategies": {"xgboost_cost_aware": {"model_params": self.model_params}}}
        self.trainer = WalkForwardTrainer(trainer_cfg)

        logger.info(
            f"XGBoostCostAware init: λ={self.lambda_cost} | "
            f"tx_cost={tx_cost_bps}bps | horizon={self.horizon} | "
            f"dead_zone={self.dead_zone_pct:.4f} | confidence≥{self.confidence_threshold}"
        )

    @property
    def name(self) -> str:
        return "XGBoostCostAware"

    # ─── Feature Selection ──────────────────────────────────────

    @staticmethod
    def _is_excluded_feature(col: str) -> bool:
        """Check if a feature column should be excluded from ML training."""
        if col in XGBoostCostAware._EXCLUDE_COLUMNS:
            return True
        for prefix in XGBoostCostAware._EXCLUDE_PREFIXES:
            if col.startswith(prefix):
                return True
        return False

    def _get_feature_columns(self, columns: list[str]) -> list[str]:
        """Filter available columns to the ML-safe training set."""
        return [c for c in columns if not self._is_excluded_feature(c)]

    # ─── Retraining ─────────────────────────────────────────────

    def retrain(self, historical_data: pd.DataFrame):
        """Build target and train XGBoost model on walk-forward training window.

        Called by BacktestEngine per fold. Constructs trinary target from
        forward return, filters noise via dead zone, trains classifier.
        """
        if "close" not in historical_data.columns:
            logger.warning("XGBoostCostAware.retrain: 'close' column missing — skipping")
            return

        df = historical_data.copy()
        n_total = len(df)

        # 1. Build trinary target: UP (+1) / DOWN (0) / NOISE (dropped)
        future_close = df["close"].shift(-self.horizon)
        future_return = (future_close - df["close"]) / df["close"]

        # UP if return > dead_zone, DOWN if < -dead_zone, NOISE otherwise
        y_raw = pd.Series(-1, index=df.index)
        y_raw[future_return > self.dead_zone_pct] = 1
        y_raw[abs(future_return) <= self.dead_zone_pct] = 0  # noise
        y_raw[future_return.isna()] = 0  # last horizon bars = noise

        # Drop noise rows, convert to binary (UP=1, DOWN=0)
        mask_keep = y_raw != 0
        y = (y_raw[mask_keep] == 1).astype(int)

        # 2. Select ML features
        avail_cols = [c for c in df.columns if c not in ("timestamp", "close")]
        feature_cols = self._get_feature_columns(avail_cols)
        if not feature_cols:
            logger.warning("XGBoostCostAware.retrain: no usable feature columns — skipping")
            return
        X = df.loc[mask_keep, feature_cols].copy()

        # 3. Clean data
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.loc[X.index].dropna()
        X = X.loc[y.index]

        # 4. Check minimum samples
        if len(X) < max(self.min_train_samples, 100):
            logger.warning(
                f"XGBoostCostAware.retrain: insufficient samples ({len(X)} < "
                f"{self.min_train_samples}) — skipping"
            )
            return

        # 5. Chronological train/val split for early stopping
        n_val = max(1, int(len(X) * self.val_fraction))
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        # 6. Train model with early stopping
        try:
            self.model = self.trainer.train(
                X_train, y_train,
                X_val=X_val if len(X_val) > 10 else None,
                y_val=y_val if len(y_val) > 10 else None,
            )
            if self.model is not None:
                self._feature_names = feature_cols
                self._last_train_n = len(X_train)
                self._candles_since_retrain = 0

                # Log feature importance (top 5)
                try:
                    imp = self.model.get_booster().get_score(importance_type="gain")
                    top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                    top5_str = ", ".join(f"{k}={v:.0f}" for k, v in top5)
                    logger.info(
                        f"XGBoostCostAware trained: {len(X_train)} samples, "
                        f"{len(feature_cols)} features. Top 5 gain: {top5_str}"
                    )
                except Exception:
                    logger.info(f"XGBoostCostAware trained: {len(X_train)} samples")

        except Exception as e:
            logger.error(f"XGBoostCostAware.retrain: training failed — {e}")

    # ─── Prediction ─────────────────────────────────────────────

    def _predict(self, features: dict) -> float:
        """Predict upward probability from a features dict.

        Returns 0.5 (neutral) if model is unavailable or prediction fails.
        """
        if self.model is None or not self._feature_names:
            return 0.5
        try:
            prob = self.trainer.predict_single(self.model, features)
            if prob is None or np.isnan(prob):
                return 0.5
            return float(prob)
        except Exception as e:
            logger.debug(f"XGBoostCostAware._predict failed: {e}")
            return 0.5

    # ─── Signal Generation ─────────────────────────────────────

    def on_candle(self, candle: dict, features: pd.Series) -> Signal:
        """Generate trading signal from prediction + cost filter + momentum.

        Called by BacktestEngine each bar. Returns LONG, SHORT, or FLAT.
        The engine syncs in_position/position_side before calling.
        """
        self._candles_since_retrain += 1

        if self.model is None:
            return Signal.FLAT

        # Convert features to dict for prediction
        feats_dict = features.to_dict() if hasattr(features, "to_dict") else dict(features)

        prob = self._predict(feats_dict)
        self._last_forecast = prob

        # Cost-aware threshold
        edge = prob - 0.5
        lambda_cost = self.lambda_cost * self.tx_cost

        long_ok = prob >= self.confidence_threshold and edge > lambda_cost
        short_ok = (1.0 - prob) >= self.confidence_threshold and (-edge) > lambda_cost

        # Momentum confirmation (EMA20 slope)
        ema_slope = feats_dict.get("ema_20_slope", 0.0)

        if long_ok and ema_slope > 0:
            if getattr(self, "in_position", False) and getattr(self, "position_side", "") == "short":
                return Signal.LONG  # Exit short + enter long
            return Signal.LONG

        if short_ok and self.allow_shorts and ema_slope < 0:
            if getattr(self, "in_position", False) and getattr(self, "position_side", "") == "long":
                return Signal.SHORT  # Exit long + enter short
            return Signal.SHORT

        return Signal.FLAT

    # ─── State Management ───────────────────────────────────────

    def on_position_closed(self):
        """Sync internal state when position is closed externally."""
        pass  # XGBoostCostAware has no internal position state beyond what engine provides

    def reset_state(self):
        """Reset between backtest folds (fresh model per fold)."""
        self.model = None
        self._feature_names = []
        self._last_forecast = 0.5
        self._candles_since_retrain = 0
        self._last_train_n = 0

    # ─── Diagnostics ────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        """Return current model state for logging or dashboard."""
        return {
            "model_trained": self.model is not None,
            "features_used": len(self._feature_names),
            "last_train_samples": self._last_train_n,
            "last_forecast": round(self._last_forecast, 4),
            "candles_since_retrain": self._candles_since_retrain,
        }
