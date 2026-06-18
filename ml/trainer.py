"""Walk-forward XGBoost training pipeline.

Based on: Slepaczuk (2026) — walk-forward cross-validation with XGBoost
for hourly BTC prediction. Each fold trains on past data, tests on
unseen future data, with strict temporal ordering.

Target: next-bar return sign (classification) or return magnitude (regression).
        Classification mode: predict up/down for directional signals.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


class WalkForwardTrainer:
    """Trains XGBoost models using strict walk-forward temporal validation.

    Usage:
        trainer = WalkForwardTrainer(config)
        model = trainer.train(X_train, y_train)
        predictions = trainer.predict(model, X_test)
        fold_results = trainer.walk_forward_train(X, y, features_df)
    """

    def __init__(self, config: dict):
        ml_cfg = config.get("strategies", {}).get("xgboost_cost_aware", {})
        self.model_params = ml_cfg.get("model_params", {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "random_state": 42,
        })

        train_cfg = ml_cfg.get("training", {})
        self.n_folds = train_cfg.get("walk_forward_folds", 27)
        self.min_train_samples = train_cfg.get("min_train_samples", 500)
        self.target_type = train_cfg.get("target_type", "classification")  # or "regression"
        self.retrain_every = train_cfg.get("retrain_every_candles", 500)

        # Model storage
        self.models_dir = Path(config.get("paths", {}).get("models_dir", "./models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Track last training
        self.last_model: Optional[xgb.XGBClassifier] = None
        self.training_metrics: dict = {}

    @staticmethod
    def build_target(close: pd.Series, horizon: int = 1) -> Tuple[pd.Series, pd.Series]:
        """Build target variable from price series.

        Classification mode:
            y = 1 if next close > current close, 0 otherwise (up/down)

        Regression mode:
            y = (close.shift(-horizon) / close) - 1  (next bar return)

        Returns (y, mask) where mask is True for valid (non-NaN) rows.
        """
        if horizon <= 0:
            raise ValueError("horizon must be >= 1")

        future_close = close.shift(-horizon)
        y = (future_close > close).astype(int)  # Classification: up/down
        mask = future_close.notna()
        return y, mask

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> xgb.XGBClassifier:
        """Train an XGBoost classifier on features X and target y.

        Args:
            X: Feature DataFrame (n_samples, n_features). Must be finite.
            y: Target Series (0 or 1). Must be finite.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation target for early stopping.

        Returns:
            Trained XGBoost model, or None if insufficient data.
        """
        # Ensure clean data
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

        if len(X_clean) < 100:
            logger.warning(f"Insufficient training data: {len(X_clean)} samples")
            return None

        model = xgb.XGBClassifier(**self.model_params)
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_val_clean = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            fit_kwargs["eval_set"] = [(X_val_clean, y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_clean, y, **fit_kwargs)

        self.last_model = model

        logger.debug(
            f"Trained XGBoost: {len(X_clean)} samples, "
            f"{X_clean.shape[1]} features, "
            f"class balance={y.mean():.2f}"
        )

        return model

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from a trained model.

        Returns predicted probabilities for class 1 (up).
        """
        if model is None:
            return np.zeros(len(X))

        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return model.predict_proba(X_clean)[:, 1]

    def predict_single(self, model, features: dict) -> float:
        """Predict for a single row (live trading).

        Returns probability of upward move.
        """
        if model is None:
            return 0.5  # Neutral

        # Convert dict to DataFrame row
        X = pd.DataFrame([features])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Ensure columns match training
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[model.feature_names_in_]

        return model.predict_proba(X)[:, 1][0]

    def walk_forward_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[dict]:
        """Run walk-forward training with expanding window.

        Each fold trains on all data before the test window, then predicts
        on the test window. Returns per-fold metrics.

        Returns:
            List of dicts with keys: fold, train_n, test_n, accuracy, precision,
            recall, f1, model
        """
        fold_results = []
        fold_size = max(len(X) // self.n_folds, 50)

        for fold in range(1, self.n_folds):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, len(X))

            X_train = X.iloc[:test_start]
            y_train = y.iloc[:test_start]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            if len(X_train) < self.min_train_samples:
                continue
            if len(X_test) < 20:
                continue

            model = self.train(X_train, y_train)
            if model is None:
                continue

            probs = self.predict(model, X_test)
            preds = (probs >= 0.5).astype(int)

            # Metrics
            accuracy = (preds == y_test).mean()
            tp = ((preds == 1) & (y_test == 1)).sum()
            fp = ((preds == 1) & (y_test == 0)).sum()
            fn = ((preds == 0) & (y_test == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            fold_results.append({
                "fold": fold,
                "train_n": len(X_train),
                "test_n": len(X_test),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "model": model,
            })

            logger.debug(
                f"Fold {fold}: acc={accuracy:.3f} prec={precision:.3f} "
                f"rec={recall:.3f} f1={f1:.3f}"
            )

        if fold_results:
            avg_acc = np.mean([r["accuracy"] for r in fold_results])
            avg_f1 = np.mean([r["f1"] for r in fold_results])
            logger.info(
                f"Walk-forward: {len(fold_results)} folds, "
                f"avg acc={avg_acc:.3f}, avg f1={avg_f1:.3f}"
            )
            self.training_metrics = {"avg_accuracy": avg_acc, "avg_f1": avg_f1}
        else:
            logger.warning("No walk-forward folds completed")

        return fold_results

    def train_final(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Train a model on all available data (for deployment)."""
        return self.train(X, y)

    def save_model(self, model, name: str):
        """Save model to JSON file (XGBoost native format)."""
        path = self.models_dir / f"{name}.json"
        model.save_model(str(path))
        logger.info(f"Model saved: {path}")

    def load_model(self, name: str):
        """Load a saved model from JSON file."""
        path = self.models_dir / f"{name}.json"
        if not path.exists():
            logger.warning(f"Model not found: {path}")
            return None

        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model

    def feature_importance(self, model, feature_names: list[str]) -> dict:
        """Return feature importance dict sorted by gain."""
        if model is None:
            return {}

        importances = model.get_booster().get_score(importance_type="gain")
        # Map f0, f1, ... to actual names
        named = {}
        for k, v in importances.items():
            idx = int(k.replace("f", ""))
            if idx < len(feature_names):
                named[feature_names[idx]] = v

        return dict(sorted(named.items(), key=lambda x: -x[1]))
