"""
ml_optimizer.py — Her 50 trade'de sinyal ağırlıklarını optimize et.
RandomForest (scikit-learn) + basit policy network (PyTorch) opsiyonel.
"""

from __future__ import annotations
import json
import os
import time
import pickle
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from config import CFG
from models import TradeRecord
from logger import log

if TYPE_CHECKING:
    from models import BotState

# Kütüphane kontrolü
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn yüklü değil — ML devre dışı")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── PyTorch Policy Network ───────────────────────────────────────

class PolicyNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Basit 3 katmanlı policy network.
    Input:  8 özellik (RSI, EMA cross, OB imbalance, liq proximity, ATR, vb.)
    Output: 1 sinyal (0=geç, 1=al/sat)
    """
    def __init__(self, input_dim: int = 8, hidden: int = 32):
        if TORCH_AVAILABLE:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.net(x)


# ─── Feature extraction ───────────────────────────────────────────

def _extract_features(record: TradeRecord) -> Optional[List[float]]:
    """TradeRecord'dan ML için özellik vektörü çıkar."""
    try:
        features = [
            record.rsi_at_entry / 100.0,                      # 0-1 normalize
            1.0 if record.ema_cross_at_entry else 0.0,
            min(record.ob_imbalance_at_entry / 3.0, 1.0),     # max 3x normalize
            1.0 - min(record.liq_proximity_at_entry / 2.0, 1.0),  # yakınsa yüksek
            min(record.atr_at_entry / 2.0, 1.0),              # normalize
            1.0 if record.side == "Buy" else 0.0,              # long=1 short=0
            min(record.duration_sec / 300.0, 1.0),            # max 5dk normalize
            record.signal_score,                               # ML skoru
        ]
        return features
    except Exception:
        return None


# ─── Ana ML sınıfı ────────────────────────────────────────────────

class MLOptimizer:
    def __init__(self):
        self.rf_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.policy_net: Optional[PolicyNetwork] = None
        self.is_ready: bool = False
        self.last_train_acc: float = 0.0
        self.feature_importances: dict = {}
        self.model_path = os.path.join(CFG.log_dir, "ml_model.pkl")
        self._load_model()

    def _load_model(self):
        """Daha önce kaydedilmiş modeli yükle."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                    self.rf_model = saved.get("rf")
                    self.scaler   = saved.get("scaler")
                    self.is_ready = saved.get("ready", False)
                log.info(f"📂 ML modeli yüklendi: {self.model_path}")
            except Exception as e:
                log.warning(f"Model yüklenemedi: {e}")

    def _save_model(self):
        """Modeli diske kaydet."""
        os.makedirs(CFG.log_dir, exist_ok=True)
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "rf":     self.rf_model,
                    "scaler": self.scaler,
                    "ready":  self.is_ready,
                }, f)
            log.debug("💾 ML modeli kaydedildi")
        except Exception as e:
            log.warning(f"Model kaydedilemedi: {e}")

    def train(self, trade_history: List[TradeRecord]) -> bool:
        """
        Son N trade ile RandomForest'ı eğit.
        Label: kazanan trade = 1, kaybeden = 0
        """
        if not SKLEARN_AVAILABLE:
            return False

        recent = trade_history[-CFG.ml_lookback_trades:]
        if len(recent) < CFG.ml_min_samples:
            log.debug(f"ML için yeterli örnek yok: {len(recent)}/{CFG.ml_min_samples}")
            return False

        X, y = [], []
        for record in recent:
            features = _extract_features(record)
            if features is not None:
                X.append(features)
                y.append(1 if record.is_win else 0)

        if len(X) < CFG.ml_min_samples:
            return False

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        # Dengesiz sınıf kontrolü
        win_count  = y_arr.sum()
        loss_count = len(y_arr) - win_count
        log.info(f"🤖 ML Eğitim | Toplam: {len(X_arr)} | Win: {win_count} | Loss: {loss_count}")

        if win_count < 5 or loss_count < 5:
            log.warning("ML: Yeterli win/loss dağılımı yok")
            return False

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        # RandomForest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_scaled, y_arr)

        # Cross-validation
        try:
            cv_scores = cross_val_score(self.rf_model, X_scaled, y_arr, cv=3, scoring="accuracy")
            self.last_train_acc = float(cv_scores.mean())
            log.info(f"✅ RF CV Accuracy: {self.last_train_acc:.1%} ± {cv_scores.std():.1%}")
        except Exception:
            self.last_train_acc = float(self.rf_model.score(X_scaled, y_arr))

        # Feature importance
        feature_names = [
            "rsi", "ema_cross", "ob_imbalance", "liq_proximity",
            "atr", "side", "duration", "signal_score",
        ]
        importances = self.rf_model.feature_importances_
        self.feature_importances = dict(zip(feature_names, importances.tolist()))
        log.info(f"📊 Feature Importances: {self.feature_importances}")

        self.is_ready = True
        self._save_model()

        # PyTorch policy network'ü de eğit (opsiyonel)
        if TORCH_AVAILABLE and len(X_arr) >= 50:
            self._train_policy_net(X_arr, y_arr)

        return True

    def _train_policy_net(self, X: np.ndarray, y: np.ndarray):
        """Basit policy network eğitimi (PyTorch)."""
        try:
            self.policy_net = PolicyNetwork(input_dim=X.shape[1])
            optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)

            for epoch in range(50):
                optimizer.zero_grad()
                pred = self.policy_net(X_tensor)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimizer.step()

            log.debug(f"Policy network eğitildi. Son loss: {loss.item():.4f}")
        except Exception as e:
            log.warning(f"Policy network hatası: {e}")

    def get_signal_multiplier(self, symbol: str, state: "BotState") -> float:
        """
        Mevcut piyasa koşulları için sinyal çarpanı döndür (0.5 – 1.5).
        Model hazır değilse 1.0 (nötr) döndür.
        """
        if not self.is_ready or self.rf_model is None:
            return 1.0

        ob = state.orderbooks.get(symbol)
        if ob is None:
            return 1.0

        # Basit feature vector (tarihsel veriye erişim olmadan)
        features = [
            0.5,   # rsi (bilinmiyor, nötr)
            0.5,   # ema_cross
            ob.imbalance_ratio() / 3.0,
            0.5,   # liq proximity
            0.3,   # atr
            0.5,   # side
            0.5,   # duration
            0.5,   # signal_score
        ]

        try:
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            proba = self.rf_model.predict_proba(X_scaled)[0][1]  # win olasılığı
            # 0.5 = nötr (1.0x), 0.7 = güçlü (1.4x), 0.3 = zayıf (0.6x)
            multiplier = 0.5 + proba  # [0.5, 1.5] aralığı
            return float(multiplier)
        except Exception:
            return 1.0

    def maybe_retrain(self, state: "BotState") -> bool:
        """
        Yeterince yeni trade birikmişse modeli yeniden eğit.
        Her CFG.ml_retrain_every_n_trades trade'de bir çalışır.
        """
        n = state.total_trades
        if n < CFG.ml_min_samples:
            return False
        if n - state.last_ml_retrain_at < CFG.ml_retrain_every_n_trades:
            return False

        log.info(f"🔄 ML yeniden eğitiliyor... ({n} trade)")
        success = self.train(state.trade_history)
        if success:
            state.last_ml_retrain_at = n
        return success


async def ml_optimizer_loop(state: "BotState", optimizer: MLOptimizer):
    """
    Arkaplanda her 60 saniyede maybe_retrain çağır.
    Trade log queue'ya gerek yok — state.trade_history kullanılır.
    """
    import asyncio
    while state.is_running:
        await asyncio.sleep(60)
        optimizer.maybe_retrain(state)
