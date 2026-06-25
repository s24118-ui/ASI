"""Logowanie predykcji oraz okresowa weryfikacja (co N-tą predykcję).

Moduł realizuje dwa zadania, niezależnie od tego, czy predykcja pochodzi
z interfejsu Streamlit (`app.py`) czy z API FastAPI
(`credit_scoring.serving.api`):

1. Logowanie predykcji — każda predykcja (wejście, wynik, prawdopodobieństwa,
   znacznik czasu, źródło) jest dopisywana jako jedna linia JSON do pliku
   `data/09_predictions/predictions_log.jsonl`.

2. **Weryfikacja co N-tą predykcję** (domyślnie N=10) — po zalogowaniu
   N-tej, 2N-tej, 3N-tej... predykcji moduł automatycznie:
   - sprawdza integralność ostatnich N wpisów (poprawność wektora cech,
     sumowanie się prawdopodobieństw do 1.0, brak NaN/Inf),
   - wylicza statystyki (średnia ufność modelu, rozkład klas w danej partii),
   - zapisuje wynik takiej weryfikacji do
     `data/09_predictions/verification_log.jsonl`,
   - loguje wynik (OK / WARNING) standardowym modułem `logging`.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_scoring.serving.schema import MODEL_FEATURES, TARGET_LABELS

logger = logging.getLogger("credit_scoring.predictions")
if not logger.handlers:
    # Domyślny handler konsolowy, gdyby aplikacja (Streamlit/FastAPI/uvicorn)
    # nie skonfigurowała własnego logowania.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

EXPECTED_FEATURE_SET = set(MODEL_FEATURES)


class PredictionLogger:
    """Loguje predykcje i co `verify_every` z nich uruchamia weryfikację."""

    def __init__(
        self,
        log_dir: Path | str,
        verify_every: int = 10,
        low_confidence_threshold: float = 0.4,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_path = self.log_dir / "predictions_log.jsonl"
        self.verification_path = self.log_dir / "verification_log.jsonl"

        self.verify_every = verify_every
        self.low_confidence_threshold = low_confidence_threshold

        self._lock = threading.Lock()
        # Odtwarzamy licznik z liczby już zapisanych predykcji, by przetrwał restart.
        self._counter = self._count_existing_lines(self.predictions_path)

    @staticmethod
    def _count_existing_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    @property
    def total_predictions(self) -> int:
        with self._lock:
            return self._counter

    def log_prediction(
        self,
        features: dict[str, Any],
        predicted_class: int,
        probabilities: dict[str, float] | None = None,
        source: str = "unknown",
        model_version: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Zapisuje jedną predykcję i — co `verify_every` predykcji — weryfikuje partię.

        Zwraca zapisany rekord (zawiera m.in. numer predykcji `prediction_index`
        oraz, jeśli akurat wypadła weryfikacja, pole `verification`).
        """
        confidence = max(probabilities.values()) if probabilities else None

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "source": source,
            "model_version": model_version,
            "features": features,
            "predicted_class": int(predicted_class),
            "predicted_label": TARGET_LABELS.get(int(predicted_class), str(predicted_class)),
            "probabilities": probabilities,
            "confidence": confidence,
        }

        with self._lock:
            self._counter += 1
            record["prediction_index"] = self._counter
            self._append_jsonl(self.predictions_path, record)
            logger.info(
                "Predykcja #%d (source=%s): klasa=%s, ufność=%s",
                record["prediction_index"],
                source,
                record["predicted_label"],
                f"{confidence:.3f}" if confidence is not None else "n/a",
            )

            verification_result = None
            if self._counter % self.verify_every == 0:
                batch = self._read_last_n_records(self.predictions_path, self.verify_every)
                verification_result = self._verify_batch(batch)
                self._append_jsonl(self.verification_path, verification_result)
                self._log_verification(verification_result)

        if verification_result is not None:
            record["verification"] = verification_result
        return record

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_last_n_records(path: Path, n: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        tail = lines[-n:] if len(lines) >= n else lines
        return [json.loads(line) for line in tail]

    def _verify_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Weryfikuje integralność i jakość ostatnich `verify_every` predykcji."""
        issues: list[str] = []

        # 1) Zgodność wektora cech ze schematem modelu (wykrywa np. rozjazd
        #    pipeline'u przetwarzania danych względem modelu).
        for rec in batch:
            feature_keys = set(rec.get("features", {}).keys())
            if feature_keys and feature_keys != EXPECTED_FEATURE_SET:
                missing = EXPECTED_FEATURE_SET - feature_keys
                extra = feature_keys - EXPECTED_FEATURE_SET
                issues.append(
                    f"prediction_index={rec.get('prediction_index')}: "
                    f"niezgodny schemat cech (brakuje={sorted(missing)[:5]}, "
                    f"nadmiarowe={sorted(extra)[:5]})"
                )

        # 2) NaN / Inf w cechach wejściowych.
        for rec in batch:
            for name, value in rec.get("features", {}).items():
                if isinstance(value, (int, float)) and (
                    math.isnan(value) or math.isinf(value)
                ):
                    issues.append(
                        f"prediction_index={rec.get('prediction_index')}: "
                        f"cecha '{name}' ma wartość NaN/Inf"
                    )

        # 3) Prawdopodobieństwa sumują się do ~1.0.
        for rec in batch:
            proba = rec.get("probabilities")
            if proba:
                total = sum(proba.values())
                if not math.isclose(total, 1.0, abs_tol=1e-2):
                    issues.append(
                        f"prediction_index={rec.get('prediction_index')}: "
                        f"prawdopodobieństwa sumują się do {total:.4f}, a nie 1.0"
                    )

        # 4) Średnia ufność modelu w partii — niska ufność może sygnalizować
        #    dryf danych względem zbioru treningowego.
        confidences = [rec["confidence"] for rec in batch if rec.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        low_confidence_count = sum(
            1 for c in confidences if c < self.low_confidence_threshold
        )
        if avg_confidence is not None and avg_confidence < self.low_confidence_threshold:
            issues.append(
                f"średnia ufność w partii ({avg_confidence:.3f}) jest niższa "
                f"od progu {self.low_confidence_threshold}"
            )

        # 5) Rozkład przewidywanych klas w partii (informacyjnie + heurystyka
        #    wykrywająca podejrzaną jednorodność predykcji).
        class_counter = Counter(rec.get("predicted_label") for rec in batch)
        if len(batch) >= self.verify_every and len(class_counter) == 1:
            issues.append(
                "wszystkie predykcje w partii należą do tej samej klasy — "
                "sprawdź, czy dane wejściowe się nie powtarzają"
            )

        last_index = batch[-1]["prediction_index"] if batch else None
        status = "WARNING" if issues else "OK"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_end_index": last_index,
            "n_checked": len(batch),
            "status": status,
            "avg_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
            "class_distribution": dict(class_counter),
            "issues": issues,
        }

    @staticmethod
    def _log_verification(result: dict[str, Any]) -> None:
        level = logging.WARNING if result["status"] == "WARNING" else logging.INFO
        logger.log(
            level,
            "Weryfikacja partii kończącej się na predykcji #%s: status=%s, "
            "śr. ufność=%s, rozkład klas=%s%s",
            result["batch_end_index"],
            result["status"],
            f"{result['avg_confidence']:.3f}" if result["avg_confidence"] is not None else "n/a",
            result["class_distribution"],
            f", problemy: {result['issues']}" if result["issues"] else "",
        )

    def stats(self) -> dict[str, Any]:
        """Zwraca podsumowanie stanu logowania (do np. endpointu /predictions/stats)."""
        with self._lock:
            total = self._counter
        last_verification = None
        if self.verification_path.exists():
            with open(self.verification_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if lines:
                last_verification = json.loads(lines[-1])
        return {
            "total_predictions": total,
            "verify_every": self.verify_every,
            "predictions_left_to_next_verification": (
                self.verify_every - (total % self.verify_every)
            )
            % self.verify_every
            or self.verify_every,
            "last_verification": last_verification,
            "predictions_log_path": str(self.predictions_path),
            "verification_log_path": str(self.verification_path),
        }


# Współdzielona instancja loggera — używana zarówno przez `app.py` (Streamlit),
# jak i przez `credit_scoring.serving.api` (FastAPI), dzięki czemu licznik
# predykcji jest spójny niezależnie od tego, skąd predykcja przyszła
# (o ile działają w tym samym procesie / tej samej maszynie — zob. docstring modułu).
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "data" / "09_predictions"
prediction_logger = PredictionLogger(DEFAULT_LOG_DIR, verify_every=10)
