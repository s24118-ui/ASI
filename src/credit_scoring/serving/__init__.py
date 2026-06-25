"""Warstwa serwowania modelu

Ten pakiet zawiera kod współdzielony między interfejsem Streamlit (``app.py``)
oraz API REST (``credit_scoring.serving.api``):

- :mod:`credit_scoring.serving.schema`            — definicje cech i słowniki kodowania,
- :mod:`credit_scoring.serving.inference`         — wczytywanie modelu i predykcja,
- :mod:`credit_scoring.serving.prediction_logger` — logowanie predykcji
  oraz weryfikacja co N-tą predykcję,
- :mod:`credit_scoring.serving.api`               — aplikacja FastAPI.
"""
