# Credit Scoring - projekt ASI

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Projekt dotyczy klasyfikacji oceny kredytowej klienta na podstawie danych finansowych i behawioralnych. Zmienną docelową jest `Credit_Score`, która przyjmuje wartości:

- `Poor`
- `Standard`
- `Good`

Projekt został rozpoczęty w notebooku eksploracyjnym, a następnie przeniesiony do struktury Kedro, aby preprocessing, trenowanie i ewaluacja mogły być uruchamiane jako powtarzalne pipeline'y.

## Aktualny zakres projektu

Na obecnym etapie zaimplementowano:

- strukturę projektu Kedro,
- konfigurację Git LFS dla danych i modeli,
- pipeline przetwarzania danych `data_processing`,
- pipeline modelu bazowego `modeling`,
- pipeline AutoML `automl`,
- zapis metryk modelu bazowego,
- zapis metryk i leaderboardu modeli AutoGluon,
- wizualizację pipeline'ów przez `kedro viz`.

## Struktura katalogów

Najważniejsze katalogi projektu:

```text
conf/base/catalog.yml                  Konfiguracja zbiorów danych Kedro
data/01_raw/credit_score.csv           Dane surowe
data/02_intermediate/                  Dane po czyszczeniu
data/05_model_input/                   Dane gotowe do modelowania
data/06_models/                        Modele zapisane przez pipeline'y
data/08_reporting/                     Metryki i raporty
src/credit_scoring/pipelines/          Kod pipeline'ów Kedro
```

## Pipeline'y Kedro

Projekt ma trzy główne pipeline'y.

### `data_processing`

Pipeline odpowiada za przygotowanie danych:

```text
credit_score_raw -> clean_credit_data -> credit_score_clean
credit_score_clean -> prepare_model_input -> model_input
```

W preprocessingu wykonywane są m.in.:

- usunięcie kolumn identyfikacyjnych,
- czyszczenie wartości numerycznych zapisanych jako tekst,
- obsługa wartości odstających,
- uzupełnianie braków na podstawie `Customer_ID`,
- kodowanie wybranych zmiennych kategorycznych,
- mapowanie `Credit_Score` na wartości liczbowe.

### `modeling`

Pipeline trenuje model bazowy Random Forest:

```text
model_input -> train_baseline_model -> baseline_model, metrics
```

Wyniki zapisywane są w:

```text
data/08_reporting/metrics.json
```

Model bazowy zapisywany jest w:

```text
data/06_models/baseline_random_forest.pkl
```

### `automl`

Pipeline trenuje modele AutoGluon:

```text
model_input -> train_autogluon_model -> automl_metrics, automl_leaderboard
```

Wyniki zapisywane są w:

```text
data/08_reporting/automl_metrics.json
data/08_reporting/automl_leaderboard.csv
```

Modele AutoGluon zapisywane są lokalnie w:

```text
data/06_models/autogluon/
```

Folder z modelami AutoGluon może być bardzo duży, dlatego nie powinien być dodawany przypadkowo przez `git add .`. Jeżeli model ma zostać zapisany w repozytorium, należy zrobić to świadomie przez Git LFS.
Aktualnie zdecydowano o nie zapisywaniu go w rezypozytorium ze względu na duży rozmiar łączny (powyżej 3GB). Proces trenowania zajął 54 minuty.

## Instalacja zależności

```powershell
python -m pip install -r requirements.txt
```

## Git LFS

Projekt używa Git LFS dla większych plików danych i modeli.
Po sklonowaniu repozytorium należy wykonać:

```powershell
git lfs install
git lfs pull
```

## Uruchamianie projektu

Uruchomienie wszystkich pipeline'ów:

```powershell
kedro run
```

Uruchomienie samego preprocessingu:

```powershell
kedro run --pipelines data_processing
```

Uruchomienie modelu bazowego:

```powershell
kedro run --pipelines modeling
```

Uruchomienie AutoML:

```powershell
kedro run --pipelines automl
```

Wizualizacja pipeline'ów:

```powershell
kedro viz
```

## Aktualne wyniki

Model bazowy Random Forest zapisuje metryki do:

```text
data/08_reporting/metrics.json
```

AutoGluon zapisuje podsumowanie najlepszego modelu oraz leaderboard 5-ciu modeli do:

```text
data/08_reporting/automl_metrics.json
data/08_reporting/automl_leaderboard.csv
```

## GitHub Actions / automatyzacja MLOps

Projekt rozdziela szybkie sprawdzanie kodu, cięższe zadania ML oraz przyszłe
wdrożenie na osobne workflowy w `.github/workflows/`:

- `ci.yml` — szybkie CI dla Pull Requestów oraz pushy do `main`/`master`:
  linting, kontrola formatowania, testy i budowanie pakietu. Nie uruchamia
  treningu ani AutoML.
- `integration.yml` — diagnostyczna walidacja środowiska i pipeline'ów Kedro
  bez pobierania dużych danych. Pełny test integracyjny będzie wymagał małego
  pliku fixture CSV.
- `security.yml` — analiza bezpieczeństwa kodu przez CodeQL oraz kontrola zmian
  zależności w Pull Requestach.
- `train.yml` — Continuous Training uruchamiany ręcznie lub raz w tygodniu.
  Domyślnie trenuje model bazowy, natomiast AutoML wymaga ręcznego wyboru.
  Wygenerowane metryki i model bazowy są publikowane jako tymczasowe artifacts;
  workflow nie promuje modelu automatycznie.
- `promote-model.yml` — ręczna bramka promocji `candidate` → `production`.
  Obecnie jest to placeholder i checklista wymagań, ponieważ MLflow Model
  Registry nie jest jeszcze skonfigurowany.
- `deploy.yml` — kontrola gotowości do CD/deploymentu. Obecnie jest to
  placeholder, ponieważ projekt nie ma jeszcze Dockerfile, hostingu ani
  skonfigurowanej infrastruktury wdrożeniowej.

Workflow z obsługą `workflow_dispatch` można uruchomić w GitHubie przez:

```text
Actions → wybierz workflow → Run workflow
```

Workflowy nie wykonują commitów ani pushy i nie modyfikują gałęzi repozytorium.
Ciężki trening jest odseparowany od CI, ponieważ wymaga danych z Git LFS,
więcej czasu i pamięci oraz tworzy duże modele. Z tego samego powodu AutoML nie
jest uruchamiany przy każdym Pull Requeście i pozostaje ręczną opcją treningu.

## Serwowanie modelu

Kod współdzielony między interfejsami serwującymi (Streamlit i FastAPI) znajduje
się w `src/credit_scoring/serving/`:

```text
src/credit_scoring/serving/schema.py             Definicje cech i słowniki kodowania
src/credit_scoring/serving/inference.py          Wczytywanie modelu, budowa wektora cech, predykcja
src/credit_scoring/serving/prediction_logger.py  Logowanie predykcji + weryfikacja co 10. predykcję
src/credit_scoring/serving/api.py                Aplikacja FastAPI
```

### Aplikacja Streamlit

```powershell
streamlit run app.py
```

### API FastAPI

```powershell
uvicorn credit_scoring.serving.api:app --reload --app-dir src
```

Dokumentacja interaktywna (Swagger UI): http://127.0.0.1:8000/docs

Najważniejsze endpointy:

| Metoda | Endpoint               | Opis                                              |
|--------|------------------------|---------------------------------------------------|
| GET    | `/health`              | Status API i informacja, czy model jest wczytany  |
| GET    | `/model/metrics`       | Zawartość `data/08_reporting/metrics.json`        |
| POST   | `/predict`             | Predykcja dla jednego klienta                     |
| POST   | `/predict/batch`       | Predykcja dla listy klientów                      |
| GET    | `/predictions/stats`   | Licznik predykcji i wynik ostatniej weryfikacji   |

Przykład żądania `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 35, "occupation": "Engineer", "annual_income": 50000,
        "monthly_salary": 4000, "monthly_balance": 300, "amount_invested": 100,
        "num_bank_accounts": 4, "num_credit_card": 4, "num_of_loan": 2,
        "interest_rate": 12, "outstanding_debt": 1200, "credit_utilization": 32,
        "total_emi": 100, "credit_mix": "Standard", "credit_history_age_months": 120,
        "delay_from_due_date": 10, "num_delayed_payment": 5, "changed_credit_limit": 8,
        "num_credit_inquiries": 4, "payment_min": "No", "loan_types": ["Personal Loan"],
        "payment_behaviour": "Low_spent_Small_value_payments"
      }'
```

### Logowanie predykcji i weryfikacja co 10. predykcję

Każda predykcja — niezależnie czy wykonana ze Streamlit, czy z FastAPI —
jest logowana przez `PredictionLogger` do:

```text
data/09_predictions/predictions_log.jsonl     jedna linia JSON na predykcję
data/09_predictions/verification_log.jsonl    jedna linia JSON co 10. predykcję
```

Co 10. predykcję (`verify_every=10`) automatycznie wykonywana jest weryfikacja
ostatniej partii: sprawdzenie zgodności wektora cech ze schematem modelu, brak
wartości NaN/Inf, sumowanie się prawdopodobieństw do 1.0, średnia ufność modelu
oraz rozkład przewidzianych klas. Wynik (`OK`/`WARNING`) jest zapisywany do
`verification_log.jsonl` oraz logowany przez standardowy moduł `logging`
(logger `credit_scoring.predictions`). Licznik predykcji jest odtwarzany z
liczby linii już zapisanych w pliku logu, dzięki czemu przetrwa restart
aplikacji. Pliki `*.jsonl` w `data/09_predictions/` nie są commitowane do
repozytorium.

## Do zrobienia

Kolejne etapy projektu:

- dodać MLflow do śledzenia eksperymentów,
- przygotować aplikację Streamlit do predykcji,
- dodać logowanie predykcji jako prosty monitoring,
- uzupełnić dokumentację architektury,

