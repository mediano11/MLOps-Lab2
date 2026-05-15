# Iris ML API — Лабораторна робота 2 & 3 (MLOps)

[![CI](https://github.com/mediano11/MLOps-Lab2/actions/workflows/ci.yml/badge.svg)](https://github.com/mediano11/MLOps-Lab2/actions/workflows/ci.yml)

**Автор:** Пестенков Дмітрій
**Дата:** 2026-05-15

## Опис системи

REST API для класифікації квіток Iris на основі навченої моделі логістичної регресії (scikit-learn).  
Сервіс реалізований на FastAPI, розширений шаром Prometheus-метрик, KS-детектором data drift та структурованим JSON-логуванням. Контейнеризований у Docker, розгорнутий на Render.

**ЛР2:** ML API + Docker + CI/CD (GitHub Actions) + Render  
**ЛР3:** Prometheus-метрики + Drift Detection (KS-тест) + Structured Logging + Evidently

## Стек технологій

| Компонент       | Технологія                        |
| --------------- | --------------------------------- |
| ML-модель       | scikit-learn (LogisticRegression) |
| API             | FastAPI + Uvicorn                 |
| Валідація       | Pydantic v2                       |
| Тести           | pytest + httpx                    |
| Метрики         | prometheus-client                 |
| Drift Detection | scipy (KS-тест)                   |
| Логування       | python-json-logger (JSON)         |
| Звіт drift      | Evidently                         |
| Контейнер       | Docker (python:3.11-slim)         |
| CI/CD           | GitHub Actions                    |
| Деплой          | Render (Docker Web Service)       |

## Структура репозиторію

```
ml-api-lab/
├── .github/workflows/ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI + middleware + всі ендпоінти
│   ├── schemas.py          # Pydantic-моделі (вхід/вихід + drift)
│   ├── metrics.py          # Prometheus Counter / Histogram / Gauge
│   ├── drift.py            # DriftDetector (KS-тест)
│   └── logging_config.py   # Structured JSON logging
├── ml/
│   ├── __init__.py
│   └── train.py            # Тренування + збереження reference_stats.joblib
├── monitoring/
│   ├── prometheus.yml                  # Конфіг Prometheus
│   └── docker-compose.monitoring.yml  # ML API + Prometheus разом
├── scripts/
│   └── evidently_report.py  # Генерація HTML drift-звіту
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Інтеграційні тести API (ЛР2)
│   ├── test_model.py        # Unit-тести моделі (ЛР2)
│   ├── test_metrics.py      # Тести Prometheus-метрик (ЛР3)
│   └── test_drift.py        # Тести детектора drift (ЛР3)
├── assets/
│   ├── health.png
│   └── predict.png
├── model.joblib             # Артефакт моделі
├── reference_stats.joblib   # Reference-вибірка для drift detection
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Як запустити локально

```bash
git clone git@github.com:mediano11/MLOps-Lab2.git
cd MLOps-Lab2

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

# Навчити модель (створює model.joblib і reference_stats.joblib)
python -m ml.train

# Запустити API
uvicorn app.main:app --reload
```

Відкрити: [http://localhost:8000/docs](http://localhost:8000/docs)

## Запуск через Docker

```bash
docker build -t ml-api:lab .
docker run --rm -p 8000:8000 ml-api:lab
```

## Як запустити моніторинг (ML API + Prometheus)

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
```

| URL                           | Опис                         |
| ----------------------------- | ---------------------------- |
| http://localhost:8000/docs    | Swagger UI                   |
| http://localhost:8000/metrics | Prometheus exposition format |
| http://localhost:9090/targets | Статус scrape-таргетів       |
| http://localhost:9090/graph   | PromQL запити та графіки     |

**Корисні PromQL-запити:**

```promql
# Швидкість прогнозів за секунду
rate(ml_predictions_total[1m])

# 95-й перцентиль latency
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))

# Розподіл прогнозів за класами
sum by (class_name) (ml_predictions_total)

# Кількість виявлених drift-подій
ml_drift_detected_total
```

## Як запустити тести

```bash
pytest -v
```

Очікуваний результат: **15 passed**.

| Тест                                   | Що перевіряє                              |
| -------------------------------------- | ----------------------------------------- |
| `test_train_creates_model_file`        | model.joblib створюється, accuracy > 0.8  |
| `test_model_predicts_three_classes`    | Модель повертає клас 0/1/2                |
| `test_root_endpoint`                   | GET `/` → 200 OK                          |
| `test_health_endpoint`                 | GET `/health` → model_loaded: true        |
| `test_predict_setosa`                  | POST `/predict` класифікує setosa         |
| `test_predict_invalid_input`           | Pydantic → 422 на невірний тип            |
| `test_metrics_endpoint_available`      | GET `/metrics` → 200, містить метрики     |
| `test_predict_increments_counter`      | Лічильник зростає після /predict          |
| `test_health_has_drift_detector`       | drift_detector_ready: true                |
| `test_no_drift_on_same_distribution`   | KS-тест: однакові розподіли → no drift    |
| `test_drift_on_shifted_distribution`   | KS-тест: зміщені дані → drift detected    |
| `test_detector_validates_dimensions`   | Помилка на некоректний shape              |
| `test_detector_raises_on_1d_reference` | Помилка на 1D reference                   |
| `test_check_drift_endpoint`            | POST `/check-drift` → 200, n_samples=10   |
| `test_check_drift_detects_anomaly`     | Аномальні значення → drift_detected: true |

## Як працює API

### Ендпоінти

| Метод | URL            | Опис                          |
| ----- | -------------- | ----------------------------- |
| GET   | `/`            | Статус сервісу                |
| GET   | `/health`      | Стан моделі та drift detector |
| POST  | `/predict`     | Класифікація квітки Iris      |
| GET   | `/metrics`     | Prometheus метрики            |
| POST  | `/check-drift` | KS-тест на data drift         |
| GET   | `/docs`        | Swagger UI                    |

### Приклад `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

```json
{ "class_id": 0, "class_name": "setosa", "probability": 0.9823 }
```

### Приклад `/check-drift`

```bash
curl -X POST "http://localhost:8000/check-drift" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [9.0,8.0,8.0,5.0],[9.5,7.5,8.5,5.5],[8.5,8.5,7.5,4.5],
      [9.2,8.2,8.2,5.2],[9.8,7.8,8.8,5.8],[8.8,8.8,7.8,4.8],
      [9.4,8.4,8.4,5.4],[9.6,7.6,8.6,5.6],[8.6,8.6,7.6,4.6],
      [9.1,8.1,8.1,5.1]
    ],
    "alpha": 0.05
  }'
```

```json
{
  "drift_detected": true,
  "n_drifted_features": 4,
  "drifted_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "per_feature": {
    "sepal_length": {"statistic": 1.0, "p_value": 0.0, "drift_detected": true},
    ...
  },
  "n_samples": 10,
  "alpha": 0.05
}
```

## Реалізовані метрики Prometheus

| Метрика                         | Тип       | Опис                                            |
| ------------------------------- | --------- | ----------------------------------------------- |
| `ml_predictions_total`          | Counter   | Кількість прогнозів (мітки: class_name, status) |
| `ml_prediction_latency_seconds` | Histogram | Час інференсу (бакети: 5ms–5s)                  |
| `ml_prediction_confidence`      | Histogram | Розподіл predict_proba                          |
| `ml_errors_total`               | Counter   | Помилки (мітка: error_type)                     |
| `ml_model_loaded`               | Gauge     | 1 якщо модель завантажена, 0 — ні               |
| `ml_drift_checks_total`         | Counter   | Кількість викликів /check-drift                 |
| `ml_drift_detected_total`       | Counter   | Drift-події по ознаках (мітка: feature)         |

## Drift Detection

Використовується двовибірковий **KS-тест** (Kolmogorov-Smirnov):

- **Reference**: тренувальна вибірка X_train (зберігається у `reference_stats.joblib`)
- **Current**: батч live-даних (мінімум 10 спостережень), надісланий у `/check-drift`
- **Рішення**: якщо `p_value < alpha` (за замовчуванням 0.05) — drift виявлено
- Тест виконується **окремо для кожної з 4 ознак**

### Evidently HTML-звіт (bonus)

```bash
python scripts/evidently_report.py
# Відкрити drift_report.html у браузері
```

## Логування

Усі ключові події логуються у форматі JSON у stdout:

```json
{
  "timestamp": "2026-05-16T00:00:00Z",
  "level": "INFO",
  "logger": "ml-api",
  "message": "prediction",
  "event": "prediction",
  "class_name": "setosa",
  "probability": 0.9823,
  "class_id": 0
}
```

| Подія              | Рівень | Опис                                              |
| ------------------ | ------ | ------------------------------------------------- |
| `startup_complete` | INFO   | Сервіс стартував, модель та drift detector готові |
| `prediction`       | INFO   | Кожен успішний прогноз                            |
| `drift_check`      | INFO   | Результат перевірки на drift                      |
| `inference_failed` | ERROR  | Помилка інференсу                                 |

## Демонстрація роботи

### GET `/health` — перевірка стану сервісу

![health endpoint](assets/health.png)

### POST `/predict` — класифікація квітки Iris

![predict endpoint](assets/predict.png)

## Посилання на деплой

Сервіс розгорнутий на Render: **https://mlops-lab2-2ozc.onrender.com**

```bash
curl https://mlops-lab2-2ozc.onrender.com/health
```

## Висновки

У ЛР3 ML API з ЛР2 розширено повноцінним шаром спостережуваності:

1. **Prometheus-метрики** дозволяють у реальному часі відстежувати throughput, latency (p50/p95/p99), розподіл класів та рівень упевненості моделі.
2. **Drift detection** на основі KS-тесту виявляє статистично значущі зміни у вхідних даних без необхідності мати реальні мітки — це дозволяє реагувати на деградацію моделі завчасно.
3. **Structured JSON logging** перетворює логи на машинно-читабельний потік подій, придатний для індексації у ELK/Loki.
4. Усі компоненти покриті **15 автоматичними тестами**, що проходять у CI без змін у `ci.yml`.
