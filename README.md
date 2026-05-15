# Iris ML API — Лабораторна робота 2 (MLOps)

![CI](https://github.com/<YOUR_USERNAME>/<YOUR_REPO>/actions/workflows/ci.yml/badge.svg)

## Опис проєкту

REST API для класифікації квіток Iris на основі навченої моделі логістичної регресії (scikit-learn). Сервіс реалізований на FastAPI, контейнеризований у Docker та розгорнутий на платформі Render.

Проєкт демонструє повний MLOps-конвеєр:
- навчання та серіалізація ML-моделі
- REST API для real-time інференсу
- unit-тести (pytest)
- CI/CD через GitHub Actions
- контейнеризація (Docker)
- хмарне розгортання (Render)

## Стек технологій

| Компонент | Технологія |
|---|---|
| ML-модель | scikit-learn (LogisticRegression) |
| API | FastAPI + Uvicorn |
| Валідація | Pydantic v2 |
| Тести | pytest + httpx |
| Контейнер | Docker (python:3.11-slim) |
| CI/CD | GitHub Actions |
| Деплой | Render (Docker Web Service) |

## Структура репозиторію

```
ml-api-lab2/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions workflow
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI застосунок
│   └── schemas.py          # Pydantic-моделі вхід/вихід
├── ml/
│   ├── __init__.py
│   └── train.py            # Скрипт тренування моделі
├── tests/
│   ├── __init__.py
│   ├── test_api.py         # Інтеграційні тести API
│   └── test_model.py       # Unit-тести моделі
├── model.joblib            # Артефакт навченої моделі
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Як запустити локально

### 1. Клонувати репозиторій та перейти у директорію

```bash
git clone <URL_РЕПОЗИТОРІЮ>
cd ml-api-lab2
```

### 2. Створити та активувати venv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Встановити залежності

```bash
pip install -r requirements.txt
```

### 4. Навчити модель

```bash
python -m ml.train
```

Після цього у корені з'явиться файл `model.joblib`.

### 5. Запустити API

```bash
uvicorn app.main:app --reload
```

Відкрити у браузері: [http://localhost:8000/docs](http://localhost:8000/docs) — інтерактивна Swagger UI документація.

## Запуск через Docker

```bash
# Зібрати образ (модель навчається під час збірки)
docker build -t ml-api:lab2 .

# Запустити контейнер
docker run --rm -p 8000:8000 ml-api:lab2
```

Перевірити: [http://localhost:8000/health](http://localhost:8000/health)

## Як запустити тести

```bash
pytest -v
```

Очікуваний результат: **6 passed**.

| Тест | Що перевіряє |
|---|---|
| `test_train_creates_model_file` | Файл моделі створюється, accuracy > 0.8 |
| `test_model_predicts_three_classes` | Модель повертає клас 0/1/2 |
| `test_root_endpoint` | GET `/` повертає 200 OK |
| `test_health_endpoint` | GET `/health` підтверджує модель завантажена |
| `test_predict_setosa` | POST `/predict` коректно класифікує setosa |
| `test_predict_invalid_input` | Pydantic повертає 422 на неправильний тип |

## Як працює API

### Ендпоінти

| Метод | URL | Опис |
|---|---|---|
| GET | `/` | Перевірка сервісу |
| GET | `/health` | Статус + чи завантажена модель |
| POST | `/predict` | Класифікація квітки Iris |
| GET | `/docs` | Swagger UI документація |

### Приклад запиту до `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Приклад відповіді

```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probability": 0.9823
}
```

## Посилання на деплой

Сервіс розгорнутий на Render: **https://<YOUR_APP>.onrender.com**

Перевірка: `curl https://<YOUR_APP>.onrender.com/health`
