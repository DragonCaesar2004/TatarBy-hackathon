# TatBy - Татарский Чат-бот

TatBy - это чат-бот для общения и изучения татарского языка, использующий GigaChat API.

## Возможности

- 💬 Диалог на татарском языке
- 📚 Режим изучения языка с объяснениями на русском
- 🔄 Сохранение контекста диалога
- 🎯 Два режима работы:
  - dialog: общение на татарском языке
  - studying: обучение с объяснениями грамматики

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/tatby.git
cd tatby
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Создайте файл `.env` с вашими настройками:
```env
GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
```

4. Запустите сервер:
```bash
python -m uvicorn main:app --reload
```

## API Endpoints

### POST /chat
Основной эндпоинт для общения с чат-ботом.

Пример запроса:
```json
{
    "message_tat": "Исәнмесез!",
    "scenario": "dialog"
}
```

Пример ответа:
```json
{
    "input_tat": "Исәнмесез!",
    "translated_to_ru": "Прямое общение с моделью",
    "model_answer_ru": "Ответ модели в зависимости от сценария",
    "translated_back_to_tat": "Исәнмесез! Хәлләрегез ничек?"
}
```

### POST /clear-history
Очистить историю диалога.

## Running with Docker

Build and run the application with Docker:

1. Build the image:

  docker build -t tatby:latest .

2. Run with docker-compose (recommended for development):

  docker compose up --build

The app will be available at http://localhost:8000

### GET /health
Проверка работоспособности сервиса.

## Лицензия

MIT
