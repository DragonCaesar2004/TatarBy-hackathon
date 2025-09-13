from typing import Literal, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from gigachat_token_manager import get_token
from chat_history import chat_history

# ------------ Settings ------------
class Settings(BaseSettings):
    TATSOFT_BASE_URL: str = "https://v2.api.translate.tatar"
    GIGACHAT_BASE_URL: str = "https://gigachat.devices.sberbank.ru/api/v1"
    GIGACHAT_MODEL: str = "GigaChat"
    REQUEST_TIMEOUT_SECONDS: int = 15

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Общий HTTP-клиент (асинхронный)
client = httpx.AsyncClient(
    timeout=httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS),
    verify=False  # Отключение проверки SSL сертификатов
)

# ------------ Schemas ------------
class ChatIn(BaseModel):
    message_tat: str
    scenario: Optional[Literal["studying", "dialog"]] = "dialog"
    system_prompt_ru: Optional[str] = None

class ChatOut(BaseModel):
    input_tat: str
    translated_to_ru: str
    model_answer_ru: str
    translated_back_to_tat: str

# ------------ Helpers ------------
async def gigachat_complete(prompt_ru: str, system_ru: Optional[str]) -> str:
    """
    Запрос в GigaChat API: chat/completions
    """
    url = f"{settings.GIGACHAT_BASE_URL}/chat/completions"
    
    # Получаем текущий действующий токен из менеджера
    bearer = get_token()
    if not bearer:
        raise HTTPException(500, "Не удалось получить токен GigaChat")

    headers = {
        "Authorization": bearer,  # Токен уже содержит префикс Bearer
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # формируем список сообщений
    messages = []
    if system_ru:
        messages.append({"role": "system", "content": system_ru})
    
    # Добавляем историю диалога
    messages.extend(chat_history.get_messages())
    
    # Добавляем текущее сообщение
    messages.append({"role": "user", "content": prompt_ru})

    payload = {
        "model": settings.GIGACHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(http2=False, timeout=20.0, verify=False) as client:
        r = await client.post(url, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"GigaChat error: {r.text}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, f"GigaChat: неожиданный ответ {data}")

# ------------ FastAPI ------------
app = FastAPI(title="Tat↔Ru dialog proxy", version="1.0.0")

@app.post("/chat", response_model=ChatOut)
async def chat(incoming: ChatIn):
    # Определяем базовый системный промпт в зависимости от сценария
    base_system_prompt = ""
    if incoming.scenario == "dialog":
        base_system_prompt = """
Ты - помощник, свободно владеющий татарским языком. 
Твоя задача - вести диалог на татарском языке, отвечая на вопросы пользователя.
Отвечай ТОЛЬКО на татарском языке, используя современную орфографию.
Будь дружелюбным и полезным собеседником.
Продолжай диалог и не присылай спецсимволы в ответ, так как потом он будет озвучиваться.
        """
    else:  # studying
        base_system_prompt = """
Ты - преподаватель татарского языка.
Твоя задача - помогать в изучении татарского языка.
Когда пользователь пишет на татарском:
1. Дай перевод на русский
2. Объясни грамматические конструкции
3. Укажи на ошибки, если они есть
4. Предложи, как можно улучшить фразу
Отвечай на русском языке.
Не используй спецсимволы в ответе, так как потом он будет озвучиваться.
        """
    
    # Объединяем базовый системный промпт с пользовательским, если он есть
    system_prompt = base_system_prompt
    if incoming.system_prompt_ru:
        system_prompt = f"{base_system_prompt}\n\nДополнительные инструкции:\n{incoming.system_prompt_ru}"

    # Сохраняем сообщение пользователя в историю
    chat_history.add_message("user", incoming.message_tat)
    
    # Получаем ответ от модели с учетом истории диалога
    response = await gigachat_complete(incoming.message_tat, system_prompt)
    
    # Сохраняем ответ ассистента в историю
    chat_history.add_message("assistant", response)

    return ChatOut(
        input_tat=incoming.message_tat,
        translated_to_ru="Прямое общение с моделью",
        model_answer_ru="Ответ модели в зависимости от сценария",
        translated_back_to_tat=response
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/clear-history")
async def clear_chat_history():
    """Очистить историю диалога"""
    chat_history.clear()
    return {"status": "ok"}

# Корректно закрываем клиент при остановке
@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()
