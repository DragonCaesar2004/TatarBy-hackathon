from typing import Literal, Optional
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ------------ Settings ------------
class Settings(BaseSettings):
    TATSOFT_BASE_URL: str = "https://v2.api.translate.tatar"
    GIGACHAT_BASE_URL: str
    GIGACHAT_TOKEN: str
    GIGACHAT_MODEL: str = "gigachat"
    REQUEST_TIMEOUT_SECONDS: int = 15

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Общий HTTP-клиент (асинхронный)
# Отключаем проверку SSL для работы с GigaChat (может потребоваться в некоторых случаях)
client = httpx.AsyncClient(
    timeout=httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS),
    verify=False  # Отключение проверки SSL сертификатов
)

# ------------ Schemas ------------
class ChatIn(BaseModel):
    message_tat: str
    # опционально — прокинуть системные инструкции модели
    system_prompt_ru: Optional[str] = None

class ChatOut(BaseModel):
    input_tat: str
    translated_to_ru: str
    model_answer_ru: str
    translated_back_to_tat: str

# ------------ Helpers ------------


async def get_gigachat_access_token() -> str:
    """
    Получение access token через OAuth2 client credentials flow.
    """
    url = f"{settings.GIGACHAT_BASE_URL}/token"  # Стандартный endpoint для получения токенов
    headers = {
        "Authorization": f"Basic {settings.GIGACHAT_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": "unique-request-id",  # Некоторые API требуют уникальный ID запроса
    }
    data = {
        "grant_type": "client_credentials",  # Стандартный тип для client credentials flow
    }
    
    for attempt in range(3):
        try:
            r = await client.post(url, headers=headers, data=data)
            if r.status_code == 200:
                token_data = r.json()
                return token_data["access_token"]
            if 400 <= r.status_code < 500:
                raise HTTPException(r.status_code, f"GigaChat Token: {r.text}")
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            pass
        await asyncio.sleep(0.4 * (attempt + 1))
    
    raise HTTPException(502, "GigaChat Token service недоступен после повторных попыток")



async def gigachat_complete(prompt_ru: str, system_ru: Optional[str]) -> str:
    """
    Запрос в GigaChat API: chat/completions
    """
    url = f"{settings.GIGACHAT_BASE_URL}/chat/completions"

    # токен в .env должен быть без кавычек и пробелов
    bearer = (settings.GIGACHAT_TOKEN or "").strip(" '\"")
    if not bearer:
        raise HTTPException(500, "Нет GIGACHAT_TOKEN")

    headers = {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # формируем список сообщений
    messages = []
    if system_ru:
        messages.append({"role": "system", "content": system_ru})
    messages.append({"role": "user", "content": prompt_ru})

    payload = {
        "model": settings.GIGACHAT_MODEL or "GigaChat",  # лучше с большой буквы
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
    # Skip TatSoft translation and send Tatar text directly to GigaChat
    # 1) GigaChat (using original Tatar text)
    tat_answer = await gigachat_complete(incoming.message_tat, incoming.system_prompt_ru)

    return ChatOut(
        input_tat=incoming.message_tat,
        translated_to_ru="Пропущено - отправлено напрямую в GigaChat",
        model_answer_ru="Пропущено - ответ от GigaChat на татарском",
        translated_back_to_tat=tat_answer,
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# Корректно закрываем клиент при остановке
@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()
