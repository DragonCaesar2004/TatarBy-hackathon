from typing import Literal, Optional
import httpx
import json
from fastapi import FastAPI, HTTPException, Response
from fastapi import UploadFile, File
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from gigachat_token_manager import get_token
from chat_history import chat_history

# ------------ Settings ------------
class Settings(BaseSettings):
    TATSOFT_BASE_URL: str = "https://v2.api.translate.tatar"
    GIGACHAT_BASE_URL: str = "https://gigachat.devices.sberbank.ru/api/v1"
    GIGACHAT_MODEL: str = "GigaChat"
    GIGACHAT_AUDIO_URL: Optional[str] = "https://gigachat.devices.sberbank.ru/api/v1/files"
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
    input_tat: Optional[str] = None
    translated_to_ru: str
    model_answer_ru: str
    audio_base64: str  # аудио в формате base64
    recognized_tat: Optional[str] = None

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


async def gigachat_audio_complete(wav_bytes: bytes, system_ru: Optional[str] = None) -> tuple:
    """Send WAV bytes to GigaChat audio endpoint and return (json_dict_or_None, assistant_or_None).

    Returns:
      (parsed_json, assistant_text) if Content-Type is JSON (parsed_json may be any JSON),
      (None, assistant_text) if response is plain text.
    """
    if not settings.GIGACHAT_AUDIO_URL:
        raise HTTPException(500, "GigaChat audio URL is not configured (GIGACHAT_AUDIO_URL)")

    bearer = get_token()
    headers = {"Authorization": bearer} if bearer else {}

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {}
    if system_ru:
        data["system_prompt"] = system_ru

    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS), verify=False) as client:
        r = await client.post(settings.GIGACHAT_AUDIO_URL, headers=headers, data=data, files=files)

    if r.status_code != 200:
        raise HTTPException(502, f"GigaChat audio error: {r.status_code} {r.text}")

    ctype = r.headers.get("Content-Type", "")
    if "json" in ctype:
        try:
            j = r.json()
        except Exception:
            raise HTTPException(502, f"GigaChat audio returned invalid JSON: {r.text}")

        assistant = None
        if isinstance(j, dict) and isinstance(j.get("assistant"), str):
            assistant = j.get("assistant")
        return j, assistant

    assistant = r.text.strip()
    return None, assistant

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

    # Получаем аудио от TatSoft API
    tts_url = "https://tat-tts.api.translate.tatar/listening/"
    params = {
        "speaker": "alsu",  # можно сделать параметром если нужно
        "text": response
    }
    
    async with httpx.AsyncClient(verify=False) as client:
        audio_response = await client.get(tts_url, params=params)
        if audio_response.status_code != 200:
            raise HTTPException(502, f"TTS API error: {audio_response.text}")
        
        # Получаем аудио в base64 напрямую из содержимого ответа
        audio_base64 = audio_response.text  # API возвращает base64 строку напрямую

    return ChatOut(
        input_tat=incoming.message_tat,
        translated_to_ru="Прямое общение с моделью",
        model_answer_ru="Ответ модели в зависимости от сценария",
        audio_base64=audio_base64  # base64 строка из ответа API
    )


@app.post("/chat-audio", response_model=ChatOut)
async def chat_audio(file: UploadFile = File(...), scenario: Optional[Literal["studying", "dialog"]] = "dialog", system_prompt_ru: Optional[str] = None):
    """
    Accept a multipart WAV file upload (raw wav) from frontend, send it to STT service,
    then forward recognized text to GigaChat. For `dialog` scenario a constant system
    prompt will be prepended so GigaChat answers in Tatar.
    """
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(400, "Only WAV audio files are supported (Content-Type audio/wav)")

    body = await file.read()

    # Build system prompt depending on scenario
    base_system_prompt = ""
    if scenario == "dialog":
        base_system_prompt = """
Ты - помощник, свободно владеющий татарским языком. 
Твоя задача - вести диалог на татарском языке, отвечая на вопросы пользователя.
Отвечай ТОЛЬКО на татарском языке, используя современную орфографию.
Будь дружелюбным и полезным собеседником.
Продолжай диалог и не присылай спецсимволы в ответ, так как потом он будет озвучиваться.
        """
    else:
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

    system_prompt = base_system_prompt
    if system_prompt_ru:
        system_prompt = f"{base_system_prompt}\n\nДополнительные инструкции:\n{system_prompt_ru}"

    # Send audio directly to GigaChat audio endpoint (or audio proxy)
    recognized, assistant_from_audio = await gigachat_audio_complete(body, system_prompt)

    # If assistant reply came back in the audio endpoint response, use it.
    if assistant_from_audio:
        model_response = assistant_from_audio
        # If recognized text present, save as user message; otherwise nobody
        if recognized:
            chat_history.add_message("user", recognized)
        chat_history.add_message("assistant", model_response)
    else:
        # No assistant reply from audio endpoint: use recognized text to call gigachat as before
        if not recognized:
            raise HTTPException(502, "Audio endpoint did not return assistant text or recognized text")
        chat_history.add_message("user", recognized)
        model_response = await gigachat_complete(recognized, system_prompt)
        chat_history.add_message("assistant", model_response)

    # Get TTS audio for model response
    tts_url = "https://tat-tts.api.translate.tatar/listening/"
    params = {"speaker": "alsu", "text": model_response}
    async with httpx.AsyncClient(verify=False) as client:
        audio_response = await client.get(tts_url, params=params)
        if audio_response.status_code != 200:
            raise HTTPException(502, f"TTS API error: {audio_response.text}")
        audio_base64 = audio_response.text

    return ChatOut(
        input_tat=recognized,
        translated_to_ru="Прямое общение с моделью",
        model_answer_ru=model_response,
        audio_base64=audio_base64,
        recognized_tat=recognized
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
