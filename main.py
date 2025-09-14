from typing import Literal, Optional, Tuple, List, Dict, Any
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from gigachat_token_manager import get_token
from chat_history import chat_history


# =========================
# Settings
# =========================
class Settings(BaseSettings):
    TATSOFT_BASE_URL: str = "https://v2.api.translate.tatar"
    GIGACHAT_BASE_URL: str = "https://gigachat.devices.sberbank.ru/api/v1"
    # Всегда используем только эту модель
    GIGACHAT_MODEL: str = "GigaChat-Pro"
    REQUEST_TIMEOUT_SECONDS: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

ALLOWED_WAV_CT = ("audio/wav", "audio/x-wav", "audio/wave", "audio/x-pn-wav")


# =========================
# Schemas
# =========================
class ChatIn(BaseModel):
    message_tat: str
    scenario: Optional[Literal["studying", "dialog"]] = "dialog"
    system_prompt_ru: Optional[str] = None


class ChatOut(BaseModel):
    input_tat: Optional[str] = None
    translated_to_ru: str
    model_answer_ru: str
    audio_base64: str
    recognized_tat: Optional[str] = None


# =========================
# Helpers: auth & messages
# =========================
def _auth_header_value() -> str:
    """
    Возвращает корректное значение для заголовка Authorization.
    Поддерживает варианты, когда get_token() уже вернёт 'Bearer ...' или сырой токен.
    """
    tok = get_token()
    if not tok:
        raise HTTPException(500, "Не удалось получить токен GigaChat")
    return tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"


async def _client_json_headers() -> Dict[str, str]:
    return {
        "Authorization": _auth_header_value(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _client_multipart_headers() -> Dict[str, str]:
    # Content-Type не задаём — httpx сам проставит boundary
    return {
        "Authorization": _auth_header_value(),
        "Accept": "application/json",
    }


def _build_messages(system_ru: Optional[str],
                    history: List[Dict[str, Any]],
                    user_msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Собирает массив messages так, чтобы:
      - был РОВНО один system (если указан) и строго первым,
      - из истории удаляются любые system,
      - затем идёт история (user/assistant),
      - в конце — текущий user_msg.
    """
    msgs: List[Dict[str, Any]] = []
    if system_ru:
        msgs.append({"role": "system", "content": system_ru})
    msgs.extend(m for m in history if m.get("role") != "system")
    msgs.append(user_msg)
    return msgs


# =========================
# GigaChat low-level (только Pro)
# =========================
async def _gc_upload_file(filename: str, content: bytes, mime: str) -> str:
    """
    POST /files — загрузка файла (multipart). Возвращает file_id (str).
    """
    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_multipart_headers()

    files = {"file": (filename, content, mime)}
    data = {"purpose": "general"}

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        r = await client.post(f"{base}/files", headers=headers, files=files, data=data)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"GigaChat /files error: {r.text}")

    j = r.json()
    fid = j.get("id")
    if not isinstance(fid, str):
        data_obj = j.get("data", {})
        fid = data_obj.get("id") if isinstance(data_obj, dict) else None

    if not isinstance(fid, str):
        raise HTTPException(502, f"Неожиданный ответ /files: {j}")

    print(f"[DEBUG] /files uploaded id: {fid}")
    return fid


async def _gc_chat_with_pro(messages: List[Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Жёсткий вызов /chat/completions только на GigaChat-Pro.
    Возвращает: (ok, assistant_text, raw_error_text)
    """
    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_json_headers()

    # Извлекаем вложения из user-сообщений (ожидаем list[str] — file_id)
    attachment_ids: List[str] = []
    processed_messages: List[Dict[str, Any]] = []

    for message in messages:
        msg = dict(message)
        if msg.get("role") == "user" and "attachments" in msg:
            ids = [a for a in msg.get("attachments", []) if isinstance(a, str)]
            attachment_ids.extend(ids)
            msg.pop("attachments", None)
        processed_messages.append(msg)

    payload: Dict[str, Any] = {
        "model": settings.GIGACHAT_MODEL,   # "GigaChat-Pro"
        "messages": processed_messages,
        "temperature": 0.2,
    }
    if attachment_ids:
        payload["attachments"] = attachment_ids  # строго list[str]

    print(f"[DEBUG] Пробуем модель: {settings.GIGACHAT_MODEL}")
    print(f"[DEBUG] Payload(messages={len(processed_messages)}, attachments={len(attachment_ids)})")

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        r = await client.post(f"{base}/chat/completions", headers=headers, json=payload)

    print(f"[DEBUG] Ответ /chat/completions {settings.GIGACHAT_MODEL}: {r.status_code}, {r.text[:500]}")

    if r.status_code == 200:
        j = r.json()
        try:
            return True, j["choices"][0]["message"]["content"], None
        except Exception:
            return False, None, f"Unexpected JSON: {r.text}"

    return False, None, r.text


# =========================
# High-level
# =========================
async def gigachat_complete(prompt_ru: str, system_ru: Optional[str]) -> str:
    history = chat_history.get_messages()
    msgs = _build_messages(system_ru, history, {"role": "user", "content": prompt_ru})

    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_json_headers()
    payload = {
        "model": settings.GIGACHAT_MODEL,  # "GigaChat-Pro"
        "messages": msgs,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        r = await client.post(f"{base}/chat/completions", headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"GigaChat error: {r.text}")

    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, f"GigaChat: неожиданный ответ {j}")


async def gigachat_audio_complete(wav_bytes: bytes, system_ru: Optional[str]) -> Tuple[None, str]:
    """
    Аудио → ответ: /files -> /chat/completions + attachments.
    Всегда используем только GigaChat-Pro. Если Pro не примет аудио, вернём ошибку от API.
    """
    print(f"[DEBUG] === gigachat_audio_complete: Starting audio processing")
    print(f"[DEBUG] Audio bytes length: {len(wav_bytes)}")
    print(f"[DEBUG] System prompt: {system_ru}")

    file_id = await _gc_upload_file("audio.wav", wav_bytes, "audio/wav")
    print(f"[DEBUG] File uploaded with ID: {file_id}")

    history = chat_history.get_messages()
    msgs = _build_messages(
        system_ru,
        history,
        {
            "role": "user",
            "content": "Транскрибируй это аудио и ответь по инструкции.",
            # В message кладём ID файла, _gc_chat_with_pro перенесёт в корневой attachments
            "attachments": [file_id],
        },
    )

    print(f"[DEBUG] Messages to send (built): roles={[m.get('role') for m in msgs]}")
    ok, assistant_text, err = await _gc_chat_with_pro(msgs)
    if ok:
        print(f"[DEBUG] Success on {settings.GIGACHAT_MODEL}")
        return None, assistant_text

    # Пробросим исходный текст ошибки API
    print(f"[ERROR] {settings.GIGACHAT_MODEL} вернула ошибку: {err}")
    raise HTTPException(502, f"Не удалось получить ответ от {settings.GIGACHAT_MODEL}: {err or 'unknown error'}")


# =========================
# FastAPI
# =========================
app = FastAPI(title="Tat↔Ru dialog proxy", version="1.0.0")


@app.post("/chat", response_model=ChatOut)
async def chat(incoming: ChatIn):
    if incoming.scenario == "dialog":
        base_system_prompt = (
            "Ты - помощник, свободно владеющий татарским языком. "
            "Веди диалог ТОЛЬКО на татарском, современная орфография. "
            "Не используй спецсимволы в ответе."
        )
    else:
        base_system_prompt = (
            "Ты - преподаватель татарского языка. "
            "1) Переведи на русский. 2) Объясни грамматику. 3) Исправь ошибки. "
            "4) Предложи улучшение. Отвечай на русском. Без спецсимволов."
        )

    system_prompt = (
        base_system_prompt
        if not incoming.system_prompt_ru
        else base_system_prompt + "\n\nДополнительные инструкции:\n" + incoming.system_prompt_ru
    )

    chat_history.add_message("user", incoming.message_tat)
    response_text = await gigachat_complete(incoming.message_tat, system_prompt)
    chat_history.add_message("assistant", response_text)

    # TTS
    tts_url = "https://tat-tts.api.translate.tatar/listening/"
    params = {"speaker": "alsu", "text": response_text}
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        audio_response = await client.get(tts_url, params=params)
    if audio_response.status_code != 200:
        raise HTTPException(502, f"TTS API error: {audio_response.text}")
    audio_base64 = audio_response.text

    return ChatOut(
        input_tat=incoming.message_tat,
        translated_to_ru="Прямое общение с моделью",
        model_answer_ru=response_text,
        audio_base64=audio_base64,
    )


@app.post("/chat-audio", response_model=ChatOut)
async def chat_audio(
    file: UploadFile = File(...),
    scenario: Optional[Literal["studying", "dialog"]] = "dialog",
    system_prompt_ru: Optional[str] = None,
):
    print(f"[DEBUG] === /chat-audio endpoint called")
    print(f"[DEBUG] File content type: {file.content_type}")
    print(f"[DEBUG] Scenario: {scenario}")
    print(f"[DEBUG] System prompt: {system_prompt_ru}")

    if file.content_type not in ALLOWED_WAV_CT:
        raise HTTPException(400, "Требуется WAV (Content-Type audio/wav)")

    wav_bytes = await file.read()
    print(f"[DEBUG] Read {len(wav_bytes)} bytes from uploaded file")

    if scenario == "dialog":
        base_system_prompt = (
            "Ты - помощник, свободно владеющий татарским языком. "
            "Отвечай ТОЛЬКО на татарском, современная орфография. "
            "Не используй спецсимволы."
        )
    else:
        base_system_prompt = (
            "Ты - преподаватель татарского языка. "
            "1) Переведи на русский. 2) Объясни грамматику. 3) Исправь ошибки. "
            "4) Предложи улучшение. Отвечай на русском. Без спецсимволов."
        )

    system_prompt = (
        base_system_prompt
        if not system_prompt_ru
        else base_system_prompt + "\n\nДополнительные инструкции:\n" + system_prompt_ru
    )

    print(f"[DEBUG] Calling gigachat_audio_complete on {settings.GIGACHAT_MODEL}")
    _, assistant = await gigachat_audio_complete(wav_bytes, system_prompt)
    print(f"[DEBUG] Received response: {assistant[:200]}...")

    chat_history.add_message("user", "[voice]")
    chat_history.add_message("assistant", assistant)

    # TTS
    tts_url = "https://tat-tts.api.translate.tatar/listening/"
    params = {"speaker": "alsu", "text": assistant}
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        audio_response = await client.get(tts_url, params=params)
    if audio_response.status_code != 200:
        raise HTTPException(502, f"TTS API error: {audio_response.text}")
    audio_base64 = audio_response.text

    return ChatOut(
        input_tat=None,
        translated_to_ru="Прямое общение с моделью",
        model_answer_ru=assistant,
        audio_base64=audio_base64,
        recognized_tat=None,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/greet")
async def get_greeting(scenario: Optional[Literal["studying", "dialog"]] = "dialog") -> ChatOut:
    if scenario == "dialog":
        greetings = [
            "Сәлам! Синең хәлләр ничек?",
            "Сәлам! Син хәзер нәрсә эшлисең?",
            "Сәлам! Синең кәеф ничек?",
            "Сәлам! Минем исемем Алия, ә синең исемең ничек?",
            "ә синең исемең ничек?"
        ]
    else:  # studying
        greetings = [
            "Привет! Давай учить татарский язык! Готов?",
            "Здравствуйте! Я буду вашим преподавателем татарского языка. Начнем?",
            "Привет! Сегодня будем изучать интересную грамматику татарского языка!",
            "Добрый день! Начинаем урок татарского. Есть вопросы?",
            "Здравствуйте! Давайте учиться разговаривать на татарском!"
        ]
    import random
    greeting = random.choice(greetings)

    # Очищаем историю перед началом нового диалога
    chat_history.clear()

    # Кладём приветствие сразу как ответ ассистента.
    # ВАЖНО: НЕ добавляем system в историю — чтобы не дублировать в следующем запросе.
    chat_history.add_message("assistant", greeting)

    # TTS для приветствия
    tts_url = "https://tat-tts.api.translate.tatar/listening/"
    params = {"speaker": "alsu", "text": greeting}
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        audio_response = await client.get(tts_url, params=params)
    if audio_response.status_code != 200:
        raise HTTPException(502, f"TTS API error: {audio_response.text}")
    audio_base64 = audio_response.text

    return ChatOut(
        input_tat=None,
        translated_to_ru="Приветствие от ассистента",
        model_answer_ru=greeting,
        audio_base64=audio_base64,
        recognized_tat=None
    )


@app.post("/clear-history")
async def clear_chat_history():
    chat_history.clear()
    return {"status": "ok"}
