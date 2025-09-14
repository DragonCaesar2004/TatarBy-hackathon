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
    # Твоя желаемая модель (может не уметь аудио) — авто-фолбэк выбирает подходящую
    GIGACHAT_MODEL: str = "GigaChat-Pro"
    REQUEST_TIMEOUT_SECONDS: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Внутренний кэш найденной аудио-модели
_AUDIO_MODEL_ID: Optional[str] = None

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
# GigaChat low-level
# =========================
def _bearer() -> str:
    bearer = get_token()
    if not bearer:
        raise HTTPException(500, "Не удалось получить токен GigaChat")
    return bearer

async def _client_json_headers() -> Dict[str, str]:
    return {
        "Authorization": _bearer(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

async def _client_multipart_headers() -> Dict[str, str]:
    # НЕ задаём Content-Type вручную — httpx сам проставит boundary
    return {
        "Authorization": _bearer(),
        "Accept": "application/json",
    }

async def _gc_list_models() -> List[str]:
    """
    Возвращает список ID моделей (или пустой список, если недоступно).
    """
    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_json_headers()
    try:
        print("[DEBUG] Запрашиваем список моделей у GigaChat...")
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
            r = await client.get(f"{base}/models", headers=headers)
        print(f"[DEBUG] /models status: {r.status_code}, response: {r.text}")
        if r.status_code != 200:
            return []
        j = r.json()
        data = j.get("data") or j.get("models") or []
        ids: List[str] = []
        for m in data:
            mid = m.get("id")
            if isinstance(mid, str):
                ids.append(mid)
        print(f"[DEBUG] Найдено моделей: {ids}")
        return ids
    except Exception as e:
        print(f"[ERROR] Ошибка при получении списка моделей: {e}")
        return []

def _rank_audio_candidates(all_models: List[str]) -> List[str]:
    """
    Составляем перечень кандидатов на аудио-модель.
    Приоритет: явно аудио-совместимые названия (по практике), затем — твоя preferred.
    """
    preferred = settings.GIGACHAT_MODEL
    # Часто аудио поддерживают «вторые/профи» модели; добавь сюда свои корпоративные ID при необходимости.
    known = [
        "GigaChat-2",
        "GigaChat-Pro",
        "GigaChat-Max",
        "GigaChat:2",           # префикс — будем матчить startswith
        "GigaChat-Pro-preview",
        "GigaChat-2-preview",
        "GigaChat",
    ]

    # 1) если список моделей известен — отсортируем по приоритетам / префиксам
    ranked: List[str] = []
    if all_models:
        # точные хиты
        for k in known:
            if ":" in k or "-" in k:
                for mid in all_models:
                    if mid == k and mid not in ranked:
                        ranked.append(mid)
        # префиксные хиты (например, GigaChat:2.0.28.2)
        for mid in all_models:
            if mid.startswith("GigaChat:2") and mid not in ranked:
                ranked.append(mid)
        # подстрахуемся: добавим preferred, если есть в списке
        if preferred in all_models and preferred not in ranked:
            ranked.append(preferred)
        # добьём остальными chat-моделями (если хочется широкого перебора)
        for mid in all_models:
            if mid not in ranked:
                ranked.append(mid)
        print('moooooooodel:   ', ranked)
        return ranked

    # 2) если /models недоступен — просто возвращаем known + preferred в конце
    base = [m for m in known if m != preferred]
    base.append(preferred)
    # уберём дубли и вернём
    seen = set()
    out = []
    for m in base:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out

async def _gc_upload_file(filename: str, content: bytes, mime: str) -> str:
    """
    POST /files — загрузка файла в хранилище (multipart).
    В некоторых контурах требуется purpose=general — передаём как обычное поле data.
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
        raise HTTPException(502, f"Неожиданный ответ /files: {j}")
    return fid

async def _gc_chat_with_model(messages: List[Dict[str, Any]], model: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Пытаемся вызвать /chat/completions на конкретной модели.
    Возвращает: (ok, assistant_text, raw_error_text)
    """
    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_json_headers()
    
    # Separate attachments from messages if present
    attachments = []
    processed_messages = []
    
    for message in messages:
        # Check if this is a user message with attachments
        if message.get("role") == "user" and "attachments" in message:
            # Extract attachments
            msg_attachments = message.get("attachments", [])
            if msg_attachments:
                # Convert file IDs to the proper attachment format
                for attachment_id in msg_attachments:
                    attachments.append({
                        "type": "input_audio",
                        "data": attachment_id  # This should be the file ID returned by /files endpoint
                    })
                # Remove attachments from the message
                message_copy = message.copy()
                message_copy.pop("attachments", None)
                processed_messages.append(message_copy)
            else:
                processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    # Build payload with attachments at top level if present
    payload = {
        "model": model,
        "messages": processed_messages,
        "temperature": 0.2
    }
    
    # Add attachments to payload if present
    if attachments:
        payload["attachments"] = attachments
    
    print(f"[DEBUG] Пробуем модель: {model}")
    print(f"[DEBUG] Payload: {payload}")
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, verify=False) as client:
        r = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
    print(f"[DEBUG] Ответ от /chat/completions для модели {model}: {r.status_code}, {r.text}")
    if r.status_code == 200:
        j = r.json()
        try:
            return True, j["choices"][0]["message"]["content"], None
        except Exception:
            print(f"[ERROR] Неожиданный JSON: {r.text}")
            return False, None, f"Unexpected JSON: {r.text}"

    # 422 из-за аудио — ключевой кейс
    if r.status_code == 422 and "Model does not support audio" in r.text:
        print(f"[DEBUG] Модель {model} не поддерживает аудио")
        return False, None, "NO_AUDIO"

    return False, None, r.text

async def _gc_chat_auto(messages: List[Dict[str, Any]]) -> str:
    """
    Вызывает чат: если текущая модель не умеет аудио — автопоиск и кэширование подходящей.
    """
    global _AUDIO_MODEL_ID
    print(f"[DEBUG] === _gc_chat_auto: AUDIO_MODEL_ID={_AUDIO_MODEL_ID}")
    print(f"[DEBUG] Сообщения для отправки: {messages}")

    # Если уже нашли и закэшировали — пробуем сначала её
    if _AUDIO_MODEL_ID:
        print(f"[DEBUG] Пробуем закэшированную модель: {_AUDIO_MODEL_ID}")
        ok, text, err = await _gc_chat_with_model(messages, _AUDIO_MODEL_ID)
        if ok:
            print(f"[DEBUG] Успех с кэшированной моделью: {_AUDIO_MODEL_ID}")
            return text
        print(f"[DEBUG] Кэшированная модель не сработала: {err}")
        _AUDIO_MODEL_ID = None

    # Составим список кандидатов: из /models (если доступен) + эвристики
    all_models = await _gc_list_models()

    
    print(f"[DEBUG] Кандидаты моделей: {all_models}")
    candidates = [settings.GIGACHAT_MODEL] 
    print(f"[DEBUG] Отранжированные кандидаты: {candidates}")

    # Пробуем по порядку, пока не найдём модель, которая принимает аудио
    last_err = None
    for mid in candidates:
        print(f"[DEBUG] Пробуем модель-кандидат: {mid}")
        ok, text, err = await _gc_chat_with_model(messages, mid)
        if ok:
            print(f"[DEBUG] Успех с моделью: {mid}")
            _AUDIO_MODEL_ID = mid  # кэшируем успешную
            return text
        print(f"[DEBUG] Ошибка для {mid}: {err}")
        last_err = err
        # Если модель явно не поддерживает аудио — просто идём дальше
        if err == "NO_AUDIO":
            continue
        # Иначе это другое отклонение (401/404/429/500) — тоже попробуем следующий mid

    # Если сюда дошли — ни одна из моделей не приняла аудио
    print(f"[ERROR] Ни одна из моделей не приняла аудио. last_err={last_err}")

    raise HTTPException(502, f"Не удалось получить ответ от моделей: {last_err or 'unknown error'}")

# =========================
# High-level
# =========================
async def gigachat_complete(prompt_ru: str, system_ru: Optional[str]) -> str:
    msgs: List[Dict[str, Any]] = []
    if system_ru:
        msgs.append({"role": "system", "content": system_ru})
    msgs.extend(chat_history.get_messages())
    msgs.append({"role": "user", "content": prompt_ru})

    # Для чисто текстового режима используем твою базовую модель напрямую
    base = settings.GIGACHAT_BASE_URL.rstrip("/")
    headers = await _client_json_headers()
    payload = {
        "model": settings.GIGACHAT_MODEL,
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
    Если модель не поддерживает аудио — авто-подбор подходящей и кэширование.
    """
    print(f"[DEBUG] === gigachat_audio_complete: Starting audio processing")
    print(f"[DEBUG] Audio bytes length: {len(wav_bytes)}")
    print(f"[DEBUG] System prompt: {system_ru}")
    
    file_id = await _gc_upload_file("audio.wav", wav_bytes, "audio/wav")
    print(f"[DEBUG] File uploaded with ID: {file_id}")

    msgs: List[Dict[str, Any]] = []
    if system_ru:
        msgs.append({"role": "system", "content": system_ru})
    msgs.extend(chat_history.get_messages())
    msgs.append({
        "role": "user",
        "content": "Транскрибируй это аудио и ответь по инструкции.",
        "attachments": [file_id],
    })
    
    print(f"[DEBUG] Messages to send: {msgs}")

    assistant_text = await _gc_chat_auto(msgs)
    print(f"[DEBUG] Received response from _gc_chat_auto: {assistant_text}")
    return None, assistant_text

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
        base_system_prompt if not incoming.system_prompt_ru
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
    
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave", "audio/x-pn-wav"):
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
        base_system_prompt if not system_prompt_ru
        else base_system_prompt + "\n\nДополнительные инструкции:\n" + system_prompt_ru
    )

    print(f"[DEBUG] Calling gigachat_audio_complete")
    _, assistant = await gigachat_audio_complete(wav_bytes, system_prompt)
    print(f"[DEBUG] Received response from gigachat_audio_complete: {assistant}")
    
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
    
    system_prompt = (
        "Ты - помощник, свободно владеющий татарским языком. "
        "Веди диалог ТОЛЬКО на татарском, современная орфография. "
        "Не используй спецсимволы в ответе. "
        f"Ты начала диалог с фразы: {greeting} - продолжай диалог в этом контексте."
    )
    
    # Добавляем системный промпт в начало истории
    chat_history.add_message("system", system_prompt)
    # Добавляем приветствие в историю
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
