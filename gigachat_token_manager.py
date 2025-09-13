import asyncio
import base64
import time
from threading import Lock, Thread
from typing import Optional

from gigachat import GigaChat

class GigaChatTokenManager:
    def __init__(self, client_id: str, client_secret: str, refresh_interval_minutes: int = 1):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_interval = refresh_interval_minutes * 60  # конвертируем в секунды
        self._token: Optional[str] = None
        self._token_lock = Lock()
        self._giga = self._create_client()
        
        # Запускаем обновление токена в отдельном потоке
        self._stop_worker = False
        self._worker_thread = Thread(target=self._token_refresh_worker, daemon=True)
        self._worker_thread.start()
    
    def _create_client(self) -> GigaChat:
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        return GigaChat(
            credentials=credentials,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )
    
    def _token_refresh_worker(self):
        """Фоновый поток для обновления токена"""
        while not self._stop_worker:
            try:
                token_obj = self._giga.get_token()
                new_token = f"Bearer {token_obj.access_token}"
                with self._token_lock:
                    self._token = new_token
                print("[GigaChat] Токен успешно обновлен")
            except Exception as e:
                print(f"[GigaChat] Ошибка при обновлении токена: {e}")
            
            # Спим до следующего обновления
            time.sleep(self.refresh_interval)
    
    def get_token(self) -> str:
        """Получить текущий токен"""
        with self._token_lock:
            if self._token is None:
                token_obj = self._giga.get_token()
                self._token = f"Bearer {token_obj.access_token}"
            return self._token
    
    def stop(self):
        """Остановить обновление токенов"""
        self._stop_worker = True
        self._worker_thread.join()

# Создаем глобальный экземпляр менеджера токенов
token_manager = GigaChatTokenManager(
    client_id="197a7642-ca99-42cd-a6a4-12051a052eed",
    client_secret="7b02f229-c521-45c1-abfc-79513074c881"
)

def get_token() -> str:
    """Получить текущий действующий токен"""
    return token_manager.get_token()
