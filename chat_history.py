from typing import List

class ChatHistory:
    def __init__(self, max_messages: int = 10):
        self.messages: List[dict] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """Добавить сообщение в историю"""
        self.messages.append({"role": role, "content": content})
        # Сохраняем только последние N сообщений
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[dict]:
        """Получить все сообщения"""
        return self.messages
    
    def clear(self):
        """Очистить историю"""
        self.messages = []

# Глобальный объект истории чата
chat_history = ChatHistory()
