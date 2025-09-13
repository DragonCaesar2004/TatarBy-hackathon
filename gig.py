"""
Установите библиотеку gigachat с помощью менеджера пакетов pip:

pip install gigachat
"""
import base64

from gigachat import GigaChat


CLIENT_ID = "197a7642-ca99-42cd-a6a4-12051a052eed"
CLIENT_SECRET = "7b02f229-c521-45c1-abfc-79513074c881"
credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

giga = GigaChat(
    credentials=credentials,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,  # <- ключевая строка
)

print(giga.get_token())