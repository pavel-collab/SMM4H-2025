import requests
import json
from typing import Optional, Dict, Any

#! Here we need to use models with supported Chat Completion
class VllmClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen2.5-7B-Instruct-1M"):
        """
        Инициализация клиента для vLLM сервера.

        Args:
            api_key (str, optional): Ключ API, если сервер требует аутентификации.
            base_url (str): Базовый URL vLLM сервера. По умолчанию: "http://127.0.0.1:8000/v1".
            model (str): Название модели на сервере. По умолчанию: "roberta-base".
        """
        self.base_url = base_url.rstrip('/')  # Убедимся, что нет слеша в конце
        self.model_name = model
        self.max_tokens = 1000
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            # Если используется API-ключ, добавим его в заголовки
            self.headers["Authorization"] = f"Bearer {api_key}"

    def suggest(self, prompt: str, use_chat_endpoint: bool = True) -> str:
        """
        Отправляет промт модели и возвращает сгенерированный текст.

        Args:
            prompt (str): Текст промта (запроса пользователя).
            use_chat_endpoint (bool): True - использовать /chat/completions, False - использовать /completions.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        try:
            if use_chat_endpoint:
                # Формируем запрос для эндпоинта /chat/completions
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                endpoint = f"{self.base_url}/chat/completions"
            else:
                # Формируем запрос для эндпоинта /completions
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                endpoint = f"{self.base_url}/completions"

            # Отправляем POST-запрос на сервер
            response = requests.post(endpoint, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()  # Выбросит исключение для статусов 4xx/5xx

            # Парсим JSON-ответ
            response_data = response.json()

            # Извлекаем текст ответа в зависимости от эндпоинта
            if use_chat_endpoint:
                return response_data['choices'][0]['message']['content']
            else:
                return response_data['choices'][0]['text']

        except requests.exceptions.ConnectionError:
            return "Ошибка подключения: Не удается подключиться к серверу vLLM. Проверьте URL и доступность сервера."
        except requests.exceptions.Timeout:
            return "Ошибка таймаута: Сервер vLLM не ответил вовремя."
        except requests.exceptions.HTTPError as e:
            return f"HTTP-ошибка: {e}"
        except (KeyError, IndexError) as e:
            return f"Ошибка парсинга ответа от сервера: {e}. Полученный ответ: {response.text}"
        except Exception as e:
            return f"Произошла непредвиденная ошибка: {e}"
