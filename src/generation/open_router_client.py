import os
from typing import List
from openai import OpenAI

class OpenRouterClient:
    def __init__(self, api_key: str, model="openai/gpt-4o"):
        """
        Инициализация клиента для работы с OpenRouter.

        :param api_key: Ваш API-ключ от OpenRouter
        :param model: Идентификатор модели в формате 'провайдер/название-модели'
                      По умолчанию 'openai/gpt-4o'
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def suggest(self, prompt) -> List[str]:
        """
        Отправка запроса к выбранной модели через OpenRouter.

        :param prompt: Текст запроса пользователя
        :return: Список предложений
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You're an assistant that helps users reuse their terminal commands efficiently."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800, #! this parametr will need to be variate, cz the limit of tokens on the open router is getting lower
            )
            result = response.choices[0].message.content
            return self._parse_output(result)
        
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {str(e)}")
            return []

    def _parse_output(self, text: str) -> List[str]:
        """
        Парсинг ответа от модели.
        TODO: Реализуйте вашу логику парсинга здесь

        :param text: Ответ от модели
        :return: Список распарсенных результатов
        """
        # Временная реализация - возвращаем как есть
        return text if text else None