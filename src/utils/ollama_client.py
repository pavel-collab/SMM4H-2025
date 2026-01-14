from ollama import Client

class OllamaClient:
    def __init__(self, server_url='http://localhost:11434', model_name='gemma3'):
        """Инициализация клиента Ollama.
        
        :param server_url: Адрес локального сервера Ollama (по умолчанию http://localhost:11434)
        :param model_name: Название используемой модели (например, llama2)
        """
        self.server_url = server_url
        self.model_name = model_name
        
        self.client = Client(
            host=self.server_url,
            # headers={'x-some-header': 'some-value'}
        )

    def suggest(self, prompt) -> list:
        try:
            response = self.client.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            
            return self._parse_output(response)
        except Exception as e:
            print(f"Ошибка при обращении к модели: {e}")
            return None

    def _parse_output(self, text: str) -> list:
        """Разбирает строку, возвращённую моделью, в список строк.
        
        :param text: Выход текста от модели
        :return: Списки команд, извлечённых из строки
        """
        # Пока этот метод просто возвращает исходный текст, но позже можно добавить обработку
        return text['message']['content']