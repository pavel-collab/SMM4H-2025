# import requests
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient

class HuggingFaceClient:
    def __init__(self, api_token: str, model_name="deepseek-ai/DeepSeek-V3-0324"):
        self.api_token = api_token
        self.model_name = model_name
        
        self.client = InferenceClient(api_key=api_token)        

    def suggest(self, prompt: str) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            # Парсим ответ
            return self._parse_output(response)
        
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {str(e)}")
            return None

    def _parse_output(self, response_data: Any) -> List[str]:
        return response_data.choices[0].message.content