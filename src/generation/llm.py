from typing import List
from google import generativeai as genai
import openai
import argparse
from dotenv import load_dotenv
import os

'''
Before run this script, you need to fill your API keys in .env file.
Check .env.example file for more information.
'''

PROMPT = f"""
Generate one [TWEET] with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use images, don't use emojies.
Generate tweet with the folowing template:

Tweet: [TWEET]
"""

# ------------------------------
# OpenAI
# ------------------------------
class OpenAIClient:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def suggest(self) -> List[str]:
        prompt = self._build_prompt()
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're an assistant that helps users reuse their terminal commands efficiently."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        result = response.choices[0].message["content"]
        return self._parse_output(result)

    def _build_prompt(self) -> str:
        return PROMPT

    def _parse_output(self, text: str) -> List[str]:
        #TODO: in this method will be text parsing
        return text


# ------------------------------
# Gemini
# ------------------------------
class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    def suggest(self) -> list[str]:
        prompt = PROMPT

        try:
            response = self.model.generate_content(prompt)
            raw_suggestions = response.text.strip()

            if not raw_suggestions:
                 return ""

            return self._parce_output(raw_suggestions)

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return []
        
    def _parce_output(self, text: str) -> List[str]:
        #TODO: in this function will be text parsing
        return text


def get_llm(provider: str, api_key: str):
    if provider == "openai":
        return OpenAIClient(api_key)
    elif provider == "gemini":
        return GeminiClient(api_key)
    else:
        raise NotImplementedError(f"지원하지 않는 provider입니다: {provider}")
    
def load_api_key(provider: str) -> str:
    load_dotenv()
    """
    선택한 provider에 맞는 API 키를 환경변수에서 불러옵니다.
    """
    if provider == "openai":
        key = os.getenv("OPENAI_TOKEN")
    elif provider == "gemini":
        key = os.getenv("GEMINI_TOKEN")
    else:
        raise ValueError(f"지원하지 않는 provider입니다: {provider}")

    if not key:
        raise EnvironmentError(f"{provider.upper()} API 키가 설정되어 있지 않습니다.")
    return key
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--provider', type=str, default='openai', help='set a type of LLM provider, available: [openai], [gemini]')
    args = parser.parse_args()
    
    api_key = load_api_key(provider)
    
    provider = args.provider
    llm = get_llm(provider=provider, api_key=api_key)
    
    suggestions = llm.suggest()
    
    if suggestions:
        for id, answer in enumerate(suggestions):
            print(f"LLM answer [{id}] is:\n\t{answer}\n")
    
if __name__ == '__main__':
    main()