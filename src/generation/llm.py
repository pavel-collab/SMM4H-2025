from typing import List
from google import generativeai as genai
import openai
import argparse
from dotenv import load_dotenv
import os
import json
import numpy as np
import pandas as pd

from ollama_client import OllamaClient
from open_router_client import OpenRouterClient
from hugging_face_client import HuggingFaceClient
from vllm_client import VllmClient

'''
Before run this script, you need to fill your API keys in .env file.
Check .env.example file for more information.
'''

PROMPT = """
Generate {num_of_examples} variative [TWEET] with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use images, don't use emojies. Don't use phrase "Adverse Drug Events" itself.
Generate answer as a string with json structure:
{{
    tweets: [
        {{tweet: [TWEET]}},
        {{tweet: [TWEET]}},
    ]
}}
"""

# ------------------------------
# OpenAI
# ------------------------------
class OpenAIClient:
    def __init__(self, api_key: str, model="gpt-4o"):
        openai.api_key = api_key
        self.model = model

    def suggest(self, prompt) -> List[str]:
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You're an assistant that helps users reuse their terminal commands efficiently."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        result = response.choices[0].message["content"]
        return self._parse_output(result)

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

    def suggest(self, prompt) -> list[str]:
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
    elif provider == "ollama":
        return OllamaClient(server_url='http://localhost:11435')
    elif provider == "openrouter":
        return OpenRouterClient(api_key)
    elif provider == "hugging_face":
        return HuggingFaceClient(api_key)
    elif provider == "vllm":
        return VllmClient(api_key=api_key)
    else:
        raise NotImplementedError(f"Set wrong provider: {provider}")
    
def load_api_key(provider: str) -> str:
    load_dotenv()

    if provider == "openai" or provider == "vllm":
        key = os.getenv("OPENAI_TOKEN")
    elif provider == "gemini":
        key = os.getenv("GEMINI_TOKEN")
    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_TOKEN")
    elif provider == "ollama":
        return None
    elif provider == "hugging_face":
        key = os.getenv("HUGGIGNFACE_TOKEN")
    else:
        raise ValueError(f"Set unsupported provider: {provider}")

    if not key:
        raise EnvironmentError(f"{provider.upper()} API key is wrong or expired.")
    return key
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--provider', type=str, default='openai', help='set a type of LLM provider, available: [openai | gemini | openrouter | ollama | hugging_face | vllm]')
    parser.add_argument('-n', type=int, default=5, help="set number of generations")
    parser.add_argument('-m', '--model', type=str, default='gpt-4o', help='set a name of used model (only for openrouter)')
    args = parser.parse_args()
    
    provider = args.provider
    
    api_key = load_api_key(provider)
    
    if api_key is None and provider != "ollama" and provider != "vllm":
        raise(Exception())
    
    llm = get_llm(provider=provider, api_key=api_key)
    
    #TODO: в генерациях Gemini часто маскирует названия лекарств. Возможно стоит обогатить
    #TODO: ее генерацию, например предоставить ей список доступных лекарств или даже использовать RAG
    llm_response = llm.suggest(PROMPT.format(num_of_examples = args.n))
    
    print(f"[DEBUG] {llm_response}")
    
    llm_response = llm_response.removesuffix('```').removeprefix('```json')
    
    print(f"[DEBUG] {llm_response}")
    
    llm_json_response = json.loads(llm_response)
    
    generated_tweets = llm_json_response.get('tweets')
    assert(generated_tweets is not None)
    
    generations = []
    for id, tweet in enumerate(generated_tweets):
        # print(f"[DEBUG] tweet [{id}]:\n\t{tweet['tweet']}")
        generations.append(tweet['tweet'])
        
    labels = np.ones((len(generations))).astype(int)
    # Сохраняем данные в csv файл
    df = pd.DataFrame({'text': generations, 'label': labels})

    #! Обратите внимание, что этот метод перезаписывает документ, то есть контент будет потерян
    df.to_csv(f'generations.csv', index=False)
    
if __name__ == '__main__':
    main()