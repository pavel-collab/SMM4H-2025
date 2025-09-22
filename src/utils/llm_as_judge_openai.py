import argparse
import pandas as pd
from pathlib import Path
import json

from ollama_client import OllamaClient

'''
В этом скрипте мы используем подход llm as judge. То есть мы используем стороннюю модель
в качестве судьи. Примеры, сгенерированные дообученной моделью мы будем оценивать с помощью
другой модели. В данном случае просим GPT оценить, если ли в сгенерированном примере нужные 
нам упоминания. Здесь можно использовать openai или open router api. Плюс в том, что оттуда можно
испортнуть мощные модели по типу gpt-4o, которые нельзя дообучить, но которые можно использовать
через API.
'''

answer_map = {
    'yes': 1,
    'no': 0
}

parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, help='set an openai token')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set a path to file with synthetic data')
args = parser.parse_args()

TOKEN = args.token
assert(TOKEN is not None)

data_path = Path(args.data_path)
assert(data_path.exists())

synth_data_df = pd.read_csv(data_path.absolute())
texts = synth_data_df['text'].tolist()

llm = OllamaClient(server_url='http://localhost:11435')

for text in texts[:10]:
    messages = f"""You are an expert in text analysis. Your task is to determine if there is any mention of negative phenomena associated with taking medications in the text.
        Target offer: {text}
        
        0. Study the offer.
        1. Determine if there is any mention of negative phenomena associated with taking medications in this sentence.
        2. Explain your choice in one or two sentences.
        
        Return your response in a JSON format with the following structure: {{"value": "your estimate", "explanation": "explanation"}}.
        In the fild 'value' use only yes/no.
        """

    try:
        llm_response = llm_response.removesuffix('```').removeprefix('```json')
        llm_json_response = json.loads(llm_response)

        #!  Временная заглушка, здесь надо отладить скрипт хотя бы на этом этапе
        print(f"[DEBUG] text: {text}\n\tAnswer: {answer_map[llm_json_response['value']]}")
    except Exception as ex:
        print(f"[ERROR] {ex}")
        print(f"[DEBUG] {llm_response}")