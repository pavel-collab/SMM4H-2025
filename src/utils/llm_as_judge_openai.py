import argparse
import openai
from openai import OpenAI
import pandas as pd
from pathlib import Path
import re
import json

'''
В этом скрипте мы используем подход llm as judge. То есть мы используем стороннюю модель
в качестве судьи. Примеры, сгенерированные дообученной моделью мы будем оценивать с помощью
другой модели. В данном случае просим GPT оценить, если ли в сгенерированном примере нужные 
нам упоминания. Здесь можно использовать openai или open router api. Плюс в том, что оттуда можно
испортнуть мощные модели по типу gpt-4o, которые нельзя дообучить, но которые можно использовать
через API.
'''

def extract_json(md_text):
    json_pattern = re.compile(r'``json\s*([\s\S]*?)\s*``', re.MULTILINE)
    matches = json_pattern.findall(md_text)
    json_objects = []
    for match in matches:
        try:
            json_objects.append(json.loads(match))
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")

    return json_objects

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

openai.api_key = TOKEN

for text in texts:
    messages = [
        {"role": "system",
        "content": "Ты эксперт по анализу текста. Твоя задача - определить, есть ли в тексте упоминание негативных явлний, связанных с приемом лекарственных препаратов."},
        {"role": "user", "content": f"""
        Целевое предложение: {text}
        
        0. Изучи предложение.
        1. Определи, есть ли в этом предложении упоминания негативных явлний, связанных с приемом лекарственных препаратов.
        2. Объясни свой выбор одним-двумя предложениями.
        
        В конце ответа обзятаельно напиши JSON с оценкой в формате {{"value": "&lt;твоя оценка&gt;", "explnation": "объяснение"}}.
        """}
    ]

    response = OpenAI().chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
    json_answer = extract_json(response.choices[0].message.content)[0]

    #!  Временная заглушка, здесь надо отладить скрипт хотя бы на этом этапе
    print(f"[DEBUG] text: {text}\n\tAnswer: {json_answer}")