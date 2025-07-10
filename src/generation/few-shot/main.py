from transformers import pipeline
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='gpt2-medium', help='set a path to generation model gguf file')
parser.add_argument('-n', '--num_generations', type=int, default=5, help='set number of output generations')
args = parser.parse_args()

model_name = args.model_name
num_generations = args.num_generations
assert(num_generations > 0)

#TODO: дополнительно написать альтернативный пример с применением chat-templetes

# Создаем генератор
generator = pipeline('text-generation', model=model_name)

# Задаем промпт с примерами
prompt = """
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug. Don't use general phrases, give an example of tweet or comment. Make a brief answer. Only text of the unswer without introduction phrases. 

Use the following examples:

1. banana? hot milk? and randomly lettuce! all contain sleepy bye chems. all i have is trazodone which means dopey all day tomo
2. Just about dead, think it's bedtime.. Fuck you quetiapine
3. oh man! i've been a total quetiapine zombie all morning! i've been taking them for years but every now  then they really mess me up!
4. My Philly dr prescribed me Trazodone,1pill made me so fkn sick, couldnt move 2day.Xtreme migraine, puke, shakes. Any1else get that react?
"""

# Генерируем данные
output = generator(
    prompt,
    max_length=200,
    num_return_sequences=num_generations,
    temperature=0.5,
    top_k=50
)

generations = [output[i]['generated_text'] for i in range(len(output))]

assert(len(generations) > 0)
assert(len(generations) == num_generations)

df = pd.DataFrame(generations, columns=['text'])

# Сохраняем в CSV
df.to_csv(f'./data/generated/generation_{model_name}.csv', index=False, encoding='utf-8')
