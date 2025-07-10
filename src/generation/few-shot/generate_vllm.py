from vllm import LLM, SamplingParams
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='gpt2-medium', help='set a path to generation model gguf file')
parser.add_argument('-n', '--num_generations', type=int, default=5, help='set number of output generations')
args = parser.parse_args()

model_name = args.model_name
num_generations = args.num_generations
assert(num_generations > 0)

llm = LLM(model=model_name)
sampling_params = SamplingParams(temperature=0.5)

# In this script, we demonstrate how to pass input to the chat method:

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug. Don't use general phrases, give an example of tweet or comment. Make a brief answer. Only text of the unswer without introduction phrases."
    },
    {
        "role": "assistant",
        "content": "My Philly dr prescribed me Trazodone,1pill made me so fkn sick, couldnt move 2day.Xtreme migraine, puke, shakes. Any1else get that react?"
    },
    {
        "role": "user",
        "content": "Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug. Don't use general phrases, give an example of tweet or comment. Make a brief answer. Only text of the unswer without introduction phrases.",
    },
]

generations = llm.generate(conversation,
                   sampling_params=sampling_params,
                   n=num_generations)

assert(len(generations) > 0)
assert(len(generations) == num_generations)

df = pd.DataFrame(generations, columns=['text'])

# Сохраняем в CSV
df.to_csv(f'./data/generated/generation_{model_name}.csv', index=False, encoding='utf-8')