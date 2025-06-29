from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
sampling_params = SamplingParams(temperature=0.5)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


print("=" * 80)

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
outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print_outputs(outputs)
