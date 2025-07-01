from llama_cpp import Llama

# Инициализация модели (скачайте предварительно модель GGUF формата)
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Путь к GGUF модели
    n_ctx=2048,  # Размер контекста
    n_threads=4  # Количество потоков для обработки
)

prompt = f"""
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use general phrases, give an example of tweet or comment.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

# Генерация текста
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are helpfull assistent"},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,  # Максимальное количество токенов в ответе
    temperature=0.7,  # "Творческость" ответа (0-1)
    stop=["\n"]       # Символы, при которых генерация останавливается
)

# Вывод ответа
print(output["choices"][0]["message"]["content"])
