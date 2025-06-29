from transformers import pipeline

# Создаем генератор с предобученной моделью
generator = pipeline('text-generation', model='gpt2-medium')

# Задаем промпт без примеров
prompt = "Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug. Don't use general phrases, give an example of tweet or comment. Make a brief answer. Only text of the unswer without introduction phrases."

# Генерируем данные
output = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7
)

# Выводим результат
print("Zero-shot результат:")
print(output[0]['generated_text'])
