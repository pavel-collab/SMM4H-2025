from transformers import pipeline

# дополнительно написать альтернативный пример с применением chat-templetes

# Создаем генератор
generator = pipeline('text-generation', model='gpt2-medium')

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
    num_return_sequences=1,
    temperature=0.5,
    top_k=50
)

# Выводим результат
print("\nFew-shot результат:")
print(output[0]['generated_text'])
