import re
from bs4 import BeautifulSoup 
import emoji

def mask_user_mentions(text: str) -> str:
    """
    Заменяет все вхождения @USER___ (любое количество подчеркиваний) на [USER]
    """
    return re.sub(r'@USER_+', '[USER]', text)


def mask_emojis(text: str) -> str:
    """
    Заменяет все эмоджи в тексте на [EMOJI]
    """
    return ''.join('[EMOJI]' if emoji.is_emoji(char) else char for char in text)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # --- Базовая предобработка ---

    # Удаление HTML-тегов
    # text = BeautifulSoup(text, "html.parser").get_text()
    
    # вместо удаления html текста, мы можем его маскировать
    text = re.sub(r'<[^>]+>', '[HTML]', text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление URL-ов
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Удаление email-адресов
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Специальная предобработка ---

    # 9. Удаление markdown форматирования
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'^#+\s?', '', text, flags=re.MULTILINE)  # заголовки

    # 10. Нормализация чисел
    text = re.sub(r'\b\d+(\.\d+)?\b', '[NUM]', text)
    
    #TODO: need to test
    text = mask_user_mentions(text)
    
    text = mask_emojis(text)

    return text

SPECIAL_TOKENS = ['MATH', 'URL', '[EMAIL]', '[CODE]', '[NUM]']