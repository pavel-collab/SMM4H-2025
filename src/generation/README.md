Каталог со скриптами для обучения и инференса всех видов генерации: генерация, аугментация, перевод, др

## Локальный запуск ollama

Для начала нужно скачать через ollama нужную модель
```
ollama pull <model name>
```

Необходимо в текущей сессии прописать переменные среды:
- сокет, на котором будет работать ollama (дефолтный порт 11434 иногда может быть занят)
- путь к сохраненным моделям (почему-то без него инстанс севера вообще не находит модели)

```
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
```

Запускаем сервер ollama
```
ollama serve
```

Обращаемся к локальному инференсу через API.

## Ограничение для публичных инференсов модели

Публичные инференсы имеют ограничение на бесплатное количество токенов, пока что я не придумал, как безболезненно пополнять сервисы, 
поэтому пока что расчитываем в основном на локальный инференс. 

Например в клиенте для openrouter, если не уменьшать периодически параметр max_tokens
```
response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You're an assistant that helps users reuse their terminal commands efficiently."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800, #! this parametr will need to be variate, cz the limit of tokens on the open router is getting lower
            )
```

то будем получать вот такой response
```
Ошибка при выполнении запроса: Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer max_tokens. You requested up to 16384 tokens, but can only afford 3920. To increase, visit https://openrouter.ai/settings/credits and upgrade to a paid account', 'code': 402, 'metadata': {'provider_name': None}}, 'user_id': 'user_2vMWLMAU339LLRtbVVRAPbUVrHh'}
```
потому что доступных токенов становится все меньше.

## Запуск сервиса vllm

Vllm позволяет запустить локальный инференс, по типу ollama. Для этого сначала вам нужно получить локально модель. 
Сделать это можно двумя способами:

1 способ: используйте huggingface-cli
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /path/to/local/model --local-dir-use-symlinks False
```
ATTENTION: с некоторыми моделями (типо llama от Meta) возникают проблемы, потому что для их получения необходим спец доступ.

2 способ: используй git
```
mkdir <loca model dir>
cd <local model dir>
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M
```
ATTENTION: фильтрациф файлов с помощью git lfs может занять какое-то время. Не прерывайте процесс.

После получения локальной модели, ее можно захостить с помощью vllm
```
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --host 127.0.0.1  # Важно: слушать все интерфейсы, а не только localhost \
    --port 8000 \
    --tensor-parallel-size 1  # Укажите количество GPU для тензорного параллелизма \
    --served-model-name my-llama-model  # Опционально: имя модели для клиента \
    --max-model-len 4096  # Опционально: максимальная длина контекста
```
ATTENTION: почему-то не со всеми моделями это хорошо срабатывает.