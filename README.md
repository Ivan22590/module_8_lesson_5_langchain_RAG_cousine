# Telegram RAG-бот на LangChain + Pinecone

Telegram-бот с RAG-агентом (ReAct) на базе LangChain. Агент сам решает, когда
поискать в векторной базе Pinecone, когда изучать присланную ссылку, когда
запоминать личные факты о пользователе, а когда просто ответить.

## Возможности

- **`search_knowledge_base`** — семантический поиск по базе знаний и личной памяти пользователя;
- **`process_url_and_save`** — парсинг URL (BeautifulSoup), разбиение на чанки и сохранение в Pinecone;
- **`get_random_cat_fact`** — случайный факт о котиках с catfact.ninja;
- **`save_user_memory`** — сохранение важных личных фактов с метаданными `user_id`
  и умной фильтрацией дубликатов (косинусное сходство + порог);
- история переписки по чатам, статусы «Думаю…/Читаю статью…», понятные сообщения об ошибках.

## Структура проекта

```
.
├── pinecone_manager.py  # ЕДИНЫЙ модуль Pinecone: чтение/запись + умная память
├── config.py            # загрузка .env и настройки (Settings)
├── service_messages.py  # служебные сообщения бота + is_service_text()
├── tools.py             # 4 инструмента агента (@tool)
├── agent.py             # ReAct-агент, история диалога, тест подключения Pinecone
├── bot.py               # логика Telegram-бота
├── main.py              # точка входа (запуск бота)
├── requirements.txt
├── Dockerfile            # сборка образа с ботом
├── docker-compose.yml    # запуск бота на сервере
├── .env.example          # шаблон переменных окружения
├── .gitignore
└── .dockerignore
```

## Умная память (косинусное сходство)

Всё взаимодействие с Pinecone идёт через класс `PineconeManager`
(файл `pinecone_manager.py`):

- **запись**: `add_documents()` (документы), `add_vectors()` (сырые векторы),
  `save_memory()` (личная память с фильтрацией дубликатов);
- **чтение**: `search_by_text()` / `search_by_text_with_score()` (по тексту),
  `search_by_vector()` / `query_vector()` (по вектору);
- **сервис**: автосоздание индекса, `delete_namespace()`, `delete_index()`.

Перед записью в долговременную память менеджер считает косинусное сходство
между текущим сообщением и уже сохранёнными фрагментами памяти пользователя:

- `similarity < COSINE_SIMILARITY_THRESHOLD` — новая информация → `action: created`;
- `similarity >= COSINE_SIMILARITY_THRESHOLD` — дубликат/вариация →
  `DUPLICATE_ACTION = "update"` (перезапись слота, `action: updated`) либо
  `"skip"` (пропуск, `action: skipped`).

Граница и поведение задаются **глобальными переменными в начале
`pinecone_manager.py`**:

```python
COSINE_SIMILARITY_THRESHOLD = 0.75  # граница «низкого» и «высокого» сходства
DUPLICATE_ACTION = "update"         # что делать с дубликатом: "update" | "skip"
DEDUPE_ENABLED = True               # общий выключатель фильтрации
```

### Ручной тест менеджера

```bash
python pinecone_manager.py                # подключение, запись/чтение, демо дедупа
python pinecone_manager.py --delete-index # удалить (тестовый) индекс Pinecone
python pinecone_manager.py --reset-memory # очистить namespace личной памяти
```

В логах теста подключения появляется `✅ Подключение успешно`, а демо
фильтрации дубликатов выводит `action: created|updated|skipped` с фактическим
`similarity` — по нему удобно подбирать порог:

```text
action: created | similarity=0.0000 (порог=0.75) | id=... | text='Хочу на Марс'
action: updated | similarity=0.8123 (порог=0.75) | id=... | text='Я полечу на Марс'
```

## Настройка

### 1. Python

Нужен Python 3.11 или новее.

### 2. Виртуальное окружение

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Переменные окружения

Скопируйте шаблон и заполните значения:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Переменная              | Обязательна | Описание                                              |
|-------------------------|-------------|-------------------------------------------------------|
| `TELEGRAM_TOKEN`        | да          | токен бота от [@BotFather](https://t.me/BotFather)    |
| `PINECONE_API_KEY`      | да          | ключ Pinecone (Console → API Keys)                    |
| `PINECONE_INDEX_NAME`   | да          | имя индекса (будет создан автоматически, если нет)    |
| `OPENAI_API_KEY`        | да          | ключ OpenAI-совместимого провайдера                   |
| `OPENAI_BASE_URL`       | да          | базовый URL API провайдера (например, прокси/локальный сервер) |
| `OPENAI_MODEL`          | нет         | модель чата (по умолчанию `gpt-4o-mini`)              |
| `EMBEDDING_MODEL`       | нет         | модель эмбеддингов (по умолчанию `text-embedding-3-small`) |
| `EMBEDDING_DIM`         | нет         | размерность эмбеддингов (по умолчанию `1536`)         |

> Индекс Pinecone создаётся автоматически при старте (serverless, AWS, `us-east-1`,
> метрика `cosine`, размерность = `EMBEDDING_DIM`). Если вы меняете модель
> эмбеддингов — измените и `EMBEDDING_DIM`, либо укажите имя другого индекса.

### 4. Проверка подключения к Pinecone

```bash
python pinecone_manager.py    # тест менеджера: подключение, чтение/запись, дедуп
python agent.py               # короткая проверка подключения из агента
```

В логах должно появиться: `✅ Подключение успешно` (менеджер) или
`✅ Подключение к Pinecone успешно` (агент), либо
`❌ Ошибка ...` с описанием причины.

### 5. Запуск бота

```bash
python main.py
```

## Запуск в Docker (на сервере)

Предпосылка: `.env` уже создан и заполнен (см. шаг 3 выше).

```bash
# Собрать образ и запустить бота в фоне
docker compose up -d --build

# Логи в реальном времени
docker compose logs -f bot

# Тест подключения к Pinecone (разовая команда в контейнере)
docker compose run --rm bot python agent.py

# Остановить бота
docker compose down
```

Контейнеры перезапускаются автоматически (`restart: unless-stopped`) после
перезагрузки сервера или падения процесса. Логи ограничены ротацией
(10 МБ × 3 файла), чтобы не забивать диск. Секреты подставляются из `.env`
через `env_file` и не попадают ни в образ, ни в git.

Для сервера без Docker Compose (или в CI):

```bash
docker build -t rag-telegram-bot .
docker run -d --env-file .env --name rag-telegram-bot --restart unless-stopped rag-telegram-bot
```

## Использование

- `/start` — приветствие и справка;
- `/reset` — очистить историю диалога;
- «Изучи эту статью и запомни: `<URL>`» — агент распарсит страницу и сохранит её в базу знаний;
- «Что ты знаешь о …?» — поиск по базе знаний и личной памяти;
- «Я живу в Питере» — факт сохранится в память пользователя;
- «Расскажи факт о котиках» — случайный факт с catfact.ninja.

## Примечания

- История переписки хранится в памяти процесса и сбрасывается при рестарте бота;
  личные факты и база знаний хранятся в Pinecone и переживают рестарт.
- Память пользователя хранится в namespace `user_memories` с метаданными
  `{"user_id": ...}` и извлекается через тот же инструмент `search_knowledge_base`
  с фильтром по текущему пользователю.
- Все инструменты перехватывают сетевые и БД-ошибки и возвращают понятный текст —
  агент объяснит проблему пользователю, не роняя диалог.
