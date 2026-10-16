# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Чистые и небуферизованные логи в docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Сначала только зависимости — слой кэшируется при правках кода
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Код приложения (секреты через env_file в docker-compose, не здесь)
COPY config.py vectorstore.py tools.py agent.py bot.py main.py ./

# Запуск от некорневого пользователя
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
