"""Точка входа: инициализация конфигурации, прогрев Pinecone-индекса и запуск бота."""

from __future__ import annotations

import logging
import sys

from bot import create_bot
from config import ConfigError, get_settings
from pinecone_manager import get_pinecone_manager


def main() -> None:
    """Запустить Telegram-бота с RAG-агентом."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("main")

    try:
        get_settings()  # валидация .env до старта
        get_pinecone_manager().vectorstore  # прогрев: подключение/создание индекса Pinecone
    except ConfigError as exc:
        logger.error("❌ Ошибка конфигурации: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Не удалось подключиться к Pinecone: %s", exc)
        sys.exit(1)

    bot = create_bot()
    logger.info("🚀 Бот запущен. Ожидаю сообщения…")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    main()
