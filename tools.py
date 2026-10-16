"""Инструменты (@tool) для RAG-агента: поиск, парсинг URL, котики, память пользователя."""

from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from pinecone_manager import get_pinecone_manager
from service_messages import is_service_text

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 20
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _user_id_from_config(config: RunnableConfig) -> str | None:
    """Достать user_id/chat_id из config'а агента (передаётся ботом при вызове).

    Параметр с аннотацией ``RunnableConfig`` инжектится фреймворком автоматически
    и не попадает в схему инструмента (LLM его не видит).
    """
    configurable: dict[str, Any] = config.get("configurable") or {}
    user_id = configurable.get("user_id")
    return str(user_id) if user_id is not None else None


def _fetch_page_text(url: str) -> str:
    """Скачать страницу и извлечь из неё основной текст."""
    response = requests.get(
        url,
        timeout=_HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """Семантический поиск по базе знаний и личной памяти пользователя.

    Используй, когда нужно вспомнить сохранённые статьи/документы или личные
    факты о собеседнике (имя, город, предпочтения и т.п.).
    """
    settings = get_settings()
    manager = get_pinecone_manager()
    try:
        docs = manager.search_by_text(
            query,
            namespace=settings.namespace_knowledge,
            k=settings.max_search_results,
        )
    except Exception as exc:  # noqa: BLE001 — инструмент не должен ронять агента
        logger.exception("Ошибка поиска по базе знаний")
        return f"Ошибка при поиске по базе знаний: {exc}"

    parts: list[str] = []
    if docs:
        parts.append("Результаты поиска по базе знаний:")
        for doc in docs:
            source = doc.metadata.get("source", "источник неизвестен")
            parts.append(f"[Источник: {source}]\n{doc.page_content}")

    user_id = _user_id_from_config(config)
    if user_id:
        try:
            memories = manager.search_by_text(
                query,
                namespace=settings.namespace_user_memories,
                k=settings.max_search_results,
                filter={"user_id": {"$eq": user_id}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка поиска по памяти пользователя")
            memories = []
            parts.append(f"Ошибка при поиске по памяти пользователя: {exc}")
        if memories:
            parts.append("Личные факты о пользователе:")
            for doc in memories:
                parts.append(f"- {doc.page_content}")

    if not parts:
        return "В базе знаний и личной памяти ничего не найдено по этому запросу."
    return "\n\n".join(parts)


@tool
def process_url_and_save(url: str) -> str:
    """Загрузить веб-страницу по URL, разбить её на фрагменты и сохранить в базу знаний.

    Используй, когда пользователь прислал ссылку и попросил изучить её содержимое
    или запомнить/сохранить его.
    """
    settings = get_settings()
    if not url.lower().startswith(("http://", "https://")):
        return "Ошибка: некорректный URL. Ожидается ссылка, начинающаяся с http:// или https://."

    try:
        text = _fetch_page_text(url)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка загрузки страницы %s", url)
        return f"Не удалось загрузить страницу {url}: {exc}"

    if len(text) < 50:
        return f"Страница {url} загружена, но содержит слишком мало текста для анализа."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": url, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]

    try:
        saved = get_pinecone_manager().add_documents(
            documents, namespace=settings.namespace_knowledge
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка сохранения в Pinecone")
        return f"Страница {url} распарсена ({len(documents)} чанков), но не сохранена в базу: {exc}"

    return f"Страница {url} успешно изучена и сохранена в базу знаний ({saved} чанков)."


@tool
def get_random_cat_fact() -> str:
    """Получить случайный факт о котиках из открытого API catfact.ninja.

    Используй, когда пользователь просит рассказать что-нибудь интересное о котиках.
    """
    try:
        response = requests.get("https://catfact.ninja/fact", timeout=10)
        response.raise_for_status()
        fact: str = response.json().get("fact", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка запроса к catfact.ninja")
        return f"Не удалось получить факт о котиках: {exc}"
    return f"Факт о котиках: {fact}" if fact else "Сервис с фактами о котиках не вернул данных."


@tool
def save_user_memory(fact: str, config: RunnableConfig) -> str:
    """Сохранить важный личный факт о пользователе в долговременную память.

    Используй, когда в диалоге появляется важная информация о собеседнике:
    имя, город, работа, увлечения, предпочтения, важные даты, а также цели,
    планы и мечты («хочу на Марс», «планирую переехать» и т.п.).
    Не сохраняй служебные и статусные сообщения.
    """
    settings = get_settings()
    user_id = _user_id_from_config(config)
    if not user_id:
        return "Ошибка: не удалось определить пользователя для сохранения памяти."

    # Служебные ответы бота не должны попадать в долговременную память.
    if is_service_text(fact):
        logger.info("Служебное сообщение не сохранено в память: %r", fact)
        return "Служебное сообщение не сохраняется в долговременную память."

    try:
        result = get_pinecone_manager().save_memory(
            fact,
            user_id=user_id,
            namespace=settings.namespace_user_memories,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка сохранения памяти пользователя")
        return f"Не удалось сохранить факт в память: {exc}"

    if result.action == "created":
        return f"Запомнил важный факт о пользователе: «{fact}»."
    if result.action == "updated":
        return (
            "Обновил существующий слот памяти — похожий факт уже был "
            f"(косинусное сходство {result.similarity:.2f}): «{fact}»."
        )
    return (
        "Похожий факт уже сохранён в памяти "
        f"(косинусное сходство {result.similarity:.2f}) — пропускаю запись."
    )


ALL_TOOLS = [
    search_knowledge_base,
    process_url_and_save,
    get_random_cat_fact,
    save_user_memory,
]
