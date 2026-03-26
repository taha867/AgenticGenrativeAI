"""
Postgres checkpointer (STM) and PostgresStore (LTM) for LangGraph.
Same Postgres instance; use async variants for ainvoke.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Tuple

from backend.core.config import Settings, getSettings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def getCheckpointerAndStore(
    settings: Optional[Settings] = None,
) -> AsyncIterator[Tuple[Any, Any]]:
    """
    Async context manager yielding (checkpointer, store) for the same Postgres instance.
    Use in app lifespan: async with getCheckpointerAndStore(settings) as (cp, store): ...
    """
    settings = settings or getSettings()
    uri = settings.postgres_uri

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        logger.warning("langgraph-checkpoint-postgres not installed; STM/LTM disabled")
        yield (None, None)
        return

    try:
        from langgraph.store.postgres import PostgresStore
    except ImportError:
        logger.warning("langgraph store postgres not found; LTM store disabled")
        PostgresStore = None

    try:
        # Plain URI only; LangGraph does not accept engine_kwargs on from_conn_string.
        # Set default search_path on the DB role if needed: ALTER ROLE <user> SET search_path TO public;
        async with AsyncPostgresSaver.from_conn_string(uri) as checkpointer:
            await checkpointer.setup()

            if PostgresStore is None:
                yield (checkpointer, None)
                return

            with PostgresStore.from_conn_string(uri) as store:
                store.setup()
                yield (checkpointer, store)

    except Exception as e:
        logger.warning(
            "Postgres checkpointer/store connection failed (%s). Running without STM/LTM. "
            "Fix POSTGRES_URI in .env for persistence (user/password must match your PostgreSQL).",
            e,
        )
        yield (None, None)