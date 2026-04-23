from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Coroutine, Optional

import aiosqlite

from .. import CollectionProtocol, ConnectionProtocol, PoolProtocol
from ..attributed_dict import AttributedDict
from ..order import OrderType
from ..typemap import typemap as tm


class SQLiteCollection(CollectionProtocol):
    def __init__(self, connection: SQLiteConnection, name: str):
        self.connection = connection
        self.name = name

    async def find(
        self,
        _filter: dict,
        *,
        fields: tuple[str] = ("*",),
        order: tuple[tuple[OrderType, str]] = (),
    ) -> list[dict]:
        where_clause = ""
        params = []
        if _filter:
            conditions = [f"{key} = ?" for key in _filter.keys()]
            where_clause = f"WHERE {' AND '.join(conditions)}"
            params.extend(_filter.values())

        order_clause = ""
        if order:
            order_parts = [
                f"{col} {'ASC' if dir == OrderType.ASC else 'DESC'}"
                for dir, col in order
            ]
            order_clause = f"ORDER BY {', '.join(order_parts)}"

        query = f"SELECT {', '.join(fields)} FROM {self.name} {where_clause} {order_clause}"
        rows = await self.connection._fetch(query, *params)
        return [AttributedDict(row) for row in rows]

    async def find_one(
        self, _filter: dict, *, fields: tuple[str] = ("*",)
    ) -> Optional[dict]:
        result = await self.find(_filter, fields=fields)
        return result[0] if result else None

    async def insert(
        self,
        _object: dict,
        *,
        returning: tuple[str] = (),
        typemap: Optional[dict] = None,
    ) -> Optional[dict]:
        if not _object:
            return None

        data = tm(_object, typemap)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        returning_clause = ""
        if returning:
            returning_clause = f"RETURNING {', '.join(returning)}"

        query = f"INSERT INTO {self.name}({columns}) VALUES({placeholders}) {returning_clause}"
        rows = await self.connection._fetch(query, *data.values())
        return rows[0] if rows else None

    async def update(
        self,
        _filter: dict,
        _object: dict,
        *,
        returning: tuple[str] = (),
        typemap: Optional[dict] = None,
    ) -> list[dict]:
        if not _object:
            return []

        data = tm(_object, typemap)
        set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
        params = list(data.values())

        where_clause = ""
        if _filter:
            conditions = [f"{key} = ?" for key in _filter.keys()]
            where_clause = f"WHERE {' AND '.join(conditions)}"
            params.extend(_filter.values())

        returning_clause = ""
        if returning:
            returning_clause = f"RETURNING {', '.join(returning)}"

        query = f"UPDATE {self.name} SET {set_clause} {where_clause} {returning_clause}"
        return await self.connection._fetch(query, *params)

    async def delete(self, _filter: dict) -> None:
        where_clause = ""
        params = []
        if _filter:
            conditions = [f"{key} = ?" for key in _filter.keys()]
            where_clause = f"WHERE {' AND '.join(conditions)}"
            params.extend(_filter.values())

        query = f"DELETE FROM {self.name} {where_clause}"
        await self.connection._execute(query, *params)

    async def pop(
        self, _filter: dict, *, returning: tuple[str] = ("*",)
    ) -> Optional[dict]:
        where_clause = ""
        params = []
        if _filter:
            conditions = [f"{key} = ?" for key in _filter.keys()]
            where_clause = f"WHERE {' AND '.join(conditions)}"
            params.extend(_filter.values())

        returning_clause = f"RETURNING {', '.join(returning)}" if returning else ""
        query = f"DELETE FROM {self.name} {where_clause} {returning_clause}"
        rows = await self.connection._fetch(query, *params)
        return rows[0] if rows else None

    async def count(self, _filter: dict) -> int:
        where_clause = ""
        params = []
        if _filter:
            conditions = [f"{key} = ?" for key in _filter.keys()]
            where_clause = f"WHERE {' AND '.join(conditions)}"
            params.extend(_filter.values())

        query = f"SELECT COUNT(*) FROM {self.name} {where_clause}"
        rows = await self.connection._fetch(query, *params)
        return rows[0][0] if rows else 0

    async def get_size(self) -> int:
        # SQLite does not have a direct table size function,
        # but we can approximate with COUNT(*).
        return await self.count({})

    async def drop(self) -> None:
        await self.connection._execute(f"DROP TABLE IF EXISTS {self.name}")


class SQLiteConnection(ConnectionProtocol):
    def __init__(self, pool: SQLitePool, connection: aiosqlite.Connection, autocommit: bool = True):
        self._pool = pool
        self._connection = connection
        self._autocommit = autocommit
        self._in_transaction = False

    def __getattr__(self, key) -> CollectionProtocol:
        return SQLiteCollection(self, key)

    async def rollback(self) -> None:
        await self._connection.rollback()
        self._in_transaction = False

    async def commit(self) -> None:
        await self._connection.commit()
        self._in_transaction = False

    async def collections(self) -> list[str]:
        rows = await self._fetch(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in rows]

    async def __aenter__(self):
        # Begin a transaction
        await self._connection.execute("BEGIN")
        self._in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if not exc_type and self._autocommit:
                await self.commit()
            else:
                await self.rollback()
        finally:
            await self._pool.release(self)

    async def _fetch(self, query: str, *args, timeout: Optional[float] = None) -> list:
        # timeout is ignored for sqlite (could be implemented with asyncio.wait_for)
        query = " ".join(line.strip() for line in query.split("\n")).strip()
        async with self._connection.execute(query, args) as cursor:
            return await cursor.fetchall()

    async def _execute(self, query: str, *args) -> None:
        query = " ".join(line.strip() for line in query.split("\n")).strip()
        await self._connection.execute(query, args)


class SQLitePool(PoolProtocol):
    def __init__(self, database: str, autocommit: bool = True, **kwargs):
        self._database = database
        self._autocommit = autocommit
        self._kwargs = kwargs  # additional aiosqlite.connect kwargs
        self._pool: list[aiosqlite.Connection] = []
        self._lock = asyncio.Lock()
        self._tasks: list[Coroutine[Any, Any, Any]] = []
        self._closed = False

    async def connect(self) -> None:
        # Nothing to do; connections are created on acquire
        pass
    
    async def acquire(self) -> SQLiteConnection:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Pool is closed")
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = await aiosqlite.connect(self._database, **self._kwargs)
                await conn.execute("PRAGMA foreign_keys = ON")
                # row_factory будет установлен ниже единообразно
            # Устанавливаем row_factory для любой сессии из пула или новой
            conn.row_factory = aiosqlite.Row

        sqlite_conn = SQLiteConnection(self, conn, self._autocommit)
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            await task(sqlite_conn)
        return sqlite_conn

    async def release(self, connection: SQLiteConnection) -> None:
        conn = connection._connection
        if connection._in_transaction:
            await conn.rollback()
        # Сбрасываем row_factory на стандартный (кортежи), чтобы не влиять на другие возможные использования
        conn.row_factory = None
        async with self._lock:
            if not self._closed:
                self._pool.append(conn)
            else:
                await conn.close()
        del connection

    def on_acquire(self, coroutine: Coroutine[ConnectionProtocol, Any, Any]) -> None:
        self._tasks.append(coroutine)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
        await asyncio.sleep(0)


def SQLite(
    dsn: Optional[str] = None,
    *,
    database: Optional[str] = None,
    autocommit: bool = True,
    **kwargs,
) -> SQLitePool:
    """
    Create an asynchronous SQLite connection pool.

    :param database: Path to the SQLite database file (use ":memory:" for in-memory)
    :param autocommit: Whether to auto-commit after successful operations
    :param kwargs: Additional arguments passed to aiosqlite.connect
    :return: SQLitePool instance
    """
    if dsn:
        return SQLitePool(dsn.split(':')[1], autocommit=autocommit, **kwargs)
    return SQLitePool(database, autocommit=autocommit, **kwargs)