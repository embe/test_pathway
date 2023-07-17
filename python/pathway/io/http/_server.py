import asyncio
import json
from typing import Any, Callable, Dict, Tuple
from uuid import uuid4

from aiohttp import web

import pathway.internals as pw
import pathway.io as io
from pathway.internals.api import BasePointer, unsafe_make_pointer


class RestServerSubject(io.python.ConnectorSubject):
    _host: str
    _port: int
    _loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        tasks: Dict[Any, Any],
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._loop = loop
        self._tasks = tasks

    def run(self):
        app = web.Application()
        app.add_routes([web.post("/", self.handle)])
        web.run_app(
            app, host=self._host, port=self._port, loop=self._loop, handle_signals=False
        )

    async def handle(self, request):
        id = unsafe_make_pointer(uuid4().int)
        query = await request.text()
        event = asyncio.Event()

        self._tasks[id] = {
            "event": event,
            "result": "-PENDING-",
        }

        payload = json.dumps({"query": query}).encode()
        self._add(id, payload)

        response = await self._fetch_response(id, event)
        return web.json_response(status=200, data=response)

    async def _fetch_response(self, id, event) -> Any:
        await event.wait()
        task = self._tasks.pop(id)
        return task["result"]


def rest_connector(host: str, port: int) -> Tuple[pw.Table, Callable]:
    loop = asyncio.new_event_loop()
    tasks: Dict[Any, Any] = {}

    input_table = io.python.read(
        subject=RestServerSubject(host=host, port=port, loop=loop, tasks=tasks),
        schema=pw.schema_builder(
            {
                "query": pw.column_definition(),
            }
        ),
        format="json",
    )

    def response_writer(responses: pw.Table):
        def on_change(
            key: BasePointer, row: Dict[str, Any], time: int, is_addition: bool
        ):
            task = tasks.get(key, None)

            assert task is not None, "query not found"

            def set_task():
                task["result"] = row["result"]
                task["event"].set()

            loop.call_soon_threadsafe(set_task)

        io.subscribe(table=responses, on_change=on_change)

    return input_table, response_writer
