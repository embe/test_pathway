import asyncio
import random
from typing import Any, Dict

import pathway as pw


class OutputSchema(pw.Schema):
    ret: int


class ExampleAsyncTransformer(pw.AsyncTransformer, output_schema=OutputSchema):  # type: ignore
    async def invoke(self, key: int, value: int) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0, 2))
        if random.randint(0, 2) > 0:
            raise Exception
        return dict(ret=value + 1)


if __name__ == "__main__":

    class InputStream(pw.io.python.ConnectorSubject):
        def run(self):
            for i in range(1, 10):
                self.next_json({"key": i, "value": i})

    input = pw.io.python.read(
        InputStream(), value_columns=["value"], primary_key=["key"], format="json"
    )

    table = (
        ExampleAsyncTransformer(input_table=input)
        .with_options(
            capacity=5,
            retry_strategy=pw.asynchronous.ExponentialBackoffRetryStrategy(),
            cache_strategy=pw.asynchronous.DiskCache("cache_example"),
        )
        .result
    )

    input.debug("input")
    table.debug("table")

    pw.io.csv.write(table, "out.csv")
    pw.run_all(debug=True)
