import asyncio
import time

import pathway as pw


@pw.asynchronous.async_options(cache_strategy=pw.asynchronous.DiskCache(name="inc"))
async def inc(value: int):
    await asyncio.sleep(2)
    return value + 1


@pw.asynchronous.async_options(cache_strategy=pw.asynchronous.DiskCache())
def mult(value: int):
    time.sleep(2)
    return value * 2


if __name__ == "__main__":
    input = "   | value\n"
    for i in range(10):
        input += f"{i}  | {i}\n"

    table = pw.debug.table_from_markdown(input)

    result = table.select(
        a=pw.apply_async(inc, pw.this.value), b=pw.apply_async(mult, pw.this.value)
    )

    result.debug("result")
    pw.run_all(debug=True, monitoring_level=pw.MonitoringLevel.NONE)
