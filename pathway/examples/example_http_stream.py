#!/usr/bin/env python

import json
import os

import pathway as pw

BEARER_TOKEN = os.environ["TWITTER_API_TOKEN"]


def jsonlines():
    def mapper(msg: bytes) -> bytes:
        result = json.loads(msg.decode())
        return json.dumps(
            {"key": result["data"]["id"], "text": result["data"]["text"]}
        ).encode()

    table = pw.io.http.read(
        "https://api.twitter.com/2/tweets/search/stream",
        method="GET",
        headers={"Authorization": f"Bearer {BEARER_TOKEN}"},
        value_columns=["key", "text"],
        response_mapper=mapper,
        autocommit_duration_ms=1000,
    )

    pw.csv.write(table, "output.csv")

    pw.run(debug=True)


def raw():
    table = pw.io.http.read(
        "https://api.twitter.com/2/tweets/search/stream",
        method="GET",
        headers={"Authorization": f"Bearer {BEARER_TOKEN}"},
        format="raw",
    )

    table.debug("table")
    pw.csv.write(table, "output.csv")

    pw.run(debug=True)


if __name__ == "__main__":
    jsonlines()
