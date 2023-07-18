#!/usr/bin/env python

from time import sleep

import pathway as pw


class ExampleSubject(pw.io.python.ConnectorSubject):
    def run(self):
        i = 0
        while True:
            msg = f'{{ "key": 0, "value": {i} }}'
            self.next_bytes(msg.encode())
            self.commit()
            i += 1
            if i == 5:
                break
            sleep(1)


class DummySubject(pw.io.python.ConnectorSubject):
    def run(self):
        while True:
            print("Ping")
            sleep(1)


def main():
    dummy = pw.io.python.read(DummySubject(), value_columns=[], primary_key=[])

    t = pw.io.python.read(
        ExampleSubject(),
        value_columns=["value"],
        primary_key=["key"],
        autocommit_duration_ms=10000,
    )

    result = t.select(res=t.value * 10)

    t.debug("t")
    result.debug("result")
    dummy.debug("dummy")

    pw.io.csv.write(result, "output.csv")

    pw.run(debug=True)


if __name__ == "__main__":
    main()
