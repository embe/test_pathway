#!/usr/bin/env python

import pathway as pw


def main():
    t = pw.io.csv.read("tests/data/minimal_multicolumn.txt", ["key"], ["foo", "foofoo"])

    pw.io.postgres.write_snapshot(
        t,
        {
            "host": "localhost",
            "port": "5432",
            "dbname": "template1",
            "user": "sergey",
            "password": "sergey",
        },
        "test_snapshot",
        ["key"],
    )

    pw.run()


if __name__ == "__main__":
    main()
