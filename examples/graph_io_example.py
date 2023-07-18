#!/usr/bin/env python

import pathway as pw
from pathway.engine import PathwayType


def main():
    types = {
        "foo": PathwayType.INT,
        "foofoo": PathwayType.INT,
    }
    t = pw.io.csv.read(
        "tests/data/minimal_multicolumn.txt",
        ["key"],
        ["foo", "foofoo"],
        pw.io.CsvParserSettings(delimiter=","),
        types=types,
    )
    t.debug("t")
    t2 = t.select(foo=42, bar=t.foofoo, baz=t.foofoo * 2)
    pw.debug.compute_and_print(t2)


if __name__ == "__main__":
    main()
