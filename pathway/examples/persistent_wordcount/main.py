#!/usr/bin/env python

import pathway as pw
from pathway.internals import parse_graph
from pathway.internals.rustpy_builder import RustpyBuilder


def main():
    words = pw.jsonlines.read(
        path="inputs/",
        value_columns=["word"],
        mode="streaming",
        persistent_id=1,
    )
    result = words.groupby(words.word).reduce(
        words.word,
        count=pw.reducers.count(),
    )
    pw.csv.write(result, "table.csv")

    RustpyBuilder(parse_graph.G).run_outputs()


if __name__ == "__main__":
    main()
