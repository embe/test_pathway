#!/usr/bin/env python

import pathway as pw
from pathway.internals import parse_graph
from pathway.internals.rustpy_builder import RustpyBuilder


def main():
    table = pw.csv.read(
        path="identity_inputs/",
        id_columns=None,
        value_columns=["key", "value"],
        mode="streaming",
        persistent_id=1,
    )

    pw.csv.write(table, "table.csv")

    RustpyBuilder(parse_graph.G).run_outputs()


if __name__ == "__main__":
    main()
