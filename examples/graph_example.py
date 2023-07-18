#!/usr/bin/env python

import io

import pandas as pd

import pathway as pw


def table_from_md(table_def, index_col=0):
    df = pd.read_table(
        io.StringIO(table_def),
        sep=r"(?:\s*\|\s*)|\s+",
        header=0,
        index_col=index_col,
        engine="python",
        skipinitialspace=True,
    )
    return pw.debug.table_from_pandas(df)


def main():
    t = table_from_md(
        """
    _ | foo
    1 | 10
    2 | 20
    """
    )
    t.debug("t")

    t2 = t.select(foo=42, bar=t.foo, baz=t.foo * 2)
    t2.debug("t2")

    state = pw.run()
    for graph_universe, api_universe in state.universes.items():
        print(graph_universe, api_universe, api_universe._data)
    for graph_column, api_column in state.columns.items():
        print(graph_column, api_column, api_column._data)


if __name__ == "__main__":
    main()
