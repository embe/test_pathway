#!/usr/bin/env python

import pathway.engine as api  # type: ignore


def main(g):
    key = api.ref_scalar(42)
    universe = g.static_universe([key])
    column = g.static_column(universe, [(key, "answer")], str)
    g.debug_universe("universe", universe)
    g.debug_column("column", column)
    table = g.table(universe, [column])
    column2 = g.map_column(table, lambda vs: vs[0] + " :)", str)
    g.debug_column("column2", column2)


if __name__ == "__main__":
    api.run_with_new_graph(main)
