# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

from __future__ import annotations

import pathway._engine_finder  # noqa: F401  # isort: split

import os

# flake8: noqa: E402

if "PYTEST_CURRENT_TEST" not in os.environ:
    from warnings import filterwarnings

    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)


from pathway import debug, demo, io
from pathway.engine import PathwayType as Type
from pathway.internals import (
    ClassArg,
    ColumnReference,
    FilteredJoinResult,
    GroupedJoinResult,
    GroupedTable,
    Joinable,
    JoinMode,
    JoinResult,
    MonitoringLevel,
    Pointer,
    Schema,
    Table,
    TableLike,
    TableSlice,
    __version__,
    apply,
    apply_async,
    apply_with_type,
    asynchronous,
    attribute,
    cast,
    coalesce,
    column_definition,
    declare_type,
    if_else,
    input_attribute,
    input_method,
    iterate,
    iterate_universe,
    left,
    make_tuple,
    method,
    numba_apply,
    output_attribute,
    reducers,
    require,
    right,
    run,
    run_all,
    schema_builder,
    schema_from_types,
    sql,
    this,
    transformer,
    universes,
)
from pathway.internals.api import PathwayType as Type
from pathway.stdlib import graphs, indexing, ml, ordered, statistical, temporal, utils
from pathway.stdlib.utils.async_transformer import AsyncTransformer
from pathway.stdlib.utils.pandas_transformer import pandas_transformer

__all__ = [
    "asynchronous",
    "ClassArg",
    "graphs",
    "utils",
    "debug",
    "indexing",
    "ml",
    "apply",
    "apply_async",
    "apply_with_type",
    "attribute",
    "declare_type",
    "cast",
    "GroupedTable",
    "input_attribute",
    "input_method",
    "iterate",
    "iterate_universe",
    "JoinResult",
    "FilteredJoinResult",
    "IntervalJoinResult",
    "method",
    "output_attribute",
    "pandas_transformer",
    "AsyncTransformer",
    "reducers",
    "transformer",
    "schema_from_types",
    "Table",
    "TableLike",
    "ColumnReference",
    "Schema",
    "Pointer",
    "MonitoringLevel",
    "WindowJoinResult",
    "this",
    "left",
    "right",
    "Joinable",
    "OuterJoinResult",
    "coalesce",
    "require",
    "sql",
    "run",
    "run_all",
    "if_else",
    "make_tuple",
    "Type",
    "numba_apply",
    "__version__",
    "io",
    "universes",
    "window",
    "JoinMode",
    "GroupedJoinResult",
    "AsofJoinResult",
    "temporal",
    "statistical",
    "schema_builder",
    "column_definition",
    "TableSlice",
    "demo",
]


def __getattr__(name: str):
    old_io_names = [
        "csv",
        "debezium",
        "elasticsearch",
        "http",
        "jsonlines",
        "kafka",
        "logstash",
        "null",
        "plaintext",
        "postgres",
        "python",
        "redpanda",
        "subscribe",
        "s3_csv",
    ]
    if name in old_io_names:
        from warnings import warn

        old_name = f"{__name__}.{name}"
        new_name = f"{__name__}.io.{name}"
        warn(f"{old_name} has been moved to {new_name}", stacklevel=3)
        return getattr(io, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


Table.asof_join = temporal.asof_join
Table.asof_join_left = temporal.asof_join_left
Table.asof_join_right = temporal.asof_join_right
Table.asof_join_outer = temporal.asof_join_outer

Table.window_join = temporal.window_join
Table.window_join_inner = temporal.window_join_inner
Table.window_join_left = temporal.window_join_left
Table.window_join_right = temporal.window_join_right
Table.window_join_outer = temporal.window_join_outer

Table.interval_join = temporal.interval_join
Table.interval_join_inner = temporal.interval_join_inner
Table.interval_join_left = temporal.interval_join_left
Table.interval_join_right = temporal.interval_join_right
Table.interval_join_outer = temporal.interval_join_outer

Table.interpolate = statistical.interpolate
Table.windowby = temporal.windowby
Table.sort = indexing.sort
Table.diff = ordered.diff
