# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Dict, Generic, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from pathway.internals.environ import engine_version
from pathway.internals.schema import Schema

_API_VARIANT = engine_version  # noqa

if TYPE_CHECKING or _API_VARIANT == "python":
    from ._python_api import *
else:
    from pathway.engine import *

    Value = Union[
        None,
        int,
        float,
        str,
        bool,
        BasePointer,
        datetime.datetime,
        datetime.timedelta,
        Tuple[Any, ...],
    ]  # Any should be Value
    CapturedTable = Dict[BasePointer, Tuple[Value, ...]]

TSchema = TypeVar("TSchema", bound=Schema)


# XXX: engine calls return BasePointer, not Pointer
class Pointer(BasePointer, Generic[TSchema]):
    """Pointer to row type.

    Example:

    >>> import pathway as pw
    >>> t1 = pw.debug.parse_to_table('''
    ... age | owner | pet
    ... 10  | Alice | dog
    ... 9   | Bob   | dog
    ... 8   | Alice | cat
    ... 7   | Bob   | dog
    ... ''')
    >>> t2 = t1.select(col=t1.id)
    >>> t2.schema['col'] is pw.Pointer
    True
    """

    pass


def _provide_for(cls):
    def extender(function):
        orig_function = function
        if isinstance(orig_function, property):
            orig_function = orig_function.fget
        name = orig_function.__name__
        if name not in cls.__dict__:
            setattr(cls, name, function)
        return function

    return extender


@_provide_for(Scope)
def static_table_from_pandas(
    self, df, dtypes=None, id_from=None, unsafe_trusted_ids=True
) -> Table:
    dtypes = dtypes or {}

    def denumpify_inner(x):
        if pd.api.types.is_scalar(x) and pd.isna(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x

    def denumpify(x):
        v = denumpify_inner(x)
        if isinstance(v, str):
            return v.encode("utf-8", "ignore").decode("utf-8")
        else:
            return v

    if id_from is None:
        if unsafe_trusted_ids:
            ids = {k: unsafe_make_pointer(k) for k in df.index}
        else:
            ids = {k: ref_scalar(k) for k in df.index}
    else:
        ids = {k: ref_scalar(*args) for (k, *args) in df[list(id_from)].itertuples()}

    universe = self.static_universe(ids.values())
    columns = []
    for c in df.columns:
        data = {ids[k]: denumpify(v) for k, v in df[c].items()}
        if c in dtypes:
            dtype = dtypes[c]
        else:
            dtype = int
            for v in data.values():
                if v is not None:
                    dtype = type(v)
                    break
        columns.append(self.static_column(universe, list(data.items()), dtype))
    return Table(universe, columns)
