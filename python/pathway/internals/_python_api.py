# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

"""Variant of API with immediate evaluation in Python."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import functools
import hashlib
import itertools
import math
import sys
from collections import abc, defaultdict
from enum import Enum, unique
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from dateutil import tz

from pathway.internals._reducers import (
    _any,
    _argmax,
    _argmin,
    _count,
    _int_sum,
    _max,
    _min,
    _sorted_tuple,
    _sum,
    _unique,
)
from pathway.internals.dtype import DType  # type: ignore

if TYPE_CHECKING:
    from pathway.internals.monitoring import StatsMonitor


@functools.total_ordering
class BasePointer:
    _inner: int

    def __str__(self) -> str:
        return f"^{self._inner}"

    def __repr__(self) -> str:
        return f"BasePointer({self._inner})"

    def __index__(self) -> int:
        return self._inner

    def __hash__(self):
        return hash(self._inner)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BasePointer):
            return NotImplemented
        return self._inner == other._inner

    def __lt__(self, other) -> bool:
        if other == -math.inf:
            return False
        if other == math.inf:
            return True
        if not isinstance(other, BasePointer):
            return NotImplemented
        return self._inner < other._inner

    def __init__(self, inner):
        self._inner = inner


def ref_scalar(*args, optional=False) -> BasePointer:
    if optional and any(arg is None for arg in args):
        return None
    return BasePointer(
        int(hashlib.md5(f"{tuple(args)}".encode("utf8")).hexdigest(), 16) % (2**64)
    )


@unique
class JoinType(Enum):
    INNER = 1
    LEFT = 2
    OUTER = 3


class PathwayType(Enum):
    ANY = 1
    STRING = 2
    INT = 3
    BOOL = 4
    FLOAT = 5


class Universe:
    def __init__(self, scope: Scope, keys: Set[BasePointer]):
        """Warning: do not call directly."""
        # TODO(janek): convert to an abstract class once stable
        self._scope = scope
        for k in keys:
            if not isinstance(k, BasePointer):
                raise ValueError(f"{k} from {keys} is not a proper BasePointer value.")
        self._keys = frozenset(keys)
        self._key_hash = hash(tuple(sorted(self._keys)))

    def issubset(self, other: Universe):
        assert self._scope is other._scope
        if self._key_hash == other._key_hash:
            return True
        return self._keys.issubset(other._keys)

    def __eq__(self, other: Universe):
        assert self._scope is other._scope
        return self._key_hash == other._key_hash

    @property
    def id_column(self) -> Column:
        return UniverseColumn(self, {key: key for key in self._keys}, BasePointer)


class RestrictedDict(abc.Mapping):
    def __init__(self, universe_keys, backing_collection):
        while isinstance(backing_collection, RestrictedDict):
            backing_collection = backing_collection._backing_collection
        assert universe_keys.issubset(backing_collection.keys())
        self._backing_collection = backing_collection
        self._universe_keys = universe_keys

    def __getitem__(self, k):
        if k not in self._universe_keys:
            raise KeyError(k)
        return self._backing_collection[k]

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return self._universe_keys

    def __iter__(self):
        return iter(self.keys())


@dataclasses.dataclass(frozen=True)
class PyTrace:
    file_name: str
    line_number: int
    line: str


@dataclasses.dataclass(frozen=True)
class EvalProperties:
    trace: Optional[PyTrace] = None
    dtype: Optional[DType] = None


class Column:
    """A Column holds data and conceptually is a Dict[Universe elems, dt]

    Columns should not be constructed directly, but using methods of the scope.
    All fields are private.
    """

    def __init__(self, universe: Universe, data: abc.Mapping, dt: DType):
        """Warning: do not call directly."""
        self.universe = universe
        self.dtype = dt
        self._data = RestrictedDict(self.universe._keys, data)
        # TODO FIXME
        # for v in self._data.values():
        #     assert dtype_isinstance(v, dt)

    def _equal_content(self, other: Column, wo_types=False) -> bool:
        if self.universe != other.universe:
            return False
        if not wo_types and self.dtype != other.dtype:
            return False
        for k in self._data:
            if self._data[k] != other._data[k]:
                return False
        return True


class UniverseColumn(Column):
    """Marker class internal to the API to indicate the provenience of a column."""


class MethodColumn(Column):
    def __init__(self, universe: Universe, dt: DType, data: Value):
        super().__init__(universe, {key: (data, key) for key in universe._keys}, dt)


@dataclasses.dataclass(frozen=True)
class Table:
    """A `Table` is a thin wrapper over a list of Columns.

    universe and columns are public fields - tables can be constructed
    """

    universe: Universe
    columns: List[Column]

    def __post_init__(self):
        for col in self.columns:
            assert self.universe == col.universe

    def __repr__(self):
        cols = {f"c{i}": pd.Series(col._data) for i, col in enumerate(self.columns)}
        return pd.DataFrame(cols, index=self.universe._keys).__repr__()

    def _equal_content(self, other: Table, wo_types=False):
        if self.universe != other.universe:
            return False
        assert len(self.columns) == len(other.columns)
        for self_c, other_c in zip(self.columns, other.columns):
            if not self_c._equal_content(other_c, wo_types):
                return False
        return True


class Missing(Exception):
    "Marker class to indicate missing attributes"


missing = Missing()


Value = Union[
    None,
    int,
    float,
    str,
    bool,
    BasePointer,
    datetime.datetime,
    datetime.timedelta,
    Tuple["Value", ...],
]
MaybeValue = Union[Value, Missing]


class Reducer:
    ARG_MIN: "Reducer"
    MIN: "Reducer"
    ARG_MAX: "Reducer"
    MAX: "Reducer"
    SUM: "Reducer"
    INT_SUM: "Reducer"
    SORTED_TUPLE: "Reducer"
    COUNT: "Reducer"
    UNIQUE: "Reducer"
    ANY: "Reducer"

    def __init__(self, fun: Callable[[Dict[int, Value]], Value], ret_dtype: Value):
        self.fun = fun
        self.ret_dtype = ret_dtype

    def apply(self, args):
        return self.fun(args)


Reducer.ARG_MIN = Reducer(_argmin, int)
Reducer.MIN = Reducer(_min, int)
Reducer.ARG_MAX = Reducer(_argmax, int)
Reducer.MAX = Reducer(_max, int)
Reducer.SUM = Reducer(_sum, int)
Reducer.INT_SUM = Reducer(_int_sum, int)
Reducer.SORTED_TUPLE = Reducer(_sorted_tuple, Tuple[Value, ...])
Reducer.COUNT = Reducer(_count, int)
Reducer.UNIQUE = Reducer(_unique, Any)
Reducer.ANY = Reducer(_any, Any)


class Expression:
    _fun: Callable[Tuple[Value, ...], Value]

    # private
    def __init__(self, fun: Callable[Tuple[Value, ...], Value]):
        self._fun = fun

    def _eval(self, values: Tuple[Value, ...]) -> Value:
        return self._fun(values)

    @staticmethod
    def const(value: Value) -> Expression:
        return Expression(lambda values: value)

    @staticmethod
    def argument(index: int) -> Expression:
        return Expression(lambda values: values[index])

    @staticmethod
    def apply(fun: Callable, /, *args: Expression) -> Expression:
        def wrapped(values: Tuple[Value, ...]) -> Value:
            arg_values = tuple(e._eval(values) for e in args)
            return fun(*arg_values)

        return Expression(wrapped)

    @staticmethod
    def is_none(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values) is None)

    @staticmethod
    def not_(expr: Expression) -> Expression:
        return Expression(lambda values: not expr._eval(values))

    @staticmethod
    def and_(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) and rhs._eval(values))

    @staticmethod
    def or_(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) or rhs._eval(values))

    @staticmethod
    def xor(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) ^ rhs._eval(values))

    @staticmethod
    def int_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def int_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def int_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def int_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def int_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def int_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def int_neg(expr: Expression) -> Expression:
        return Expression(lambda values: -expr._eval(values))

    @staticmethod
    def int_add(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def int_sub(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def int_mul(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def int_floor_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) // rhs._eval(values))

    @staticmethod
    def int_true_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) / rhs._eval(values))

    @staticmethod
    def int_mod(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) % rhs._eval(values))

    @staticmethod
    def int_pow(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) ** rhs._eval(values))

    @staticmethod
    def int_lshift(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) << rhs._eval(values))

    @staticmethod
    def int_rshift(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >> rhs._eval(values))

    @staticmethod
    def int_and(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) & rhs._eval(values))

    @staticmethod
    def int_or(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) | rhs._eval(values))

    @staticmethod
    def int_xor(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) ^ rhs._eval(values))

    @staticmethod
    def float_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def float_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def float_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def float_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def float_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def float_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def float_neg(expr: Expression) -> Expression:
        return Expression(lambda values: -expr._eval(values))

    @staticmethod
    def float_add(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def float_sub(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def float_mul(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def float_floor_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) // rhs._eval(values))

    @staticmethod
    def float_true_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) / rhs._eval(values))

    @staticmethod
    def float_mod(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) % rhs._eval(values))

    @staticmethod
    def float_pow(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(
            lambda values: np.float64(lhs._eval(values))
            ** np.float64(rhs._eval(values))
        )

    @staticmethod
    def str_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def str_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def str_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def str_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def str_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def str_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def str_add(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def str_rmul(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def str_lmul(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def ptr_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def ptr_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def if_else(if_: Expression, then: Expression, else_: Expression) -> Expression:
        return Expression(
            lambda values: then._eval(values)
            if if_._eval(values)
            else else_._eval(values)
        )

    @staticmethod
    def int_to_float(expr: Expression) -> Expression:
        return Expression(lambda values: float(expr._eval(values)))

    @staticmethod
    def int_to_bool(expr: Expression) -> Expression:
        return Expression(lambda values: bool(expr._eval(values)))

    @staticmethod
    def int_to_str(expr: Expression) -> Expression:
        return Expression(lambda values: str(expr._eval(values)))

    @staticmethod
    def float_to_int(expr: Expression) -> Expression:
        return Expression(lambda values: int(expr._eval(values)))

    @staticmethod
    def float_to_bool(expr: Expression) -> Expression:
        return Expression(lambda values: bool(expr._eval(values)))

    @staticmethod
    def float_to_str(expr: Expression) -> Expression:
        return Expression(lambda values: str(expr._eval(values)))

    @staticmethod
    def bool_to_int(expr: Expression) -> Expression:
        return Expression(lambda values: int(expr._eval(values)))

    @staticmethod
    def bool_to_float(expr: Expression) -> Expression:
        return Expression(lambda values: float(expr._eval(values)))

    @staticmethod
    def bool_to_str(expr: Expression) -> Expression:
        return Expression(lambda values: str(expr._eval(values)))

    @staticmethod
    def str_to_int(expr: Expression) -> Expression:
        return Expression(lambda values: int(expr._eval(values)))

    @staticmethod
    def str_to_float(expr: Expression) -> Expression:
        return Expression(lambda values: float(expr._eval(values)))

    @staticmethod
    def str_to_bool(expr: Expression) -> Expression:
        return Expression(lambda values: bool(expr._eval(values)))

    @staticmethod
    def date_time_naive_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def date_time_naive_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def date_time_naive_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def date_time_naive_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def date_time_naive_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def date_time_naive_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def date_time_naive_nanosecond(expr: Expression) -> Expression:
        def get_nanosecond(values):
            timestamp = expr._eval(values)
            return timestamp.nanosecond + 1000 * timestamp.microsecond

        return Expression(get_nanosecond)

    @staticmethod
    def date_time_naive_microsecond(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).microsecond)

    @staticmethod
    def date_time_naive_millisecond(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).microsecond // 1000)

    @staticmethod
    def date_time_naive_second(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).second)

    @staticmethod
    def date_time_naive_minute(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).minute)

    @staticmethod
    def date_time_naive_hour(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).hour)

    @staticmethod
    def date_time_naive_day(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).day)

    @staticmethod
    def date_time_naive_month(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).month)

    @staticmethod
    def date_time_naive_year(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).year)

    @staticmethod
    def date_time_naive_timestamp(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).value)

    @staticmethod
    def date_time_naive_sub(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def date_time_naive_add_duration(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def date_time_naive_sub_duration(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def date_time_naive_strptime(expr: Expression, fmt: Expression) -> Expression:
        return Expression(
            lambda values: pd.to_datetime(
                expr._eval(values), format=fmt._eval(values).replace("%6f", "%f")
            )
        )

    @staticmethod
    def date_time_naive_strftime(expr: Expression, fmt: Expression) -> Expression:
        return Expression(
            lambda values: expr._eval(values).strftime(
                fmt._eval(values).replace("%6f", "%f")
            )
        )

    @staticmethod
    def date_time_naive_from_timestamp(
        expr: Expression, unit: Expression
    ) -> Expression:
        def convert(timestamp: int, unit: str):
            mult = {
                "s": 1_000_000_000,
                "ms": 1_000_000,
                "us": 1_000,
                "ns": 1,
            }.get(unit)
            if mult is None:
                raise ValueError(f"unit has to be one of s, ms, us, ns but is {unit}.")
            return pd.Timestamp(timestamp * mult)

        return Expression(
            lambda values: convert(expr._eval(values), unit._eval(values))
        )

    @staticmethod
    def date_time_naive_to_utc(
        expr: Expression, from_timezone: Expression
    ) -> Expression:
        return Expression(
            lambda values: expr._eval(values)
            .tz_localize(
                from_timezone._eval(values).replace("%6f", "%f"), ambiguous=False
            )  # ambiguous=False uses later date if more than one is available
            .tz_convert(tz.UTC)
        )

    @staticmethod
    def date_time_naive_round(expr: Expression, duration: Expression) -> Expression:
        return Expression(
            lambda values: expr._eval(values).round(duration._eval(values))
        )

    @staticmethod
    def date_time_naive_floor(expr: Expression, duration: Expression) -> Expression:
        return Expression(
            lambda values: expr._eval(values).floor(duration._eval(values))
        )

    @staticmethod
    def date_time_utc_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def date_time_utc_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def date_time_utc_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def date_time_utc_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def date_time_utc_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def date_time_utc_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def _date_time_utc_localize(expr: Expression, values: Any) -> pd.Timestamp:
        return expr._eval(values).tz_convert(tz.UTC)

    @staticmethod
    def date_time_utc_nanosecond(expr: Expression) -> Expression:
        def get_nanosecond(values):
            timestamp = Expression._date_time_utc_localize(expr, values)
            return timestamp.nanosecond + 1000 * timestamp.microsecond

        return Expression(get_nanosecond)

    @staticmethod
    def date_time_utc_microsecond(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).microsecond
        )

    @staticmethod
    def date_time_utc_millisecond(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).microsecond
            // 1000
        )

    @staticmethod
    def date_time_utc_second(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).second
        )

    @staticmethod
    def date_time_utc_minute(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).minute
        )

    @staticmethod
    def date_time_utc_hour(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).hour
        )

    @staticmethod
    def date_time_utc_day(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).day
        )

    @staticmethod
    def date_time_utc_month(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).month
        )

    @staticmethod
    def date_time_utc_year(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).year
        )

    @staticmethod
    def date_time_utc_timestamp(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).value
        )

    @staticmethod
    def date_time_utc_sub(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def date_time_utc_add_duration(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def date_time_utc_sub_duration(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def date_time_utc_strptime(expr: Expression, fmt: Expression) -> Expression:
        return Expression(
            lambda values: pd.to_datetime(
                expr._eval(values), format=fmt._eval(values)
            ).tz_convert(tz.UTC)
        )

    @staticmethod
    def date_time_utc_strftime(expr: Expression, fmt: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values).strftime(
                fmt._eval(values).replace("%6f", "%f")
            )
        )

    @staticmethod
    def date_time_utc_to_naive(expr: Expression, to_timezone: Expression) -> Expression:
        return Expression(
            lambda values: Expression._date_time_utc_localize(expr, values)
            .tz_convert(to_timezone._eval(values).replace("%6f", "%f"))
            .tz_localize(None)
        )

    @staticmethod
    def date_time_utc_round(expr: Expression, duration: Expression) -> Expression:
        return Expression(
            lambda values: expr._eval(values).round(duration._eval(values))
        )

    @staticmethod
    def date_time_utc_floor(expr: Expression, duration: Expression) -> Expression:
        return Expression(
            lambda values: expr._eval(values).floor(duration._eval(values))
        )

    @staticmethod
    def duration_eq(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) == rhs._eval(values))

    @staticmethod
    def duration_ne(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) != rhs._eval(values))

    @staticmethod
    def duration_lt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) < rhs._eval(values))

    @staticmethod
    def duration_le(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) <= rhs._eval(values))

    @staticmethod
    def duration_gt(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) > rhs._eval(values))

    @staticmethod
    def duration_ge(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) >= rhs._eval(values))

    @staticmethod
    def _duration_get_in_unit(
        expr: Expression, values: Any, num_nanoseconds: int
    ) -> int:
        nanoseconds = expr._eval(values).value
        return (abs(nanoseconds) // num_nanoseconds) * (-1 if nanoseconds < 0 else 1)

    @staticmethod
    def duration_nanoseconds(expr: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values).value)

    @staticmethod
    def duration_microseconds(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(expr, values, 1_000)
        )

    @staticmethod
    def duration_milliseconds(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(expr, values, 1_000_000)
        )

    @staticmethod
    def duration_seconds(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(expr, values, 1_000_000_000)
        )

    @staticmethod
    def duration_minutes(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(
                expr, values, 1_000_000_000 * 60
            )
        )

    @staticmethod
    def duration_hours(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(
                expr, values, 1_000_000_000 * 60 * 60
            )
        )

    @staticmethod
    def duration_days(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(
                expr, values, 1_000_000_000 * 60 * 60 * 24
            )
        )

    @staticmethod
    def duration_weeks(expr: Expression) -> Expression:
        return Expression(
            lambda values: Expression._duration_get_in_unit(
                expr, values, 1_000_000_000 * 60 * 60 * 24 * 7
            )
        )

    @staticmethod
    def duration_neg(expr: Expression) -> Expression:
        return Expression(lambda values: -expr._eval(values))

    @staticmethod
    def duration_add(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def duration_add_date_time_naive(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def duration_add_date_time_utc(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) + rhs._eval(values))

    @staticmethod
    def duration_sub(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) - rhs._eval(values))

    @staticmethod
    def duration_mul_by_int(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def int_mul_by_duration(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) * rhs._eval(values))

    @staticmethod
    def duration_div_by_int(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) // rhs._eval(values))

    @staticmethod
    def duration_floor_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) // rhs._eval(values))

    @staticmethod
    def duration_true_div(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) / rhs._eval(values))

    @staticmethod
    def duration_mod(lhs: Expression, rhs: Expression) -> Expression:
        return Expression(lambda values: lhs._eval(values) % rhs._eval(values))

    @staticmethod
    def pointer_from(*args: Expression, optional: bool) -> Expression:
        return Expression.apply(
            lambda *args: ref_scalar(*args, optional=optional), *args
        )

    @staticmethod
    def make_tuple(*args: Expression) -> Expression:
        return Expression.apply(lambda *args: tuple(args), *args)

    @staticmethod
    def sequence_get_item_checked(
        expr: Expression, index: Expression, default: Expression
    ) -> Expression:
        def get(values):
            result = expr._eval(values)
            try:
                return result[index._eval(values)]
            except IndexError:
                return default._eval(values)

        return Expression(get)

    @staticmethod
    def sequence_get_item_unchecked(expr: Expression, index: Expression) -> Expression:
        return Expression(lambda values: expr._eval(values)[index._eval(values)])


class MonitoringLevel(Enum):
    NONE = 0
    IN_OUT = 1
    ALL = 2


class Context:
    # "Location" of the current attribute in the transformer computation
    this_row: BasePointer
    data: Tuple[Value, BasePointer]

    # Transformer call context: fetching of missing values
    def get(self, column: int, row: BasePointer, *args: Value) -> MaybeValue:
        """Try getting inputs[column][row](*args).

        Returns Missing if the value is not available. The calling code should
        abort when trying to use a missing value.
        """
        raise NotImplementedError()

    def raising_get(self, column: int, row: BasePointer, *args: Value) -> Value:
        ret = self.get(column, row, *args)
        if isinstance(ret, Missing):
            raise ret
        return ret


@dataclasses.dataclass(frozen=True)
class Computer:
    _fun: Callable[[Context], MaybeValue]
    dtype: DType
    is_output: bool  # Is this a requested output
    is_method: bool  # Is this a method attr
    universe: Universe
    data: Value
    data_column: Optional[Column]

    @classmethod
    def from_raising_fun(
        cls, fun, *, dtype, is_output, is_method, universe, data=None, data_column=None
    ):
        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            try:
                return fun(*args, **kwargs)
            except Missing as e:
                return e

        return cls(wrapped, dtype, is_output, is_method, universe, data, data_column)


ComplexColumn = Union[Column, Computer]


class Computation:
    @dataclasses.dataclass(frozen=True)
    class Request:
        column: int
        row: BasePointer
        args: Tuple[Value]

        def cache_key(self):
            if self.args:
                return (self.row, tuple(self.args))
            else:
                return self.row

    def __init__(self, inputs: List[ComplexColumn]):
        self.inputs = inputs
        self.results = []
        for col in inputs:
            if isinstance(col, Column):
                self.results.append(None)
            else:
                assert isinstance(col, Computer)
                self.results.append({})
        self._evaluation_stack: List[Computation.Request] = []

    def get(self, column: int, row: BasePointer, *args: Value) -> MaybeValue:
        ret_column = self.results[column]
        if ret_column is None:
            # It's an input!
            assert not args, "Calling an input column which is not a method!"
            return self.inputs[column]._data[row]
        else:
            r = self.Request(column, row, tuple(args))
            return self.queue(r)

    def fetch_results(self) -> List[Table]:
        ret = []
        for res, col_def in zip(self.results, self.inputs):
            if isinstance(col_def, Computer) and col_def.is_output:
                if col_def.is_method:
                    col = MethodColumn(col_def.universe, col_def.dtype, col_def.data)
                else:
                    col = Column(
                        col_def.universe,
                        res,
                        col_def.dtype,
                    )
                ret.append(col)
        return ret

    def queue(self, request):
        col = self.results[request.column]
        key = request.cache_key()
        if key not in col:
            col[key] = missing
        if isinstance(col[key], Missing):
            self._evaluation_stack.append(request)
        return col[key]

    def compute_all_outputs(self):
        for col, col_def in enumerate(self.inputs):
            if isinstance(col_def, Computer) and col_def.is_output:
                if col_def.is_method:
                    continue
                for row in col_def.universe._keys:
                    self.queue(self.Request(col, row, ()))
                    self._compute()

    def _compute(self):
        stack = self._evaluation_stack
        limit = sys.getrecursionlimit()
        while stack:
            r = stack[-1]
            cache_key = r.cache_key()
            if not isinstance(self.results[r.column][cache_key], Missing):
                stack.pop()
            else:
                stack_depth = len(stack)
                computer = self.inputs[r.column]
                data = (
                    computer.data_column._data[r.row]
                    if computer.data_column is not None
                    else None
                )
                maybe_value = computer._fun(
                    PrivateContext(self, r.column, r.row, data), *r.args
                )
                if isinstance(maybe_value, Missing):
                    if len(stack) == stack_depth:
                        raise RecursionError("Circular attribute dependency")
                    if len(stack) > limit:
                        raise RecursionError("Recursion limit reached")
                else:
                    assert (
                        len(stack) == stack_depth
                    ), "Computation succeeded but made new requests"
                    stack.pop()
                    self.results[r.column][cache_key] = maybe_value


class PrivateContext(Context):
    def __init__(self, computation, column, row, data) -> None:
        self.computation = computation
        self.this_row = row
        self.data = data

    def get(self, column: int, row: BasePointer, *args: Value) -> MaybeValue:
        return self.computation.get(column, row, *args)


class VennUniverses:
    _only_left: Universe
    _only_right: Universe
    _both: Universe

    def __init__(self, only_left, only_right, both):
        self._only_left = only_left
        self._only_right = only_right
        self._both = both

    def only_left(self):
        return self._only_left

    def only_right(self):
        return self._only_right

    def both(self):
        return self._both


class Scope:
    _empty_universe: Universe
    # empty_column: Column
    _parent: Optional[Scope]

    def __init__(self, parent=None) -> None:
        self._parent = parent
        self._empty_universe = Universe(self, {})

    # Transport data between child and parent scopes

    def _import_universe_from_parent(self, universe: Universe) -> Universe:
        assert universe._scope is self._parent
        return Universe(self, universe._keys)

    def __import_tables(self, import_universe, tables: List[Table]) -> List[Table]:
        ret = []
        for table in tables:
            universe = import_universe(table.universe)
            # no need to further check scopes because all columns in a table must have a
            # compatible universe with same scope
            columns = [
                Column(universe, column._data, column.dtype) for column in table.columns
            ]
            ret.append(Table(universe, columns))
        return ret

    def _import_tables_from_parent(self, tables):
        return self.__import_tables(self._import_universe_from_parent, tables)

    def _import_universe_from_child(self, universe: Universe) -> Universe:
        assert universe._scope._parent is self
        return Universe(self, universe._keys)

    def _import_tables_from_child(self, tables):
        return self.__import_tables(self._import_universe_from_child, tables)

    # IO

    def empty_table(self, dtypes: Iterable[DType]) -> Table:
        columns = [Column(self._empty_universe, {}, dt) for dt in dtypes]
        return Table(self._empty_universe, columns)

    def _equal_tables(self, lefts, rights):
        assert len(lefts) == len(rights)
        for left, right in zip(lefts, rights):
            if not left._equal_content(right):
                return False
        return True

    def iterate(
        self,
        iterated: List[Table],
        iterated_with_universe: List[Table],
        extra: List[Table],
        logic: Callable[
            [Scope, List[Table], List[Table], List[Table]],
            Tuple[List[Table], List[Table]],
        ],
        *,
        limit: Optional[int] = None,
    ) -> Tuple[List[Table], List[Table]]:
        """Fixed-point iteration

        logic is called with a new scope, clones of tables from iterated,
        clones of tables from extra.
        logic should not use any other outside tables.
        logic must return a list of tables corresponding to iterated:
        result[i] is the result of single iteration on iterated[i]
        """
        if limit is not None and limit <= 1:
            raise ValueError("iteration limit must be greater than 1")
        sub_scope = Scope(parent=self)
        iterated = sub_scope._import_tables_from_parent(iterated)
        iterated_with_universe = sub_scope._import_tables_from_parent(
            iterated_with_universe
        )
        extra = sub_scope._import_tables_from_parent(extra)
        for iteration in itertools.count():
            if limit is not None and iteration >= limit:
                break
            (updated, updated_with_universe) = logic(
                sub_scope, iterated, iterated_with_universe, extra
            )
            if self._equal_tables(iterated, updated) and self._equal_tables(
                iterated_with_universe, updated_with_universe
            ):
                break
            else:
                iterated = updated
                iterated_with_universe = updated_with_universe
        return (
            self._import_tables_from_child(iterated),
            self._import_tables_from_child(iterated_with_universe),
        )

    # Evaluators for expressions

    def _validate(self, universe, *columns):
        assert universe._scope is self
        for column in columns:
            assert universe.issubset(column.universe)

    def static_universe(self, keys: Iterable[BasePointer]):
        return Universe(self, set(keys))

    def static_column(
        self, universe: Universe, rows: Iterable[Tuple[BasePointer, Any]], dt: DType
    ):
        return Column(universe, dict(rows), dt)  # XXX: assert

    def map_column(
        self,
        table: Table,
        function: Callable[[[Value]], Value],
        properties: EvalProperties,
    ) -> Column:
        universe = table.universe
        self._validate(universe)
        data = {}
        for k in universe._keys:
            inputs = [c._data[k] for c in table.columns]
            data[k] = function(inputs)
        return Column(universe, data, properties.dtype)

    def expression_column(
        self, table: Table, expression: Expression, properties: EvalProperties
    ) -> Column:
        return self.map_column(
            table, lambda values: expression._eval(values), properties
        )

    def async_map_column(
        self, table: Table, function: Callable[..., Value], properties: EvalProperties
    ):
        async def task(key, *values):
            result = await function(*values)
            return (key, result)

        async def run():
            universe = table.universe
            self._validate(universe)
            tasks = []
            for k in universe._keys:
                inputs = (c._data[k] for c in table.columns)
                a_task = asyncio.create_task(task(k, *inputs))
                tasks.append(a_task)
            results = await asyncio.gather(*tasks)
            return Column(universe, dict(results), properties.dtype)

        loop = asyncio.new_event_loop()
        return loop.run_until_complete(run())

    def unsafe_map_column_numba(
        self, table: Table, function: Any, properties: EvalProperties
    ) -> Column:
        universe = table.universe
        self._validate(universe)
        data = {}
        for k in universe._keys:
            inputs = [c._data[k] for c in table.columns]
            # not needed in python, shown for demo purposes
            # function._sig.args holds the dtypes and can be used to construct appropriate
            # casting code
            inputs_cast = [
                t(i) for t, i in zip(function._sig.args, inputs, strict=True)
            ]
            data[k] = function.ctypes(*inputs_cast)
        return Column(universe, data, properties.dtype)

    def filter_universe(self, universe: Universe, column: Column) -> Universe:
        self._validate(universe, column)
        keys = {k for k in universe._keys if column._data[k]}
        return Universe(self, keys)

    def intersect_universe(self, universe: Universe, *universes: Universe) -> Universe:
        keys = universe._keys.intersection(*[u._keys for u in universes])
        return Universe(self, set(keys))

    def union_universe(self, universe: Universe, *universes: Universe) -> Universe:
        keys = universe._keys.union(*[u._keys for u in universes])
        return Universe(self, set(keys))

    def venn_universes(
        self, left_universe: Universe, right_universe: Universe
    ) -> VennUniverses:
        only_left = Universe(self, left_universe._keys - right_universe._keys)
        only_right = Universe(self, right_universe._keys - left_universe._keys)
        both = Universe(self, right_universe._keys & left_universe._keys)
        return VennUniverses(only_left=only_left, only_right=only_right, both=both)

    def reindex_universe(self, column: Column) -> Universe:
        new_universe = set()
        for k in column.universe._keys:
            v = column._data[k]
            assert v not in new_universe
            new_universe.add(v)
        return Universe(self, new_universe)

    def restrict_column(
        self,
        universe: Universe,
        column: Column,
    ) -> Column:
        if column.universe == universe:
            return column
        self._validate(universe, column)
        return Column(universe, column._data, column.dtype)

    def override_column_universe(self, universe: Universe, column: Column):
        assert universe == column.universe
        return Column(universe, column._data, column.dtype)

    def reindex_column(
        self,
        column_to_reindex: Column,
        reindexing_column: Column,
        reindexing_universe: Universe,
    ):
        assert column_to_reindex.universe.issubset(reindexing_column.universe)
        new_data = {}
        for k in reindexing_column.universe._keys:
            v = reindexing_column._data[k]
            assert v not in new_data
            new_data[v] = column_to_reindex._data[k]
        return Column(reindexing_universe, new_data, column_to_reindex.dtype)

    @staticmethod
    def table(universe: Universe, columns: List[Column]):
        scope = universe._scope
        columns = [scope.restrict_column(universe, column) for column in columns]
        return Table(universe, columns)

    # Grouping and joins

    def group_by(
        self, table: Table, requested_columns: List[Column], set_id: bool = False
    ) -> Grouper:
        """
        Args:
            table: a list of columns to group by.
        """

        source_universe = table.universe
        groupby_columns = table.columns

        id_column = None
        if set_id:
            assert len(groupby_columns) == 1
            id_column = groupby_columns[0]

        key_to_inputs = {}
        key_mapping: Dict[int, List[int]] = defaultdict(list)
        for k in source_universe._keys:
            inputs = [c._data[k] for c in groupby_columns]
            new_key = None
            if id_column is None:
                new_key = ref_scalar(*inputs)
            else:
                new_key = id_column._data[k]
            key_to_inputs[new_key] = inputs
            key_mapping[new_key].append(k)

        new_universe = Universe(self, key_to_inputs.keys())
        new_columns_data = [{} for _ in groupby_columns]
        for k, row in key_to_inputs.items():
            for i, col in enumerate(new_columns_data):
                col[k] = row[i]
        dtypes = [col.dtype for col in groupby_columns]
        new_columns = [
            Column(new_universe, data, dtype)
            for data, dtype in zip(new_columns_data, dtypes)
        ]
        ret_table = Scope.table(new_universe, new_columns)

        if not set(requested_columns) <= set(table.columns):
            raise ValueError(
                "requested columns must be a subset of input table columns"
            )

        return Grouper(
            table, requested_columns, ret_table, key_mapping, source_universe
        )

    def ix(
        self,
        keys_column: Column,
        input_universe: Universe,
        strict: bool,
        optional: bool,
    ):
        assert strict or not optional
        return Ixer(
            self,
            keys_column._data,
            keys_column.universe,
            input_universe,
            strict,
            optional,
        )

    def join(
        self,
        left_table: Table,
        right_table: Table,
        assign_id: bool = False,
        left_ear: bool = False,
        right_ear: bool = False,
    ):
        def val_to_idx(table: Table) -> Dict[Tuple, List[int]]:
            universe = table.universe
            key_cols = table.columns
            res: Dict[Tuple, List[int]] = defaultdict(list)
            for id in universe._keys:
                new_key = tuple(col._data[id] for col in key_cols)
                res[new_key].append(id)
            return res

        left_val_to_idx = val_to_idx(left_table)
        right_val_to_idx = val_to_idx(right_table)
        return Joiner(
            self,
            left_table,
            right_table,
            left_val_to_idx,
            right_val_to_idx,
            assign_id,
            left_ear,
            right_ear,
        )

    def join_on_id(self, universes: List[Universe], how: JoinType) -> NJoiner:
        assert len(universes) > 0
        assert how in [JoinType.INNER, JoinType.OUTER]
        keys = universes[0]._keys
        if how == JoinType.INNER:
            for uni in universes:
                keys = keys.intersection(uni._keys)
        else:
            for uni in universes:
                keys = keys.union(uni._keys)
        universe = Universe(self, keys)
        return NJoiner(universe, how)

    # Inspired by https://github.com/frankmcsherry/blog/blob/master/posts/2021-04-26.md
    def _linear_operator(
        self,
        table: Table,
        logic: Callable[[[Value]], [Tuple[BasePointer, Value]]],
        dt: DType,
    ) -> Column:
        universe = table.universe
        self._validate(universe)
        data = {}
        for k in universe._keys:
            inputs = [c._data[k] for c in table.columns]
            for new_key, value in logic(k, inputs):
                data[new_key] = value
        new_universe = Universe(self, data.keys())
        return Column(new_universe, data, dt)

    def flat_map(
        self,
        table: Table,
        function: Callable[[List[Value]], Iterable[Value]],
        dt: DType,
    ) -> Column:
        return self._linear_operator(
            table,
            lambda old_key, row: [
                (ref_scalar(old_key, res, i), res)
                for i, res in enumerate(function(row))
            ],
            dt,
        )

    # Transformers

    def complex_columns(self, inputs: List[ComplexColumn]) -> List[Column]:
        computation = Computation(inputs)
        computation.compute_all_outputs()
        result = computation.fetch_results()
        computation.inputs = None  # disallow later access to our inputs
        return result

    def register_table(table, table_name, column_names):
        ...

    # Updates

    def update_rows(
        self, universe: Universe, column: Column, updates: Column
    ) -> Column:
        """Updates rows of `column`, breaking ties in favor of `updates`"""
        self._validate(column.universe)
        self._validate(updates.universe)
        data = dict(column._data.items())
        data.update(dict(updates._data.items()))
        assert frozenset(data.keys()) == universe._keys
        return Column(universe, data, column.dtype)

    def debug_universe(self, name: str, universe: Universe):
        print(f"[{name}] universe: {universe!r}")
        for key in universe._keys:
            print(f"[{name}] {key}")

    def debug_column(self, name: str, column: Column):
        print(f"[{name}] universe: {column.universe}")
        for key, value in column._data.items():
            print(f"[{name}] {key}: {value}")

    def concat(self, universes: Iterable[Universe]) -> Concat:
        universes = list(universes)
        keys = set()
        for universe in universes:
            if not keys.isdisjoint(universe._keys):
                raise KeyError
            keys |= universe._keys
        universe = Universe(scope=self, keys=keys)
        return Concat(universe=universe, source_universes=universes)

    def flatten(self, flatten_column: Column) -> Flatten:
        new_keys = []
        key_mapping = defaultdict(list)
        data = {}
        source_universe = flatten_column.universe
        dtype = Any
        for k in source_universe._keys:
            for i, a in enumerate(flatten_column._data[k]):
                new_key = ref_scalar(k, i)
                new_keys.append(new_key)
                key_mapping[k].append(new_key)
                data[new_key] = a
                if dtype == Any:
                    dtype = type(a)
                assert dtype == type(a)

        new_universe = Universe(self, new_keys)
        flattened_column = Column(new_universe, data, dtype)
        return Flatten(
            universe=new_universe,
            flattened_column=flattened_column,
            key_mapping=key_mapping,
        )

    def sort(
        self, key_column: Column, instance_column: Column
    ) -> Tuple[Column, Column]:
        universe = key_column.universe
        values = []
        for key in key_column.universe._keys:
            values.append(
                SortingCell(instance_column._data[key], key_column._data[key], key)
            )
        values.sort()
        prev_data = {}
        next_data = {}
        prev_entry = None
        for entry in values:
            if prev_entry is not None and prev_entry.instance == entry.instance:
                prev_data[entry.id_] = prev_entry.id_
                next_data[prev_entry.id_] = entry.id_
            elif prev_entry is not None:
                prev_data[entry.id_] = None
                next_data[prev_entry.id_] = None
            else:
                prev_data[entry.id_] = None
            prev_entry = entry

        if prev_entry is not None:
            next_data[prev_entry.id_] = None

        prev_column = Column(universe, prev_data, Optional[BasePointer])
        next_column = Column(universe, next_data, Optional[BasePointer])
        return prev_column, next_column

    def probe_universe(self, universe: Universe, operator_id: int) -> None:
        pass

    def probe_column(self, column: Column, operator_id: int) -> None:
        pass

    def subscribe_table(
        self, table: Table, on_change: Callable, on_end: Callable
    ) -> None:
        raise NotImplementedError

    def output_table(
        self, table: Table, data_sink: DataStorage, data_format: DataFormat
    ) -> None:
        raise NotImplementedError


class NJoiner:
    def __init__(self, universe: Universe, type: JoinType):
        self.universe = universe
        assert type in [JoinType.INNER, JoinType.OUTER]
        self.type = type

    def select_column(self, column: Column) -> Column:
        def extend_universe(column, universe: Universe) -> Column:
            assert column.universe.issubset(universe)
            data = {k: column._data.get(k, None) for k in universe._keys}
            return Column(universe, data, column.dtype)

        if self.type == JoinType.INNER:
            return self.universe._scope.restrict_column(self.universe, column)
        else:
            return extend_universe(column, self.universe)

    def select_id_column(self) -> UniverseColumn:
        return self.universe.id_column


class Joiner:
    def __init__(
        self,
        scope: Scope,
        left_table: Table,
        right_table: Table,
        left_keys_to_idx: dict[Tuple, list[int]],
        right_keys_to_idx: dict[Tuple, list[int]],
        assign_id: bool,
        left_ear: bool,
        right_ear: bool,
    ):
        self.scope = scope
        self.left_table = left_table
        self.right_table = right_table
        self.left_keys_to_idx = left_keys_to_idx
        self.right_keys_to_idx = right_keys_to_idx
        self.new_id_to_old: dict[int, Tuple[int, int]] = dict()
        self._left_ear = left_ear
        self._right_ear = right_ear
        self._assign_id = assign_id
        self._determine_key_mappings()

    @property
    def universe(self):
        return self._ret_universe

    def _determine_key_mappings(self):
        for l_key, l_idxs in self.left_keys_to_idx.items():
            if l_key not in self.right_keys_to_idx:
                continue
            r_idxs = self.right_keys_to_idx[l_key]
            for li, ri in itertools.product(l_idxs, r_idxs):
                if self._assign_id:
                    new_key = li
                else:
                    new_key = ref_scalar(li, ri)
                if new_key in self.new_id_to_old:
                    raise KeyError("duplicate value")
                self.new_id_to_old[new_key] = (li, ri)

        if self._left_ear:
            for l_key, l_idxs in self.left_keys_to_idx.items():
                if l_key not in self.right_keys_to_idx:
                    for li in l_idxs:
                        if self._assign_id:
                            new_key = li
                        else:
                            new_key = ref_scalar(li, None)
                        if new_key in self.new_id_to_old:
                            raise KeyError("duplicate value")
                        self.new_id_to_old[new_key] = (li, None)
        if self._right_ear:
            for r_key, r_idxs in self.right_keys_to_idx.items():
                if r_key not in self.left_keys_to_idx:
                    for ri in r_idxs:
                        if self._assign_id:
                            raise KeyError
                        else:
                            new_key = ref_scalar(None, ri)
                        if new_key in self.new_id_to_old:
                            raise KeyError("duplicate value")
                        self.new_id_to_old[new_key] = (None, ri)

        if self._assign_id and self._left_ear and not self._right_ear:
            self._ret_universe = self.left_table.universe
        else:
            self._ret_universe = Universe(self.scope, self.new_id_to_old.keys())

    def _select_internal(self, column: Column, side: int) -> Column:
        data = {}
        for new_id, old_ids in self.new_id_to_old.items():
            row_id = old_ids[side]
            data[new_id] = None
            if row_id is not None:
                data[new_id] = column._data[row_id]
        return Column(self.universe, data, column.dtype)

    def select_left_column(self, column: Column) -> Column:
        assert column.universe == self.left_table.universe
        return self._select_internal(column, 0)

    def select_right_column(self, column: Column) -> Column:
        assert column.universe == self.right_table.universe
        return self._select_internal(column, 1)


class Ixer:
    def __init__(
        self,
        scope: Scope,
        key_mapping: dict[int, int],
        output_universe: Universe,
        input_universe: Universe,
        strict: bool,
        optional: bool,
    ):
        self.scope = scope
        self.key_mapping = key_mapping
        self.input_universe = input_universe
        self.optional = optional
        if strict:
            for l_key in self.key_mapping.values():
                if l_key not in self.input_universe._keys:
                    if l_key is None and self.optional:
                        pass
                    else:
                        raise KeyError(f"missing value {l_key}")

            self.universe = output_universe
        else:
            self.universe = Universe(
                scope,
                [
                    key
                    for key, l_key in self.key_mapping.items()
                    if l_key in self.input_universe._keys
                ],
            )

    def ix_column(self, column: Column) -> Column:
        assert column.universe == self.input_universe
        data = {}
        for new_id in self.universe._keys:
            row_id = self.key_mapping[new_id]
            if row_id is None:
                data[new_id] = None
            else:
                data[new_id] = column._data[row_id]
        return Column(self.universe, data, column.dtype)


class Grouper:
    universe: Universe
    source_table: Table
    requested_columns: List[Column]
    ret_table: Table
    key_mapping: Dict[int, List[int]]
    source_universe: Universe

    def __init__(
        self,
        source_table: Table,
        requested_columns: List[Column],
        ret_table: Table,
        key_mapping: Dict[int, List[int]],
        source_universe: Universe,
    ):
        self.universe = ret_table.universe
        self.source_table = source_table
        self.requested_columns = requested_columns
        self.ret_table = ret_table
        self.key_mapping = key_mapping
        self.source_universe = source_universe

    def input_column(self, column: Column) -> Column:
        assert column in self.source_table.columns
        assert column in self.requested_columns
        i = self.source_table.columns.index(column)
        return self.ret_table.columns[i]

    def count_column(self) -> Column:
        data = {k: len(rows) for k, rows in self.key_mapping.items()}
        return Column(self.universe, data, int)

    def reducer_column(self, reducer: Reducer, column: Column) -> Column:
        assert column.universe == self.source_universe
        data = {}
        for k, rows in self.key_mapping.items():
            args = {row: column._data[row] for row in rows}
            data[k] = reducer.apply(args)
        return Column(self.universe, data, reducer.ret_dtype)

    def reducer_ix_column(
        self, reducer: Reducer, ixer: Ixer, input_column: Column
    ) -> Column:
        return self.reducer_column(
            reducer,
            ixer.ix_column(input_column),
        )


@dataclasses.dataclass(frozen=True)
class Concat:
    universe: Universe
    source_universes: List[Universe]

    def concat_column(self, columns: List[Column]):
        assert len(self.source_universes) == len(columns)
        data = {}
        for universe, column in zip(self.source_universes, columns):
            assert column.universe == universe
            data.update(column._data)
        return Column(self.universe, data, columns[0].dtype)


@dataclasses.dataclass(frozen=True)
class Flatten:
    universe: Universe
    flattened_column: Column
    key_mapping: Dict[int, List[int]]

    def get_flattened_column(self) -> Column:
        return self.flattened_column

    def explode_column(self, column: Column) -> Column:
        data = {}
        for old_key, new_keys in self.key_mapping.items():
            old_value = column._data[old_key]
            for new_key in new_keys:
                data[new_key] = old_value
        return Column(self.universe, data, column.dtype)


@dataclasses.dataclass(frozen=True, order=True)
class SortingCell:
    instance: Any
    key: Any
    id_: Any


CapturedTable = Dict[BasePointer, Tuple[Value, ...]]


def run_with_new_graph(
    logic: Callable[[Scope], Iterable[Table]],
    event_loop: asyncio.AbstractEventLoop,
    stats_monitor: Optional[StatsMonitor] = None,
    *,
    ignore_asserts: bool = False,
    monitoring_level: MonitoringLevel = MonitoringLevel.NONE,
    with_http_server: bool = False,
) -> List[CapturedTable]:
    tables = logic(Scope())
    res = []
    for table in tables:
        captured = {}
        for key in table.universe._keys:
            captured[key] = tuple(column._data[key] for column in table.columns)
        res.append(captured)
    return res


def unsafe_make_pointer(arg):
    return BasePointer(arg)


class DataFormat:
    value_fields: Any

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class DataStorage:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class CsvParserSettings:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class AwsS3Settings:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class ValueField:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def set_default(self, *args, **kwargs):
        raise NotImplementedError


class PythonSubject:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
