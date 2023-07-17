# Copyright (c) 2023 NavAlgo
#
# Proprietary and confidential.

from __future__ import annotations

import pytest

from pathway.internals import parse_graph


@pytest.fixture(autouse=True)
def parse_graph_teardown():
    yield
    parse_graph.G.clear()
