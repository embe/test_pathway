# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

from pathway.internals import parse_graph
from pathway.internals.monitoring import MonitoringLevel
from pathway.internals.runtime_type_check import runtime_type_check
from pathway.internals.rustpy_builder import RustpyBuilder


@runtime_type_check
def run(
    debug: bool = False,
    monitoring_level: MonitoringLevel = MonitoringLevel.AUTO,
    with_http_server: bool = False,
    default_logging: bool = True,
):
    """Runs the computation graph.

    Args:
        debug: enable output out of table.debug() operators
        monitoring_level: the verbosity of stats monitoring mechanism. One of
            pathway.MonitoringLevel.NONE, pathway.MonitoringLevel.IN_OUT,
            pathway.MonitoringLevel.ALL. If unset, pathway will choose between
            NONE and IN_OUT based on output interactivity.
        with_http_server: whether to start a http server with runtime metrics. Learn
            more in a `tutorial </developers/tutorials/prometheus-monitoring/>`_ .
        default_logging: whether to allow pathway to set its own logging handler. Set
            it to False if you want to set your own logging handler.
    """
    RustpyBuilder(
        parse_graph.G,
        debug=debug,
        monitoring_level=monitoring_level,
        with_http_server=with_http_server,
        default_logging=default_logging,
    ).run_outputs()


@runtime_type_check
def run_all(
    debug: bool = False,
    monitoring_level: MonitoringLevel = MonitoringLevel.AUTO,
    with_http_server: bool = False,
    default_logging: bool = True,
):
    RustpyBuilder(
        parse_graph.G,
        debug=debug,
        monitoring_level=monitoring_level,
        with_http_server=with_http_server,
        default_logging=default_logging,
    ).run_all()
