#!/usr/bin/env python

import pathway.engine as api  # type: ignore
from pathway.engine import PathwayType


def main(g):
    table = g.connector_table(
        data_source=api.DataStorage(
            storage_type="csv",
            path="tests/data/sql_injection.txt",
            csv_parser_settings=api.CsvParserSettings(
                delimiter="+",
            ),
        ),
        data_format=api.DataFormat(
            format_type="dsv",
            key_field_names=["key"],
            value_fields=[
                api.ValueField("key", PathwayType.ANY),
                api.ValueField("foo", PathwayType.ANY),
                api.ValueField("foofoo", PathwayType.ANY),
            ],
            delimiter="+",
        ),
    )
    g.output_table(
        table=table,
        data_sink=api.DataStorage(
            storage_type="postgres",
            connection_string="host=localhost port=5432 dbname=template1 user=sergey password=sergey",
        ),
        data_format=api.DataFormat(
            format_type="sql",
            key_field_names=[],
            value_fields=[
                api.ValueField("key", PathwayType.ANY),
                api.ValueField("foo", PathwayType.ANY),
                api.ValueField("foofoo", PathwayType.ANY),
            ],
            table_name="test_incremental_updates",
        ),
    )


if __name__ == "__main__":
    api.run_with_new_graph(main)
