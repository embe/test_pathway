#!/usr/bin/env python

import pathway as pw
from pathway.internals import api, datasource
from pathway.internals._io_helpers import _form_value_fields
from pathway.internals.decorators import table_from_datasource


def main():
    data_storage = api.DataStorage(
        storage_type="fs",
        path="tests/data/json_complex_paths.txt",
        poll_new_objects=False,
    )
    data_format = api.DataFormat(
        format_type="jsonlines",
        key_field_names=None,
        value_fields=_form_value_fields([], ["pet_name", "pet_height"], None, None),
        column_paths={"pet_name": "/pet/name", "pet_height": "/pet/measurements/1"},
    )
    table = table_from_datasource(
        datasource.GenericDataSource(
            data_storage,
            data_format,
            None,
        )
    )
    pw.debug.compute_and_print(table)


if __name__ == "__main__":
    main()
