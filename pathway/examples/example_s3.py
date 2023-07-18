#!/usr/bin/env python

import os

import pathway as pw
from pathway.internals import parse_graph
from pathway.internals.rustpy_builder import RustpyBuilder


def main():
    table = pw.io.s3_csv.read(
        path="CsvS3/",
        aws_s3_settings=pw.io.s3_csv.AwsS3Settings(
            bucket_name="zxqfd555-test-bucket",
            region="eu-west-3",
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        ),
        id_columns=["key"],
        value_columns=["key", "value", "some_different_value"],
        poll_new_objects=False,
    )

    pw.io.csv.write(table, "table.csv")

    RustpyBuilder(parse_graph.G).run_outputs()


if __name__ == "__main__":
    main()
