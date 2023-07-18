#!/usr/bin/env python

import os

import pathway as pw
from pathway.internals import parse_graph
from pathway.internals.rustpy_builder import RustpyBuilder


def main():
    table = pw.io.s3_csv.read(
        path="testfile",
        aws_s3_settings=pw.io.s3_csv.AwsS3Settings(
            bucket_name="ovh-s3-test",
            region="rbx",
            endpoint="s3.rbx.io.cloud.ovh.net",
            access_key=os.environ["OVH_S3_ACCESS_KEY"],
            secret_access_key=os.environ["OVH_S3_SECRET_ACCESS_KEY"],
        ),
        id_columns=["seq_id"],
        value_columns=["key", "value"],
        mode="static",
    )
    pw.io.csv.write(table, "table.csv")
    RustpyBuilder(parse_graph.G).run_outputs()


if __name__ == "__main__":
    main()
