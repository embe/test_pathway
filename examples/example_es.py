#!/usr/bin/env python

import pathway as pw


def main():
    table = pw.io.csv.read(
        "tests/data/minimal_multicolumn.txt", ["key"], ["foo", "foofoo"]
    )
    pw.io.elasticsearch.write(
        table,
        "http://localhost:9200",
        auth=pw.io.elasticsearch.ElasticSearchAuth.basic("elastic", "changeme"),
        index_name="tweets",
    )
    pw.run()


if __name__ == "__main__":
    main()
