#!/usr/bin/env python

import pathway as pw


def main():
    t = pw.debug.parse_to_table(
        "age owner pet \n 1 10 Alice dog \n 2 9 Bob cat \n 3 8 Alice cat"
    )
    pw.io.csv.write(t, "table.csv")

    pw.run()


if __name__ == "__main__":
    main()
