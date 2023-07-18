import argparse
import os

import pathway as pw
from pathway.internals.monitoring import MonitoringLevel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pathway wordcount program")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--pstorage", type=str)
    parser.add_argument("--n-cpus", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    os.environ["PATHWAY_PERSISTENT_STORAGE"] = args.pstorage
    os.environ["PATHWAY_THREADS"] = str(args.n_cpus)

    class InputSchema(pw.Schema):
        word: str

    words = pw.io.fs.read(
        path=args.input,
        schema=InputSchema,
        format="json",
        mode=args.mode,
        persistent_id=1,
        autocommit_duration_ms=10,
    )
    result = words.groupby(words.word).reduce(
        words.word,
        count=pw.reducers.count(),
    )
    pw.io.csv.write(result, args.output)
    pw.run(monitoring_level=MonitoringLevel.NONE)
