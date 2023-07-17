import os

import pathway as pw

if __name__ == "__main__":
    table = pw.io.kafka.read_from_upstash(
        endpoint="measured-octopus-5411-eu1-kafka.upstash.io:9092",
        topic="test_0",
        username=os.environ["KAFKA_USERNAME"],
        password=os.environ["KAFKA_PASSWORD"],
    )

    pw.io.csv.write(table, "upstash_output.csv")
    pw.run()
