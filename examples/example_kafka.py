import argparse
import json
import os

from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import TopicPartition


def run_as_consumer(output_type):
    import pathway as pw

    rdkafka_settings = {
        "bootstrap.servers": "clean-panther-8776-eu1-kafka.upstash.io:9092",
        "security.protocol": "sasl_ssl",
        "sasl.mechanism": "SCRAM-SHA-256",
        "group.id": "$GROUP_NAME",
        "session.timeout.ms": "60000",
        "sasl.username": os.environ["KAFKA_USERNAME"],
        "sasl.password": os.environ["KAFKA_PASSWORD"],
        "enable.ssl.certificate.verification": "false",
    }

    t = pw.io.kafka.read(
        rdkafka_settings,
        topic="test_0",
        format="json",
        value_columns=["owner", "pet"],
    )

    t.debug("t")
    if output_type == "kafka":
        output_rdkafka_settings = {
            "bootstrap.servers": "clean-panther-8776-eu1-kafka.upstash.io:9092",
            "security.protocol": "sasl_ssl",
            "sasl.mechanism": "SCRAM-SHA-256",
            "sasl.username": os.environ["KAFKA_USERNAME"],
            "sasl.password": os.environ["KAFKA_PASSWORD"],
            "enable.ssl.certificate.verification": "false",
        }
        pw.io.kafka.write(
            t,
            output_rdkafka_settings,
            "test_1",
            commit_frequency_in_messages=10**5,
            commit_frequency_ms=100000,
        )

    pw.run()


def run_as_producer():
    producer = KafkaProducer(
        bootstrap_servers=["measured-octopus-5411-eu1-kafka.upstash.io:9092"],
        sasl_mechanism="SCRAM-SHA-256",
        security_protocol="SASL_SSL",
        sasl_plain_username=os.environ["KAFKA_USERNAME"],
        sasl_plain_password=os.environ["KAFKA_PASSWORD"],
        api_version=(0, 10, 1),
    )

    test_message = {
        "owner": "Sergey",
        "pet": "Bear",
    }
    producer.send("test_0", json.dumps(test_message).encode("utf-8"))
    producer.send("test_0", "*COMMIT*".encode("utf-8"))
    producer.close()


def run_as_listener():
    consumer = KafkaConsumer(
        "test_1",
        bootstrap_servers=["clean-panther-8776-eu1-kafka.upstash.io:9092"],
        sasl_mechanism="SCRAM-SHA-256",
        security_protocol="SASL_SSL",
        sasl_plain_username=os.environ["KAFKA_USERNAME"],
        sasl_plain_password=os.environ["KAFKA_PASSWORD"],
        api_version=(0, 10, 1),
    )

    partitions = consumer.partitions_for_topic("test_1")
    for p in partitions:
        topic_partition = TopicPartition("test_1", p)
        # Seek offset 0
        consumer.seek(partition=topic_partition, offset=0)
        for msg in consumer:
            print(msg.value.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["producer", "consumer", "listener"]
    )
    parser.add_argument(
        "--output-type", type=str, required=False, choices=["kafka", "debug"]
    )
    args = parser.parse_args()

    if args.mode == "consumer":
        run_as_consumer(args.output_type)
    elif args.mode == "producer":
        run_as_producer()
    elif args.mode == "listener":
        run_as_listener()
