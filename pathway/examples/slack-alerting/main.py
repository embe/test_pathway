#!/usr/bin/env python

import os

import requests

import pathway as pw

SLACK_CHANNEL_ID = os.environ["SLACK_ALERT_CHANNEL_ID"]
SLACK_TOKEN = os.environ["SLACK_ALERT_TOKEN"]


if __name__ == "__main__":
    alerts = pw.io.csv.read(
        "alerts.txt", id_columns=["alert_id"], value_columns=["meta"]
    )

    def on_alert_event(key, row, time, is_addition):
        alert_message = "Alert '{}' changed state to {}".format(
            row["alert_id"],
            "ACTIVE" if is_addition else "INACTIVE",
        )
        requests.post(
            "https://slack.com/api/chat.postMessage",
            data="text={}&channel={}".format(alert_message, SLACK_CHANNEL_ID),
            headers={
                "Authorization": "Bearer {}".format(SLACK_TOKEN),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        ).raise_for_status()

    pw.io.subscribe(alerts, on_alert_event)

    pw.run()
