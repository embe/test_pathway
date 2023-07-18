# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

from __future__ import annotations

import datetime


def truncate_to_minutes(time: datetime.datetime) -> datetime.datetime:
    return time - datetime.timedelta(seconds=time.second, microseconds=time.microsecond)
