# Copyright (c) 2023 NavAlgo
#
# Proprietary and confidential.

import os

engine_version = os.environ.get("PATHWAY_ENGINE", "rust").lower()
if engine_version not in ["python", "rust"]:
    raise ValueError("variable PATHWAY_ENGINE must be either 'rust' or 'python'")


ignore_asserts = os.environ.get("PATHWAY_IGNORE_ASSERTS", "false").lower() in (
    "1",
    "true",
    "yes",
)
