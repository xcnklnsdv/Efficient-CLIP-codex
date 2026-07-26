"""Compatibility import for the canonical root-level :mod:`dataset_coviar`.

New code must import ``dataset_coviar`` directly.  Replacing this module object
also keeps monkeypatching of CoViAR globals compatible for older callers.
"""

import sys

import dataset_coviar as _dataset_coviar


sys.modules[__name__] = _dataset_coviar
