from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TTLTier(str,Enum):
    FACTUAL = "factual"
    STANDARD = "standard"
    VOLATILE = "volatile"
    SKIP = "skip"

