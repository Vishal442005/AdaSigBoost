"""Physical and differentiable signal descriptor extractors."""

from .differentiable_stats import extract_differentiable_stats
from .time_freq import DESCRIPTOR_NAMES, extract_descriptors

__all__ = [
    "DESCRIPTOR_NAMES",
    "extract_descriptors",
    "extract_differentiable_stats",
]
