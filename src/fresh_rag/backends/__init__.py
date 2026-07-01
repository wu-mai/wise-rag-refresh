"""Pluggable document backends for the Freshness-Aware RAG pipeline."""
from .base import Backend
from .simulated import SimulatedBackend
from .nq import NQBackend
from .timevarying import TimeVaryingBackend

__all__ = ["Backend", "SimulatedBackend", "NQBackend", "TimeVaryingBackend"]
