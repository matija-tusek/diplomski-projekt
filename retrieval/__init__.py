"""Grupa 3: retrieval. Za kandidata vraća top k semantički najsličnijih oglasa."""

from .index import JobIndex
from .retrieve import retrieve
from .text_builder import candidate_text, job_text

__all__ = ["JobIndex", "retrieve", "candidate_text", "job_text"]
