"""Reporters package for APIMagic."""

from apimagic.reporters.console import ConsoleReporter
from apimagic.reporters.html import HTMLReporter
from apimagic.reporters.json_report import JSONReporter, JUnitReporter

__all__ = [
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
    "JUnitReporter",
]
