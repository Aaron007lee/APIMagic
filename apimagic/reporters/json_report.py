"""JSON and JUnit XML reporters for CI/CD pipelines."""

import json
from typing import Optional
from xml.etree import ElementTree as ET

from apimagic.core.models import TestSuiteResult


class JSONReporter:
    """Exports test suite results to JSON file."""

    def __init__(self, output_path: str = "apimagic_report.json"):
        self.output_path = output_path

    def generate(self, suite: TestSuiteResult) -> str:
        """Serializes the test suite result to JSON."""
        data = suite.model_dump()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return self.output_path


class JUnitReporter:
    """Exports test suite results to standard JUnit XML format."""

    def __init__(self, output_path: str = "junit_report.xml"):
        self.output_path = output_path

    def generate(self, suite: TestSuiteResult) -> str:
        """Serializes the test suite result to JUnit XML."""
        root = ET.Element(
            "testsuite",
            name=suite.suite_name,
            tests=str(suite.total_cases),
            failures=str(suite.failed_cases),
            errors="0",
            time=f"{suite.duration_ms / 1000.0:.3f}",
        )

        for case in suite.case_results:
            tc_elem = ET.SubElement(
                root,
                "testcase",
                name=case.name,
                classname=f"{suite.suite_name}.{case.id}",
                time=f"{case.avg_latency_ms / 1000.0:.3f}",
            )
            if not case.passed:
                fail_messages = []
                for t in case.trials:
                    for m in t.metrics:
                        if not m.passed:
                            fail_messages.append(f"[{m.name}] {m.message}")
                if case.error:
                    fail_messages.append(f"[Error] {case.error}")

                fail_elem = ET.SubElement(
                    tc_elem,
                    "failure",
                    message="; ".join(fail_messages),
                    type="AssertionError",
                )
                fail_elem.text = "\n".join(fail_messages)

        tree = ET.ElementTree(root)
        tree.write(self.output_path, encoding="utf-8", xml_declaration=True)
        return self.output_path
