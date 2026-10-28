"""Rich console reporter for APIMagic test execution."""

import sys
from typing import Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from apimagic.core.models import TestCaseResult, TestSuiteResult


class ConsoleReporter:
    """Renders test execution results to the terminal with Rich."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_case_result(self, result: TestCaseResult) -> None:
        """Prints a single test case line as it completes."""
        if result.passed:
            status_text = Text("[PASS]", style="bold green")
        else:
            status_text = Text("[FAIL]", style="bold red")

        name_text = Text(f" {result.name:<35}", style="bold white")
        lat_text = Text(f"{result.avg_latency_ms:>6.0f}ms", style="cyan")
        cost_text = Text(f"${result.total_cost_usd:.5f}", style="magenta")
        rate_text = Text(f"({result.pass_rate * 100:.0f}% pass)", style="yellow" if result.pass_rate < 1.0 else "green")

        self.console.print(status_text, name_text, lat_text, cost_text, rate_text)

        if not result.passed:
            for trial in result.trials:
                for m in trial.metrics:
                    if not m.passed:
                        self.console.print(f"    └─ [red]{m.name}[/red]: {m.message}")
            if result.error:
                self.console.print(f"    └─ [red]Error[/red]: {result.error}")

    def print_suite_summary(self, suite: TestSuiteResult) -> None:
        """Prints a comprehensive summary table for the test suite."""
        self.console.print()

        table = Table(title=f"Test Suite: {suite.suite_name}", border_style="bright_blue")
        table.add_column("Case ID", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Trials", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("P95 Latency", justify="right")
        table.add_column("Cost (USD)", justify="right")
        table.add_column("Evaluations", justify="left")

        for r in suite.case_results:
            status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
            metrics_summary = []
            for m_name, m_data in r.metrics_summary.items():
                p = m_data["passed"]
                t = m_data["total"]
                color = "green" if p == t else "red"
                metrics_summary.append(f"[{color}]{m_name} ({p}/{t})[/{color}]")

            table.add_row(
                r.id,
                status,
                f"{r.pass_rate * 100:.0f}% ({len(r.trials)})",
                f"{r.avg_latency_ms:.0f} ms",
                f"{r.p95_latency_ms:.0f} ms",
                f"${r.total_cost_usd:.5f}",
                ", ".join(metrics_summary) if metrics_summary else "-",
            )

        self.console.print(table)

        # Overview Panel
        pass_color = "green" if suite.failed_cases == 0 else "red"
        kpi_text = (
            f"Total: {suite.total_cases}  |  "
            f"[{pass_color}]Passed: {suite.passed_cases}[/{pass_color}]  |  "
            f"[red]Failed: {suite.failed_cases}[/red]  |  "
            f"Pass Rate: [bold {pass_color}]{suite.pass_rate * 100:.1f}%[/bold {pass_color}]  |  "
            f"Duration: {suite.duration_ms / 1000.0:.2f}s  |  "
            f"Avg Latency: {suite.avg_latency_ms:.0f}ms  |  "
            f"P95: {suite.p95_latency_ms:.0f}ms  |  "
            f"Total Cost: [magenta]${suite.total_cost_usd:.5f}[/magenta]"
        )

        self.console.print(Panel(kpi_text, title="Suite Execution Summary", border_style="cyan"))
