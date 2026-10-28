"""Interactive HTML report generator for APIMagic test suites."""

import datetime
import html
import json
from typing import List, Optional
from apimagic.core.models import TestSuiteResult

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APIMagic LLM Test Report - {{ suite.suite_name }}</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #1c2128;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --green-bg: rgba(63, 185, 80, 0.15);
            --red: #f85149;
            --red-bg: rgba(248, 81, 73, 0.15);
            --yellow: #d29922;
            --purple: #bc8cff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 32px 20px;
            line-height: 1.5;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }
        .title-group h1 { font-size: 26px; color: #fff; margin-bottom: 4px; }
        .title-group .meta { color: var(--text-muted); font-size: 14px; }
        .kpi-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px 20px;
        }
        .kpi-card .label { font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-card .value { font-size: 28px; font-weight: 700; color: #fff; margin-top: 4px; }
        .kpi-card.pass .value { color: var(--green); }
        .kpi-card.fail .value { color: var(--red); }
        .kpi-card.latency .value { color: var(--accent); }
        .kpi-card.cost .value { color: var(--purple); }

        .toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .filters { display: flex; gap: 8px; }
        .btn-filter {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        .btn-filter.active {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }
        .search-box {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 14px;
            width: 260px;
        }

        .case-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        .case-card:hover { border-color: #58a6ff55; }
        .case-header {
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        .case-left { display: flex; align-items: center; gap: 14px; }
        .badge {
            font-size: 12px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 12px;
            text-transform: uppercase;
        }
        .badge.pass { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
        .badge.fail { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
        .case-name { font-weight: 600; font-size: 16px; color: #fff; }
        .case-tags { display: flex; gap: 6px; }
        .tag { font-size: 11px; background: #21262d; color: var(--text-muted); padding: 2px 8px; border-radius: 4px; }
        .case-stats { display: flex; gap: 20px; color: var(--text-muted); font-size: 13px; font-family: monospace; }

        .case-details {
            border-top: 1px solid var(--border);
            padding: 20px;
            background: #0f141a;
            display: none;
        }
        .case-details.open { display: block; }
        .grid-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .pane {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
        }
        .pane-title {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .code-box {
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 12px;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 280px;
            overflow-y: auto;
            color: #e6edf3;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }
        .metrics-table th, .metrics-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .metrics-table th { color: var(--text-muted); text-transform: uppercase; font-size: 11px; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="title-group">
                <h1>APIMagic LLM Test Report</h1>
                <div class="meta">{{ suite.suite_name }} &bull; Executed at {{ timestamp }}</div>
            </div>
            <div>
                <span class="badge {{ 'pass' if suite.failed_cases == 0 else 'fail' }}" style="font-size: 16px; padding: 6px 18px;">
                    {{ 'ALL PASSED' if suite.failed_cases == 0 else 'FAILED' }}
                </span>
            </div>
        </header>

        <section class="kpi-cards">
            <div class="kpi-card">
                <div class="label">Total Test Cases</div>
                <div class="value">{{ suite.total_cases }}</div>
            </div>
            <div class="kpi-card pass">
                <div class="label">Pass Rate</div>
                <div class="value">{{ "%.1f"|format(suite.pass_rate * 100) }}%</div>
            </div>
            <div class="kpi-card fail">
                <div class="label">Failed Cases</div>
                <div class="value">{{ suite.failed_cases }}</div>
            </div>
            <div class="kpi-card latency">
                <div class="label">P95 Latency</div>
                <div class="value">{{ "%.0f"|format(suite.p95_latency_ms) }} ms</div>
            </div>
            <div class="kpi-card cost">
                <div class="label">Total Cost</div>
                <div class="value">${{ "%.5f"|format(suite.total_cost_usd) }}</div>
            </div>
        </section>

        <div class="toolbar">
            <div class="filters">
                <button class="btn-filter active" onclick="filterCases('all', this)">All ({{ suite.total_cases }})</button>
                <button class="btn-filter" onclick="filterCases('pass', this)">Passed ({{ suite.passed_cases }})</button>
                <button class="btn-filter" onclick="filterCases('fail', this)">Failed ({{ suite.failed_cases }})</button>
            </div>
            <input type="text" id="searchInput" class="search-box" placeholder="Search test cases..." onkeyup="searchCases()">
        </div>

        <section id="casesList">
            {% for case in suite.case_results %}
            <div class="case-card" data-status="{{ 'pass' if case.passed else 'fail' }}" data-name="{{ case.name|lower }}">
                <div class="case-header" onclick="toggleDetails(this)">
                    <div class="case-left">
                        <span class="badge {{ 'pass' if case.passed else 'fail' }}">
                            {{ 'PASS' if case.passed else 'FAIL' }}
                        </span>
                        <div>
                            <span class="case-name">{{ case.name }}</span>
                            <div class="case-tags">
                                {% for tag in case.tags %}
                                <span class="tag">{{ tag }}</span>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                    <div class="case-stats">
                        <span>Latency: <strong>{{ "%.0f"|format(case.avg_latency_ms) }}ms</strong></span>
                        <span>Cost: <strong>${{ "%.5f"|format(case.total_cost_usd) }}</strong></span>
                        <span>Pass Rate: <strong>{{ "%.0f"|format(case.pass_rate * 100) }}%</strong></span>
                    </div>
                </div>

                <div class="case-details">
                    {% set sample_trial = case.trials[0] if case.trials else None %}
                    <div class="grid-split">
                        <div class="pane">
                            <div class="pane-title">Input Prompt / Messages</div>
                            <div class="code-box">{{ sample_prompt(case) }}</div>
                        </div>
                        <div class="pane">
                            <div class="pane-title">Candidate LLM Output</div>
                            <div class="code-box">{% if sample_trial and sample_trial.response %}{{ sample_trial.response.content or '(Empty / Tool Calls)' }}{% if sample_trial.response.tool_calls %}\n\nTool Calls:\n{{ format_json(sample_trial.response.tool_calls) }}{% endif %}{% else %}(No response / Error: {{ case.error }}){% endif %}</div>
                        </div>
                    </div>

                    {% if sample_trial and sample_trial.metrics %}
                    <div class="pane">
                        <div class="pane-title">Evaluations & Assertions</div>
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th>Evaluator</th>
                                    <th>Result</th>
                                    <th>Score</th>
                                    <th>Threshold</th>
                                    <th>Diagnostics / Message</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for m in sample_trial.metrics %}
                                <tr>
                                    <td><strong>{{ m.name }}</strong></td>
                                    <td><span class="badge {{ 'pass' if m.passed else 'fail' }}">{{ 'PASS' if m.passed else 'FAIL' }}</span></td>
                                    <td>{{ "%.2f"|format(m.score) if m.score is not none else '-' }}</td>
                                    <td>{{ m.threshold if m.threshold is not none else '-' }}</td>
                                    <td>{{ m.message }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </section>
    </div>

    <script>
        function toggleDetails(header) {
            const details = header.nextElementSibling;
            details.classList.toggle('open');
        }

        function filterCases(status, btn) {
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cards = document.querySelectorAll('.case-card');
            cards.forEach(card => {
                if (status === 'all' || card.getAttribute('data-status') === status) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function searchCases() {
            const query = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.case-card');
            cards.forEach(card => {
                const name = card.getAttribute('data-name');
                if (name.includes(query)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""


class HTMLReporter:
    """Generates standalone interactive HTML test reports."""

    def __init__(self, output_path: str = "apimagic_report.html"):
        self.output_path = output_path

    def _sample_prompt(self, case) -> str:
        # Helper to extract prompt text
        if case.trials and case.trials[0].response:
            # Try raw request or fallback to case name
            return f"Case ID: {case.id}\nTags: {', '.join(case.tags)}"
        return f"Case ID: {case.id}"

    def generate(self, suite: TestSuiteResult) -> str:
        """Generates and writes HTML report file."""
        from jinja2 import Template

        template = Template(HTML_TEMPLATE)
        rendered = template.render(
            suite=suite,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sample_prompt=lambda c: self._sample_prompt(c),
            format_json=lambda obj: json.dumps([o.model_dump() if hasattr(o, "model_dump") else o for o in obj], indent=2),
        )

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        return self.output_path
