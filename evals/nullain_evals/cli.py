"""Nullain Agent SDK evals — command-line entry point.

    uv run python -m nullain_evals.cli offline
    uv run python -m nullain_evals.cli live --provider ollama --model glm-5.2:cloud
    uv run python -m nullain_evals.cli live --provider ollama --model glm-5.2:cloud --save-fixtures

``offline`` never touches the network. ``live`` requires
``OLLAMA_API_KEY``/``NULLAIN_OLLAMA_API_KEY`` to be set (same resolution
``OllamaCloudProvider``/``load_settings`` already use) and calls the real
API — opt-in only, matching ``make evals-live``'s naming.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nullain.config import load_settings
from nullain.llm import OllamaCloudProvider

from nullain_evals.report import EvalReport
from nullain_evals.runner import run_live, run_offline
from nullain_evals.tasks import ALL_TASKS

REPORT_PATH = Path(__file__).resolve().parent.parent / "report.json"


def _print_summary(report: EvalReport) -> None:
    print(f"\n{report.mode} eval report — provider={report.provider} model={report.model}")
    print(f"{'task_id':<35} {'result':<6} {'steps':>5} {'time':>7}  reason")
    for r in report.results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"{r.task_id:<35} {marker:<6} {r.steps:>5} {r.wall_time_seconds:>6.1f}s  {r.reason}")
    print(f"\n{report.pass_count}/{report.total_count} passed ({report.pass_rate:.0%})")


async def _cmd_offline(args: argparse.Namespace) -> int:
    report = await run_offline(ALL_TASKS)
    _print_summary(report)
    REPORT_PATH.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    return 0 if not args.gate else (0 if report.pass_count == report.total_count else 1)


async def _cmd_live(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.ollama_api_key:
        print(
            "error: no Ollama Cloud API key found (set OLLAMA_API_KEY or configure nullain.toml)",
            file=sys.stderr,
        )
        return 2
    provider = OllamaCloudProvider(
        api_key=settings.ollama_api_key, base_url=settings.ollama_base_url
    )
    report = await run_live(
        ALL_TASKS,
        provider=provider,
        provider_name="ollama",
        model=args.model,
        save_fixtures=args.save_fixtures,
    )
    _print_summary(report)
    REPORT_PATH.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    if args.save_fixtures:
        print("Fixtures updated for every passing task.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nullain-evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_offline = sub.add_parser("offline", help="Run the suite against recorded fixtures")
    p_offline.add_argument(
        "--gate",
        action="store_true",
        help="Exit nonzero if any task fails (default: informational, always exit 0)",
    )
    p_offline.set_defaults(handler=_cmd_offline)

    p_live = sub.add_parser("live", help="Run the suite against a real provider")
    p_live.add_argument("--provider", default="ollama", choices=["ollama"])
    p_live.add_argument("--model", required=True)
    p_live.add_argument(
        "--save-fixtures",
        action="store_true",
        help="Record passing tasks' responses as new offline fixtures",
    )
    p_live.set_defaults(handler=_cmd_live)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
