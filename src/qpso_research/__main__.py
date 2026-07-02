"""
Allow running the package directly: ``python -m qpso_research``.

This is the most reliable way to invoke Python code on Windows — it
works even when the Microsoft Store app-execution alias is active,
as long as ``python`` resolves to *any* Python interpreter (including
via the ``py`` launcher: ``py -m qpso_research``).

Subcommands:
  python -m qpso_research compare --benchmark sphere --dims 2
  python -m qpso_research study --config configs/default.yaml
"""

import sys

from .cli import run_comparison, run_full_study


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m qpso_research <command> [options]")
        print()
        print("Commands:")
        print("  compare   Run a single PSO vs QPSO benchmark comparison")
        print("  study     Run the full multi-benchmark publication study")
        print()
        print("Examples:")
        print("  python -m qpso_research compare --benchmark sphere --dims 2")
        print("  python -m qpso_research study --config configs/default.yaml")
        print()
        print("Equivalent installed commands (no 'python' needed):")
        print("  qpso-compare --benchmark sphere --dims 2")
        print("  qpso-study --config configs/default.yaml")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in the CLI functions
    # sees only its own arguments.
    sys.argv = [f"qpso_research {command}"] + sys.argv[2:]

    if command == "compare":
        run_comparison()
    elif command == "study":
        run_full_study()
    else:
        print(f"Unknown command: '{command}'")
        print("Use 'compare' or 'study'. Run with --help for details.")
        sys.exit(1)


main()
