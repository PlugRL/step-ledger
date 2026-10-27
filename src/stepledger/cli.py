"""Command line: ``stepledger compare ENV_DIR TRAINER_FILE``.

Exit codes: 0 no differences, 1 differences found, 2 could not compare.
"""

from __future__ import annotations

import argparse
import sys

from stepledger._version import __version__
from stepledger.compare import compare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stepledger", description="Compare env-side and trainer-side step ledgers.")
    parser.add_argument("--version", action="version", version=f"stepledger {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser(
        "compare",
        help="find the first difference between the two sides, per env",
        description="Exit code 0: no differences. 1: differences found. 2: could not compare.",
    )
    cmd.add_argument("env_dir", help="directory holding the env-*.npz ledgers written by LedgerEnv")
    cmd.add_argument("trainer_file", help="ledger written by LedgerVectorEnv or sb3.LedgerVecEnv")
    args = parser.parse_args(argv)

    try:
        report = compare(args.env_dir, args.trainer_file)
    except (OSError, ValueError, KeyError) as exc:
        print(f"stepledger: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
