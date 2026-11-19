"""Run the theory 4.9 mandatory mutation suite (review m5).

    python scripts/run_mutation_suite.py [--only 1,3] [--output result.json]

For every executable entry of ``tests/mutation_suite.py``:

1. copy ``src/`` into a fresh temporary directory;
2. run the killing tests against the unmutated copy, which must pass (the
   control);
3. apply the exact textual mutation, which must occur once, and run the
   killing tests again.  The mutation is killed only if pytest reports test
   failures (exit code 1).  Collection or usage errors are reported as runner
   errors, never as kills.

``obligation-open`` entries are reported as gaps.  The theory's required kill
rate is 100%; the summary states the measured rate and lists every gap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mutation_suite import MUTATIONS, Mutation  # noqa: E402


def _pytest(source_root: Path, killers: tuple[str, ...]) -> tuple[int, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "src"), str(ROOT / "tests"))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        "pytest",
        "-q",
        "-x",
        "-p",
        "no:cacheprovider",
        *killers,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    output = (completed.stdout + completed.stderr).strip()
    tail = "\n".join(output.splitlines()[-6:])
    return completed.returncode, output, tail


def _vacuous(output: str) -> bool:
    """A killing test that skips or xfails proves nothing."""

    summary = output.splitlines()[-1] if output else ""
    return any(word in summary for word in ("skipped", "xfailed", "xpassed", "deselected"))


def run(mutation: Mutation) -> dict[str, object]:
    record: dict[str, object] = {
        "number": mutation.number,
        "defect": mutation.defect,
        "obligations": mutation.obligations,
        "status": mutation.status,
    }
    if mutation.status == "obligation-open":
        record["verdict"] = "gap: killing obligation not implemented"
        record["note"] = mutation.note
        return record
    if mutation.status != "executable":
        record["verdict"] = f"not run: {mutation.status}"
        record["note"] = mutation.note
        return record
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"mutation-{mutation.number}-") as directory:
        workspace = Path(directory)
        shutil.copytree(
            ROOT / "src", workspace / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        control_code, control_output, control_tail = _pytest(workspace, mutation.killers)
        record["control_exit_code"] = control_code
        if control_code != 0:
            record["verdict"] = "error: killing tests fail without the mutation"
            record["control_output"] = control_tail
            return record
        if _vacuous(control_output):
            record["verdict"] = "error: a killing test was skipped or xfailed in the control"
            record["control_output"] = control_tail
            return record
        target = workspace / mutation.path
        source = target.read_text()
        if source.count(mutation.find) != 1:
            record["verdict"] = "error: mutation snippet does not occur exactly once"
            return record
        target.write_text(source.replace(mutation.find, mutation.replace))
        code, _, tail = _pytest(workspace, mutation.killers)
    record["mutant_exit_code"] = code
    record["mutant_output"] = tail
    record["seconds"] = round(time.perf_counter() - started, 1)
    if code == 1:
        record["verdict"] = "killed"
    elif code == 0:
        record["verdict"] = "SURVIVED"
    else:
        record["verdict"] = f"error: pytest exit code {code}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated mutation numbers")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = MUTATIONS
    if args.only:
        wanted = {int(value) for value in args.only.split(",")}
        selected = tuple(mutation for mutation in MUTATIONS if mutation.number in wanted)
    results = []
    for mutation in selected:
        result = run(mutation)
        results.append(result)
        print(f"{mutation.number:2d} {result['verdict']}: {mutation.defect}", flush=True)
    executable = [r for r in results if r["status"] == "executable"]
    killed = sum(r["verdict"] == "killed" for r in executable)
    summary = {
        "entries": len(results),
        "executable": len(executable),
        "killed": killed,
        "gaps": [r["number"] for r in results if r["status"] == "obligation-open"],
        "blocked": [
            r["number"]
            for r in results
            if r["status"] not in ("executable", "obligation-open")
        ],
        "not_killed": [r["number"] for r in executable if r["verdict"] != "killed"],
    }
    print(json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps({"summary": summary, "results": results}, indent=1) + "\n"
        )


if __name__ == "__main__":
    main()
