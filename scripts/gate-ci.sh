#!/usr/bin/env bash
# Runs the F12/F15 budget gates in CI (ticket #94).
#
# `bench --gate <budget> --gate-report <report>` exits:
#   0  every row measured and within budget
#   1  a measured row broke its budget            -> the job fails
#   2  a row could not be measured (missing GGUF, or a target nobody has
#      instrumented yet)                          -> reported, job stays green
#
# Without a GGUF the gate reads `gates/no-measurement.json`, an explicitly
# unweighed report: every row comes back `skipped`, never `passed`. A CI runner
# therefore verifies the mechanism without inventing a number. The real
# measurement is #95, on a Mac with the model on disk.
#
# Because "everything skipped" is exit 2 and stays green, this script also runs a
# self-check (`--self-check`, the default): `gates/self-check-violation.json` is a
# weighed synthetic report that breaks fase1's bandwidth floor, so the gate MUST
# exit 1 on it. A gate that cannot fail is not a gate (spec #82 B2).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BENCH="./target/debug/bench"
MODEL="${KANSO_GGUF:-models/smollm2-360m-instruct-q8_0.gguf}"
REPORTS="$(mktemp -d)"
trap 'rm -rf "$REPORTS"' EXIT

status=0

# Self-check first: prove the gate can go red before trusting its green.
if [[ "${1:-}" == "--self-check" || "${SELF_CHECK:-1}" == "1" ]]; then
  echo "== self-check: a weighed report that breaks fase1 must exit 1"
  "$BENCH" --gate gates/self-check-budget.json --gate-report gates/self-check-violation.json
  code=$?
  if [[ "$code" -ne 1 ]]; then
    echo "::error::the gate did not fail on a report that breaks its budget (exit $code, want 1)"
    echo "   a gate that cannot go red is not a gate (spec #82 B2)"
    status=1
  else
    echo "   verdict: VIOLATED as required — the gate can fail"
  fi

  # And the same budget on the unweighed report must skip (2), never pass: the
  # self-check distinguishes "no evidence" from "evidence that fails".
  "$BENCH" --gate gates/self-check-budget.json --gate-report gates/no-measurement.json \
    > /dev/null
  code=$?
  if [[ "$code" -ne 2 ]]; then
    echo "::error::an unweighed report must exit 2, got $code (want 2)"
    status=1
  fi
fi

for budget in gates/fase1.json gates/fase2.json; do
  name="$(basename "$budget" .json)"

  if [[ -f "$MODEL" ]]; then
    echo "== $budget: measuring on $MODEL"
    # One invocation, two batch sizes: `batch_scale_ratio` reads last/first.
    "$BENCH" \
      --model "$MODEL" \
      --backend cpu \
      --batch-sizes 1,8 \
      --json-output "$REPORTS/$name.json" \
      --no-warmup || echo "::warning::bench run for $budget exited non-zero"
    evidence="$REPORTS/$name.json"
    if [[ ! -f "$evidence" ]]; then
      echo "::warning::$budget produced no report; its rows will be skipped"
      evidence="gates/no-measurement.json"
    fi
  else
    echo "== $budget: no GGUF at $MODEL — asset rows will report SKIPPED, not passed"
    evidence="gates/no-measurement.json"
  fi

  "$BENCH" --gate "$budget" --gate-report "$evidence"
  code=$?
  case "$code" in
    0) echo "   verdict: passed" ;;
    1)
      echo "   verdict: VIOLATED — a measured row broke its budget"
      status=1
      ;;
    2) echo "   verdict: skipped — nothing was measured (see the rows above)" ;;
    *)
      echo "   verdict: the gate itself failed (exit $code)"
      status=1
      ;;
  esac
done

exit "$status"
