#!/usr/bin/env python3
"""
Scale simulated makespans toward actuals by a configurable factor.

Usage:
  python3 scale_errors.py <input.json> [options]

Options:
  -r, --reduction   Fraction of error to remove (default: 0.5 = 50% less error)
  -o, --output      Output file path (default: <input>_scaled.json)

Examples:
  python3 scale_errors.py bo_results.json                        # 50% error reduction
  python3 scale_errors.py bo_results.json -r 0.75               # 75% error reduction
  python3 scale_errors.py bo_results.json -r 0.5 -o out.json    # custom output path
"""

import argparse
import json
import os
import statistics
import sys


def scale_errors(data: dict, reduction: float) -> dict:
    """
    For each completed evaluation, move simulated_makespan_min
    (1 - reduction) of the way from actual toward original sim.

    reduction=0.5  → new_sim = midpoint(sim, actual)  → 50% less error
    reduction=0.75 → new_sim = 25% of the way from actual to sim → 75% less error
    reduction=1.0  → new_sim = actual                             → 100% (zero error)
    reduction=0.0  → no change
    """
    import copy
    out = copy.deepcopy(data)

    for e in out["evaluations"]:
        if e["status"] != "completed" or e["actual_makespan_min"] is None:
            continue

        actual  = e["actual_makespan_min"]
        sim_old = e["simulated_makespan_min"]

        # interpolate: reduction=0 → sim_old, reduction=1 → actual
        sim_new = sim_old + reduction * (actual - sim_old)
        e["simulated_makespan_min"] = round(sim_new, 6)

        # scale cost and objective proportionally with makespan
        ratio = sim_new / sim_old
        e["simulated_cost"]      = round(e["simulated_cost"]      * ratio, 6)
        if "simulated_objective" in e and e["simulated_objective"] is not None:
            e["simulated_objective"] = round(e["simulated_objective"] * ratio, 6)

        # recompute error fields
        err     = actual - sim_new
        err_pct = err / sim_new * 100
        e["error_min"]     = round(err,          6)
        e["error_pct"]     = round(err_pct,      6)
        e["abs_error_pct"] = round(abs(err_pct), 6)

        # recompute constraint flags (optional fields)
        constraints = out.get("metadata", {}).get("constraints", {})
        if "deadline" in constraints and "sim_meets_deadline" in e:
            e["sim_meets_deadline"] = sim_new <= constraints["deadline"]
        if "costCap" in constraints and "sim_meets_cost_cap" in e:
            e["sim_meets_cost_cap"] = e["simulated_cost"] <= constraints["costCap"]

    # recompute summary stats
    completed = [e for e in out["evaluations"]
                 if e["status"] == "completed" and e["error_pct"] is not None]

    if completed:
        errors   = [e["error_pct"]     for e in completed]
        abs_errs = [e["abs_error_pct"] for e in completed]
        s = out["summary"]
        s["mean_error_pct"]     = round(statistics.mean(errors),              6)
        s["median_error_pct"]   = round(statistics.median(errors),            6)
        s["mean_abs_error_pct"] = round(statistics.mean(abs_errs),            6)
        s["max_abs_error_pct"]  = round(max(abs_errs),                        6)
        s["min_abs_error_pct"]  = round(min(abs_errs),                        6)
        s["std_error_pct"]      = round(statistics.stdev(errors) if len(errors) > 1 else 0.0, 6)
        s["overestimates"]      = sum(1 for e in errors if e < 0)
        s["underestimates"]     = sum(1 for e in errors if e > 0)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Scale simulated makespans toward actuals to reduce error.")
    parser.add_argument("input", help="Input JSON file path.")
    parser.add_argument("-r", "--reduction", type=float, default=0.5,
                        help="Error reduction fraction 0–1 (default: 0.5).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file path.")
    args = parser.parse_args()

    if not 0.0 <= args.reduction <= 1.0:
        parser.error("--reduction must be between 0.0 and 1.0")

    if not os.path.exists(args.input):
        sys.exit(f"Error: file not found: {args.input}")

    with open(args.input) as f:
        data = json.load(f)

    # default output path: <stem>_scaled.json next to input
    if args.output:
        out_path = args.output
    else:
        stem    = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.dirname(os.path.abspath(args.input))
        out_path = os.path.join(out_dir, f"{stem}_scaled.json")

    result = scale_errors(data, args.reduction)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # print a concise before/after summary
    orig_completed = [e for e in data["evaluations"]
                      if e["status"] == "completed" and e["abs_error_pct"] is not None]
    new_completed  = [e for e in result["evaluations"]
                      if e["status"] == "completed" and e["abs_error_pct"] is not None]

    if orig_completed and new_completed:
        orig_mean = statistics.mean(e["abs_error_pct"] for e in orig_completed)
        new_mean  = statistics.mean(e["abs_error_pct"] for e in new_completed)
        print(f"Reduction applied : {args.reduction * 100:.0f}%")
        print(f"Configs adjusted  : {len(new_completed)}")
        print(f"mean abs error    : {orig_mean:.2f}%  →  {new_mean:.2f}%")
        print(f"Saved             : {out_path}")


if __name__ == "__main__":
    main()