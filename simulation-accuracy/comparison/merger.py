#!/usr/bin/env python3
"""
Merge Bayesian Optimization Results with Real Kubernetes Evaluation
====================================================================
Takes:
  - optimization_results.json  (output of bayesian_opt.py)
  - eval_accuracy.json         (output of eval_simulation_accuracy.py)

Cross-references configurations by machine_counts fingerprint, then
produces a new JSON that mirrors eval_accuracy.json's structure but
uses REAL makespans instead of simulated ones.

Usage:
    python3 merge_bo_real.py
    python3 merge_bo_real.py --bo optimization_results.json --eval eval_accuracy.json --output real_scored.json
"""

import argparse
import json
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ============================================================================
# HELPERS
# ============================================================================

def config_key(machine_counts: List[int]) -> Tuple[int, ...]:
    """Canonical fingerprint for a cluster configuration."""
    return tuple(machine_counts)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def calculate_cluster_cost(
    machine_counts: List[int],
    available_machines: List[Dict],
    runtime_minutes: float,
) -> float:
    """Replicate the cost formula from bayesian_opt.py."""
    runtime_hours = runtime_minutes / 60.0
    total = 0.0
    for idx, count in enumerate(machine_counts):
        if count > 0 and idx < len(available_machines):
            total += available_machines[idx]["costPerHour"] * runtime_hours
    return total


# ============================================================================
# CORE LOGIC
# ============================================================================

def build_real_scored_output(
    bo_data: Dict,
    eval_data: Dict,
    skip_no_real_data: bool = False,
) -> Dict:
    """
    Produce a new eval_accuracy-style document where every entry uses the
    REAL makespan (from eval_accuracy.json) instead of the simulated one.

    Entries whose config was never executed on real K8s are marked as
    status='no_real_data'.
    """

    bo_evals: List[Dict] = (
        bo_data.get("optimization_results", {}).get("all_evaluations", [])
    )
    available_machines: List[Dict] = bo_data.get("config", {}).get("availableMachines", [])
    constraints: Dict = bo_data.get("config", {}).get("constraints", {})

    # Index real eval results by config fingerprint
    real_by_key: Dict[Tuple, Dict] = {}
    for ev in eval_data.get("evaluations", []):
        if ev.get("status") == "completed" and ev.get("actual_makespan_min") is not None:
            real_by_key[config_key(ev["machine_counts"])] = ev

    print(f"  BO evaluations         : {len(bo_evals)}")
    print(f"  Real completed configs : {len(real_by_key)}")

    # De-duplicate BO configs so we emit one merged entry per unique config
    seen: Dict[Tuple, Dict] = {}
    for ev in bo_evals:
        k = config_key(ev["machine_counts"])
        # Keep the first occurrence (lowest iteration number)
        if k not in seen:
            seen[k] = ev

    merged_evaluations: List[Dict] = []
    matched = 0
    unmatched = 0

    for k, bo_ev in seen.items():
        real_ev = real_by_key.get(k)
        mc = bo_ev["machine_counts"]

        entry: Dict = {
            # ---- identity ------------------------------------------------
            "config_id"             : bo_ev["iteration"],
            "machine_counts"        : mc,
            "num_machines"          : bo_ev["num_machines"],
            "selected_nodes"        : [
                available_machines[i]["machineId"]
                for i, c in enumerate(mc) if c > 0 and i < len(available_machines)
            ],
            # ---- simulated (from BO run) ---------------------------------
            "simulated_makespan_min": bo_ev["makespan_minutes"],
            "simulated_cost"        : bo_ev["total_cost"],
            "simulated_objective"   : bo_ev["objective_score"],
            "sim_meets_deadline"    : bo_ev.get("meets_deadline"),
            "sim_meets_cost_cap"    : bo_ev.get("meets_cost_cap"),
        }

        if real_ev:
            real_min  = real_ev["actual_makespan_min"]
            real_cost = calculate_cluster_cost(mc, available_machines, real_min)
            sim_min   = bo_ev["makespan_minutes"]

            # Re-score objective using real makespan/cost
            priority        = constraints.get("priority", "cost")
            deadline_target = constraints.get("deadline", float("inf"))
            cost_cap        = constraints.get("costCap", float("inf"))

            COST_PENALTY     = 2.0
            DEADLINE_PENALTY = 2.0

            if priority == "cost":
                real_objective = real_cost
                if real_min > deadline_target:
                    penalty = (real_min - deadline_target) / deadline_target * COST_PENALTY
                    real_objective += penalty * real_cost
            else:
                real_objective = real_min
                if real_cost > cost_cap:
                    penalty = (real_cost - cost_cap) / cost_cap * DEADLINE_PENALTY
                    real_objective += penalty * real_min

            error_min  = real_min - sim_min
            error_pct  = (error_min / real_min * 100) if real_min > 0 else 0.0

            entry.update({
                # ---- real (from K8s eval) --------------------------------
                "status"                : "completed",
                "actual_makespan_min"   : real_min,
                "actual_makespan_ms"    : real_ev.get("actual_makespan_ms"),
                "actual_cost"           : real_cost,
                "real_objective"        : real_objective,
                "real_meets_deadline"   : real_min <= deadline_target,
                "real_meets_cost_cap"   : real_cost <= cost_cap,
                # ---- accuracy -------------------------------------------
                "error_min"             : error_min,
                "error_pct"             : error_pct,
                "abs_error_pct"         : abs(error_pct),
                # ---- pass-through from eval file ------------------------
                "actual_avg_queue_time" : real_ev.get("actual_avg_queue_time"),
                "actual_avg_exec_time"  : real_ev.get("actual_avg_exec_time"),
                "actual_cluster_util"   : real_ev.get("actual_cluster_util"),
            })
            matched += 1
        else:
            entry.update({
                "status"              : "no_real_data",
                "actual_makespan_min" : None,
                "actual_makespan_ms"  : None,
                "actual_cost"         : None,
                "real_objective"      : None,
                "real_meets_deadline" : None,
                "real_meets_cost_cap" : None,
                "error_min"           : None,
                "error_pct"           : None,
                "abs_error_pct"       : None,
            })
            unmatched += 1
            if skip_no_real_data:
                continue

        merged_evaluations.append(entry)

    # Sort by config_id (original BO iteration order)
    merged_evaluations.sort(key=lambda e: e["config_id"])

    # ---- summary stats over completed entries ----------------------------
    completed = [e for e in merged_evaluations if e["status"] == "completed"]
    summary: Dict = {
        "total_unique_configs"   : len(merged_evaluations),
        "matched_with_real_data" : matched,
        "no_real_data"           : unmatched,
        "completed"              : len(completed),
        "failed"                 : 0,  # no failures introduced here
    }

    if completed:
        errors     = [e["error_pct"]     for e in completed]
        abs_errors = [e["abs_error_pct"] for e in completed]
        sim_objs   = [e["simulated_objective"] for e in completed]
        real_objs  = [e["real_objective"]      for e in completed]

        summary.update({
            "mean_error_pct"        : statistics.mean(errors),
            "median_error_pct"      : statistics.median(errors),
            "mean_abs_error_pct"    : statistics.mean(abs_errors),
            "max_abs_error_pct"     : max(abs_errors),
            "min_abs_error_pct"     : min(abs_errors),
            "std_error_pct"         : statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "overestimates"         : sum(1 for e in errors if e < 0),
            "underestimates"        : sum(1 for e in errors if e > 0),
            # Best config by simulated objective vs real objective
            "best_sim_config_id"    : min(completed, key=lambda e: e["simulated_objective"])["config_id"],
            "best_real_config_id"   : min(completed, key=lambda e: e["real_objective"])["config_id"],
            "sim_obj_range"         : [min(sim_objs), max(sim_objs)],
            "real_obj_range"        : [min(real_objs), max(real_objs)],
        })

        # Did the BO best config also have the best real objective?
        best_sim_id  = summary["best_sim_config_id"]
        best_real_id = summary["best_real_config_id"]
        summary["bo_picked_best_real"] = (best_sim_id == best_real_id)

        print(f"\n  Best by simulated obj  : config_id={best_sim_id}")
        print(f"  Best by real obj       : config_id={best_real_id}")
        print(f"  BO picked best real?   : {summary['bo_picked_best_real']}")
        print(f"\n  Mean error             : {summary['mean_error_pct']:+.2f}%")
        print(f"  Mean |error|           : {summary['mean_abs_error_pct']:.2f}%")
        print(f"  Max  |error|           : {summary['max_abs_error_pct']:.2f}%")
        print(f"  Std dev                : {summary['std_error_pct']:.2f}%")
        print(f"  Overestimates          : {summary['overestimates']}  (sim > actual)")
        print(f"  Underestimates         : {summary['underestimates']}  (sim < actual)")

    return {
        "metadata": {
            "generated_at"   : datetime.now().isoformat(),
            "description"    : (
                "BO optimization results re-scored with real Kubernetes makespans. "
                "Each entry includes both the simulated metrics used during BO and "
                "the real metrics measured on the actual cluster."
            ),
            "constraints"    : constraints,
            "total_bo_evals" : len(bo_evals),
            "is_partial"     : False,
        },
        "summary"    : summary,
        "evaluations": merged_evaluations,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge BO results with real K8s eval data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 merge_bo_real.py
  python3 merge_bo_real.py --bo optimization_results.json --eval eval_accuracy.json
  python3 merge_bo_real.py --bo results.json --eval accuracy.json --output merged.json
  python3 merge_bo_real.py --skip-no-real-data
        """,
    )
    parser.add_argument(
        "--bo",
        default="optimization_results.json",
        help="Path to BO output (default: optimization_results.json)",
    )
    parser.add_argument(
        "--eval",
        default="eval_accuracy.json",
        help="Path to real eval output (default: eval_accuracy.json)",
    )
    parser.add_argument(
        "--output",
        default="real_scored.json",
        help="Output file (default: real_scored.json)",
    )
    parser.add_argument(
        "--skip-no-real-data",
        action="store_true",
        help="Omit entries with no matching real K8s run (default: include as no_real_data)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("MERGE BO + REAL KUBERNETES RESULTS")
    print("=" * 64)
    print(f"  BO file           : {args.bo}")
    print(f"  Eval file         : {args.eval}")
    print(f"  Output            : {args.output}")
    print(f"  Skip no-real-data : {args.skip_no_real_data}")
    print()

    bo_data   = load_json(args.bo)
    eval_data = load_json(args.eval)

    output = build_real_scored_output(bo_data, eval_data, skip_no_real_data=args.skip_no_real_data)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    n = output["summary"]["completed"]
    print(f"\n  Wrote {n} merged entries to: {args.output}")
    print("=" * 64)


if __name__ == "__main__":
    main()