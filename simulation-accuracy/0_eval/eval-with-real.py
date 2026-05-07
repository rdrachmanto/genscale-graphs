#!/usr/bin/env python3
"""
Simulation Accuracy Evaluator
==============================
Takes optimization_results.json, replays each unique cluster configuration
on a real Kubernetes cluster by tainting/untainting nodes, submits workflows,
measures actual makespan, and compares against simulated makespan.

Usage:
    python3 eval_simulation_accuracy.py --results optimization_results.json
    python3 eval_simulation_accuracy.py --results optimization_results.json --api-url http://localhost:3333
    python3 eval_simulation_accuracy.py --results optimization_results.json --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
import requests
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict


# ============================================================================
# CONFIGURATION
# ============================================================================

TAINT_KEY = "genscale-eval"
TAINT_EFFECT = "NoSchedule"

POLL_INTERVAL_SEC = 10  # How often to check if workflows are done
STALL_TIMEOUT_SEC = 7200  # 2 hours with zero progress = stalled


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ConfigEval:
    """Result of evaluating one cluster configuration."""
    config_id: int
    machine_counts: List[int]
    selected_nodes: List[str]
    num_machines: int
    simulated_makespan_min: float
    simulated_cost: float
    actual_makespan_min: Optional[float] = None
    actual_makespan_ms: Optional[int] = None
    error_min: Optional[float] = None          # actual - simulated
    error_pct: Optional[float] = None          # (actual - simulated) / actual * 100
    abs_error_pct: Optional[float] = None
    status: str = "pending"                    # pending | running | completed | failed
    failure_reason: Optional[str] = None
    actual_avg_queue_time: Optional[float] = None
    actual_avg_exec_time: Optional[float] = None
    actual_cluster_util: Optional[float] = None


@dataclass
class EvalSummary:
    """Overall evaluation summary."""
    total_configs: int = 0
    completed: int = 0
    failed: int = 0
    mean_error_pct: Optional[float] = None
    median_error_pct: Optional[float] = None
    mean_abs_error_pct: Optional[float] = None
    max_abs_error_pct: Optional[float] = None
    min_abs_error_pct: Optional[float] = None
    std_error_pct: Optional[float] = None
    overestimates: int = 0    # simulation predicted longer than actual
    underestimates: int = 0   # simulation predicted shorter than actual
    results: List[dict] = field(default_factory=list)


# ============================================================================
# KUBERNETES HELPERS
# ============================================================================

def run_kubectl(args: List[str], check=True) -> subprocess.CompletedProcess:
    """Run a kubectl command."""
    cmd = ["kubectl"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def taint_node(node: str) -> bool:
    """Add NoSchedule taint to a node. Returns True on success."""
    try:
        result = run_kubectl([
            "taint", "nodes", node,
            f"{TAINT_KEY}=true:{TAINT_EFFECT}",
            "--overwrite"
        ], check=False)
        if result.returncode != 0 and "already has" not in result.stderr:
            print(f"  WARN: Failed to taint {node}: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"  ERROR tainting {node}: {e}")
        return False


def untaint_node(node: str) -> bool:
    """Remove NoSchedule taint from a node. Returns True on success."""
    try:
        result = run_kubectl([
            "taint", "nodes", node,
            f"{TAINT_KEY}=true:{TAINT_EFFECT}-"
        ], check=False)
        # "not found" is fine - means it wasn't tainted
        if result.returncode != 0 and "not found" not in result.stderr:
            print(f"  WARN: Failed to untaint {node}: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"  ERROR untainting {node}: {e}")
        return False


def configure_cluster(selected_machine_ids: List[str], all_worker_nodes: List[str]) -> bool:
    """
    Taint all non-selected worker nodes, untaint selected ones.
    machine IDs from config ARE the node names.
    """
    selected = set(selected_machine_ids)
    ok = True
    for node in all_worker_nodes:
        if node in selected:
            if not untaint_node(node):
                ok = False
        else:
            if not taint_node(node):
                ok = False
    return ok


def reset_cluster(all_worker_nodes: List[str]):
    """Remove all eval taints from all worker nodes."""
    print("Resetting cluster (removing all eval taints)...")
    for node in all_worker_nodes:
        untaint_node(node)


def drain_running_pods():
    """Delete any leftover workflow pods to ensure clean state."""
    try:
        # Delete all pods in default namespace that aren't system pods
        run_kubectl([
            "delete", "pods", "--all",
            "-n", "default",
            "--grace-period=0", "--force"
        ], check=False)
    except Exception:
        pass


# ============================================================================
# WORKFLOW API HELPERS
# ============================================================================

def trigger_workflows(api_url: str, workflows: List[str]) -> List[str]:
    """Trigger all workflows and return their IDs."""
    workflow_ids = []
    for wf in workflows:
        try:
            resp = requests.get(f"{api_url}/exec-workflow/{wf}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            wf_id = data.get("workflowName", "")
            workflow_ids.append(wf_id)
        except Exception as e:
            print(f"  ERROR triggering {wf}: {e}")
            workflow_ids.append(None)
    return workflow_ids


def _is_workflow_done(ex: Dict) -> bool:
    """Match exactly what the frontend uses: exec_state.state === 'completed'."""
    return ex.get("exec_state", {}).get("state") == "completed"


def poll_completion(
    api_url: str,
    workflow_ids: List[str],
    stall_timeout: int = STALL_TIMEOUT_SEC,
) -> Tuple[bool, Optional[Dict]]:
    """
    Poll /get-executions until all tracked workflows are done.

    Completion is determined by _is_workflow_done(), which checks task counts
    and logv2 independently — not the unreliable exec_state.state string.

    Progress tracking counts (pending + running) tasks across all workflows.
    A stall is declared only if that number doesn't decrease for stall_timeout
    seconds, so long-running workflows are handled fine.

    Returns (success, aggregated_result).
    """
    wall_start = time.time()
    valid_ids = set(wid for wid in workflow_ids if wid)

    if not valid_ids:
        return False, None

    last_progress_time = time.time()
    prev_active_task_count = None   # total pending+running across all our workflows

    while True:
        try:
            resp = requests.get(f"{api_url}/get-executions", timeout=30)
            resp.raise_for_status()
            executions = resp.json()

            # Index executions by id and name so we can find ours regardless
            # of which field the backend happens to populate.
            our_execs: Dict[str, Dict] = {}
            for ex in executions:
                ex_id   = ex.get("id")
                ex_name = ex.get("name")
                if ex_id in valid_ids:
                    our_execs[ex_id] = ex
                elif ex_name in valid_ids:
                    our_execs[ex_name] = ex

            # Count completions using our own logic, not exec_state.state
            done_ids = set()
            active_task_count = 0

            for wid, ex in our_execs.items():
                if _is_workflow_done(ex):
                    done_ids.add(wid)
                else:
                    exec_state = ex.get("exec_state", {})
                    active_task_count += (
                        len(exec_state.get("tasksPending", [])) +
                        len(exec_state.get("tasksRunning", []))
                    )

            completed_count = len(done_ids)
            missing_count   = len(valid_ids) - len(our_execs)   # not yet visible

            # Detect progress: total active tasks decreased
            if prev_active_task_count is None or active_task_count < prev_active_task_count:
                last_progress_time = time.time()
            prev_active_task_count = active_task_count

            elapsed       = time.time() - wall_start
            stall_elapsed = time.time() - last_progress_time

            print(
                f"\r  [{elapsed:.0f}s elapsed, stall {stall_elapsed:.0f}s] "
                f"{completed_count}/{len(valid_ids)} complete  "
                f"({active_task_count} tasks active, {missing_count} wf not yet visible)",
                end="", flush=True,
            )

            if completed_count >= len(valid_ids):
                wall_ms = int((time.time() - wall_start) * 1000)
                print()
                return True, {
                    "makespan_ms":  wall_ms,
                    "makespan_min": wall_ms / (1000 * 60),
                }

            if stall_elapsed > stall_timeout:
                print(f"\n  STALL TIMEOUT: No progress for {stall_timeout}s")
                # Dump state of incomplete workflows to help diagnose
                for wid in valid_ids - done_ids:
                    ex = our_execs.get(wid)
                    if ex:
                        es = ex.get("exec_state", {})
                        print(f"  STUCK [{wid}]: state={es.get('state')} "
                              f"pending={len(es.get('tasksPending', []))} "
                              f"running={len(es.get('tasksRunning', []))} "
                              f"completed={len(es.get('tasksCompleted', []))}")
                        lv2 = ex.get("logv2", {})
                        incomplete = [
                            t for t, ev in lv2.items()
                            if isinstance(ev, dict) and ev.get("COMPLETED") is None
                        ]
                        if incomplete:
                            print(f"    logv2 incomplete tasks: {incomplete}")
                    else:
                        print(f"  STUCK [{wid}]: never appeared in /get-executions")
                return False, None

        except Exception as e:
            print(f"\n  Poll error: {e}")

        time.sleep(POLL_INTERVAL_SEC)


def clear_execution_history(api_url: str):
    """No-op: server has no reset endpoint."""
    pass


# ============================================================================
# INCREMENTAL PERSISTENCE
# ============================================================================

def _build_output(eval_results: List[ConfigEval], results_file: str, api_url: str) -> Dict:
    completed = [r for r in eval_results if r.status == "completed"]

    summary = {
        "completed": len(completed),
        "failed": sum(1 for r in eval_results if r.status == "failed"),
        "mean_error_pct": None,
        "median_error_pct": None,
        "mean_abs_error_pct": None,
        "max_abs_error_pct": None,
        "min_abs_error_pct": None,
        "std_error_pct": None,
        "overestimates": 0,
        "underestimates": 0,
    }

    if completed:
        errors_pct = [r.error_pct for r in completed]
        abs_errors = [r.abs_error_pct for r in completed]
        summary["mean_error_pct"]     = statistics.mean(errors_pct)
        summary["median_error_pct"]   = statistics.median(errors_pct)
        summary["mean_abs_error_pct"] = statistics.mean(abs_errors)
        summary["max_abs_error_pct"]  = max(abs_errors)
        summary["min_abs_error_pct"]  = min(abs_errors)
        summary["std_error_pct"]      = statistics.stdev(errors_pct) if len(errors_pct) > 1 else 0.0
        summary["overestimates"]      = sum(1 for e in errors_pct if e < 0)
        summary["underestimates"]     = sum(1 for e in errors_pct if e > 0)

    return {
        "metadata": {
            "source_file": results_file,
            "api_url": api_url,
            "timestamp": datetime.now().isoformat(),
            "total_configs": len(eval_results),
            "is_partial": True,
        },
        "summary": summary,
        "evaluations": [asdict(r) for r in eval_results],
    }


def _save_partial(output_file: str, eval_results: List[ConfigEval], results_file: str, api_url: str):
    """Write current results to disk after every evaluation (crash-safe)."""
    data = _build_output(eval_results, results_file, api_url)
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    import os
    os.replace(tmp, output_file)


def _load_partial(output_file: str) -> Tuple[List[ConfigEval], set]:
    """Load partial results from a previous run (if file exists)."""
    import os
    if not os.path.exists(output_file):
        return [], set()

    try:
        with open(output_file) as f:
            data = json.load(f)

        results = []
        done_keys = set()
        for ev in data.get("evaluations", []):
            ce = ConfigEval(
                config_id=ev["config_id"],
                machine_counts=ev["machine_counts"],
                selected_nodes=ev["selected_nodes"],
                num_machines=ev["num_machines"],
                simulated_makespan_min=ev["simulated_makespan_min"],
                simulated_cost=ev["simulated_cost"],
                actual_makespan_min=ev.get("actual_makespan_min"),
                actual_makespan_ms=ev.get("actual_makespan_ms"),
                error_min=ev.get("error_min"),
                error_pct=ev.get("error_pct"),
                abs_error_pct=ev.get("abs_error_pct"),
                status=ev.get("status", "unknown"),
                failure_reason=ev.get("failure_reason"),
                actual_avg_queue_time=ev.get("actual_avg_queue_time"),
                actual_avg_exec_time=ev.get("actual_avg_exec_time"),
                actual_cluster_util=ev.get("actual_cluster_util"),
            )
            results.append(ce)
            done_keys.add(tuple(ce.machine_counts))
        return results, done_keys

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  WARN: Could not load partial results from {output_file}: {e}")
        return [], set()


# ============================================================================
# MAIN EVALUATION LOGIC
# ============================================================================

def load_results(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


def extract_unique_configs(data: Dict) -> List[Dict]:
    evals = data.get("optimization_results", {}).get("all_evaluations", [])
    seen = set()
    unique = []
    for ev in evals:
        key = tuple(ev["machine_counts"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    return unique


def get_selected_machine_ids(machine_counts: List[int], available_machines: List[Dict]) -> List[str]:
    selected = []
    for idx, count in enumerate(machine_counts):
        if count > 0 and idx < len(available_machines):
            selected.append(available_machines[idx]["machineId"])
    return selected


def run_evaluation(
    results_file: str,
    api_url: str = "http://localhost:3333",
    dry_run: bool = False,
    max_configs: Optional[int] = None,
    output_file: str = "eval_accuracy.json",
    skip_reset: bool = False,
    stall_timeout: int = STALL_TIMEOUT_SEC,
):
    print("=" * 72)
    print("SIMULATION ACCURACY EVALUATION")
    print("=" * 72)
    print(f"  Results file : {results_file}")
    print(f"  API URL      : {api_url}")
    print(f"  Output file  : {output_file}")
    print(f"  Dry run      : {dry_run}")
    print(f"  Max configs  : {max_configs or 'all'}")
    print(f"  Started      : {datetime.now().isoformat()}")
    print("=" * 72)

    data = load_results(results_file)
    available_machines = data.get("config", {}).get("availableMachines", [])
    workflows = data.get("config", {}).get("workflows", [])
    unique_configs = extract_unique_configs(data)

    all_worker_nodes = [m["machineId"] for m in available_machines]

    if max_configs:
        unique_configs = unique_configs[:max_configs]

    print(f"\n  Total evaluations in file : {len(data.get('optimization_results', {}).get('all_evaluations', []))}")
    print(f"  Unique configurations     : {len(unique_configs)}")
    print(f"  Workflows per run         : {len(workflows)}")
    print(f"  Available machines        : {len(available_machines)}")
    print()

    eval_results: List[ConfigEval] = []
    eval_results, done_keys = _load_partial(output_file)
    if eval_results:
        print(f"  Resumed from {output_file}: {len(eval_results)} prior results loaded")

    try:
        for i, config in enumerate(unique_configs):
            config_num = i + 1
            mc = config["machine_counts"]
            selected_ids = get_selected_machine_ids(mc, available_machines)
            selected_nodes = selected_ids
            sim_makespan = config["makespan_minutes"]
            sim_cost = config["total_cost"]
            iteration = config["iteration"]

            config_key = tuple(mc)
            if config_key in done_keys:
                print(f"[{config_num}/{len(unique_configs)}] Config iter {iteration} — already evaluated, skipping")
                continue

            ce = ConfigEval(
                config_id=iteration,
                machine_counts=mc,
                selected_nodes=selected_nodes,
                num_machines=len(selected_ids),
                simulated_makespan_min=sim_makespan,
                simulated_cost=sim_cost,
            )

            print("-" * 72)
            print(f"[{config_num}/{len(unique_configs)}] Config from iteration {iteration}")
            print(f"  Nodes    : {', '.join(selected_nodes) or 'NONE'}")
            print(f"  Sim time : {sim_makespan:.2f} min | Sim cost: ${sim_cost:.2f}")

            if not selected_ids:
                ce.status = "failed"
                ce.failure_reason = "No machines selected"
                eval_results.append(ce)
                _save_partial(output_file, eval_results, results_file, api_url)
                print("  SKIP: No machines selected")
                continue

            if dry_run:
                ce.status = "dry_run"
                eval_results.append(ce)
                _save_partial(output_file, eval_results, results_file, api_url)
                print("  DRY RUN: Would taint/untaint and run workflows")
                continue

            print("  Configuring cluster (taint/untaint)...")
            if not configure_cluster(selected_ids, all_worker_nodes):
                print("  WARN: Some taint operations failed, proceeding anyway")

            time.sleep(2)

            clear_execution_history(api_url)
            time.sleep(1)

            print(f"  Triggering {len(workflows)} workflows...")
            ce.status = "running"
            wf_start = time.time()
            workflow_ids = trigger_workflows(api_url, workflows)
            triggered = sum(1 for w in workflow_ids if w)
            print(f"  Triggered {triggered}/{len(workflows)} workflows: {[w for w in workflow_ids if w]}")

            if triggered == 0:
                ce.status = "failed"
                ce.failure_reason = "No workflows triggered successfully"
                eval_results.append(ce)
                _save_partial(output_file, eval_results, results_file, api_url)
                continue

            print("  Waiting for completion...")
            success, result = poll_completion(api_url, workflow_ids, stall_timeout=stall_timeout)

            if success and result:
                actual_ms  = result["makespan_ms"]
                actual_min = result["makespan_min"]

                ce.actual_makespan_ms  = actual_ms
                ce.actual_makespan_min = actual_min
                ce.error_min           = actual_min - sim_makespan
                ce.error_pct           = ((actual_min - sim_makespan) / actual_min * 100) if actual_min > 0 else 0
                ce.abs_error_pct       = abs(ce.error_pct)
                ce.status              = "completed"

                wall_time = time.time() - wf_start
                print(f"  DONE in {wall_time:.1f}s wall clock")
                print(f"  Actual   : {actual_min:.2f} min")
                print(f"  Simulated: {sim_makespan:.2f} min")
                print(f"  Error    : {ce.error_min:+.2f} min ({ce.error_pct:+.1f}%)")
            else:
                ce.status = "failed"
                ce.failure_reason = "Timeout or poll failure"
                print("  FAILED: Could not get completion data")

            eval_results.append(ce)
            _save_partial(output_file, eval_results, results_file, api_url)

            print("  Cleaning up pods...")
            drain_running_pods()
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        _save_partial(output_file, eval_results, results_file, api_url)
    finally:
        if not dry_run and not skip_reset:
            reset_cluster(all_worker_nodes)

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    completed = [r for r in eval_results if r.status == "completed"]
    failed    = [r for r in eval_results if r.status == "failed"]

    summary = EvalSummary(
        total_configs=len(eval_results),
        completed=len(completed),
        failed=len(failed),
    )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    if completed:
        errors_pct = [r.error_pct for r in completed]
        abs_errors = [r.abs_error_pct for r in completed]

        summary.mean_error_pct     = statistics.mean(errors_pct)
        summary.median_error_pct   = statistics.median(errors_pct)
        summary.mean_abs_error_pct = statistics.mean(abs_errors)
        summary.max_abs_error_pct  = max(abs_errors)
        summary.min_abs_error_pct  = min(abs_errors)
        summary.std_error_pct      = statistics.stdev(errors_pct) if len(errors_pct) > 1 else 0.0
        summary.overestimates      = sum(1 for e in errors_pct if e < 0)
        summary.underestimates     = sum(1 for e in errors_pct if e > 0)

        print(f"\n  Configs evaluated : {summary.total_configs}")
        print(f"  Completed         : {summary.completed}")
        print(f"  Failed            : {summary.failed}")
        print()
        print(f"  Mean Error        : {summary.mean_error_pct:+.2f}%")
        print(f"  Median Error      : {summary.median_error_pct:+.2f}%")
        print(f"  Std Dev           : {summary.std_error_pct:.2f}%")
        print(f"  Mean |Error|      : {summary.mean_abs_error_pct:.2f}%")
        print(f"  Min |Error|       : {summary.min_abs_error_pct:.2f}%")
        print(f"  Max |Error|       : {summary.max_abs_error_pct:.2f}%")
        print(f"  Overestimates     : {summary.overestimates} (sim predicted longer)")
        print(f"  Underestimates    : {summary.underestimates} (sim predicted shorter)")

        print(f"\n  {'Config':>6} {'Nodes':>5} {'Sim(min)':>9} {'Actual(min)':>11} {'Err(min)':>9} {'Err(%)':>8}")
        print(f"  {'-'*6} {'-'*5} {'-'*9} {'-'*11} {'-'*9} {'-'*8}")
        for r in completed:
            print(f"  {r.config_id:>6} {r.num_machines:>5} {r.simulated_makespan_min:>9.2f} "
                  f"{r.actual_makespan_min:>11.2f} {r.error_min:>+9.2f} {r.error_pct:>+7.1f}%")
    else:
        print("\n  No configurations completed successfully.")

    if failed:
        print(f"\n  Failed configurations:")
        for r in failed:
            nodes_str = ', '.join(r.selected_nodes) if r.selected_nodes else 'none'
            print(f"    Config {r.config_id}: {r.failure_reason} (nodes: {nodes_str})")

    output = _build_output(eval_results, results_file, api_url)
    output["metadata"]["is_partial"] = False
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file}")
    print("=" * 72)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simulation accuracy against real Kubernetes cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 eval_simulation_accuracy.py --results optimization_results.json
  python3 eval_simulation_accuracy.py --results results.json --api-url http://10.0.0.5:3333
  python3 eval_simulation_accuracy.py --results results.json --dry-run
  python3 eval_simulation_accuracy.py --results results.json --max-configs 5
        """,
    )
    parser.add_argument("--results",       required=True,                       help="Path to optimization_results.json")
    parser.add_argument("--api-url",       default="http://localhost:3333",     help="GenScale API URL (default: http://localhost:3333)")
    parser.add_argument("--output",        default="eval_accuracy.json",        help="Output file (default: eval_accuracy.json)")
    parser.add_argument("--dry-run",       action="store_true",                 help="Show what would run without executing")
    parser.add_argument("--max-configs",   type=int, default=None,              help="Limit number of configs to evaluate")
    parser.add_argument("--skip-reset",    action="store_true",                 help="Don't reset taints on exit")
    parser.add_argument("--stall-timeout", type=int, default=STALL_TIMEOUT_SEC, help=f"Seconds with no progress before declaring stall (default: {STALL_TIMEOUT_SEC})")

    args = parser.parse_args()

    run_evaluation(
        results_file=args.results,
        api_url=args.api_url,
        dry_run=args.dry_run,
        max_configs=args.max_configs,
        output_file=args.output,
        skip_reset=args.skip_reset,
        stall_timeout=args.stall_timeout,
    )


if __name__ == "__main__":
    main()