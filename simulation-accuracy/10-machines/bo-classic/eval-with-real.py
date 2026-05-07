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
CONTROL_PLANE_NODE = "k8s-cp"  # Never taint/untaint the control plane

# All worker nodes in the cluster (must match your cluster)
ALL_WORKER_NODES = [
    "k8s-tiny-1", "k8s-tiny-2", "k8s-tiny-3", "k8s-tiny-4",
    "k8s-small-1", "k8s-small-2", "k8s-small-3",
    "k8s-medium-1", "k8s-medium-2",
    "k8s-large",
]

# Mapping from machineId in config -> actual k8s node name
# (handles cases like "k8s-large-1" in config vs "k8s-large" in cluster)
NODE_NAME_MAP = {
    "k8s-tiny-1": "k8s-tiny-1",
    "k8s-tiny-2": "k8s-tiny-2",
    "k8s-tiny-3": "k8s-tiny-3",
    "k8s-tiny-4": "k8s-tiny-4",
    "k8s-small-1": "k8s-small-1",
    "k8s-small-2": "k8s-small-2",
    "k8s-small-3": "k8s-small-3",
    "k8s-medium-1": "k8s-medium-1",
    "k8s-medium-2": "k8s-medium-2",
    "k8s-large-1": "k8s-large",  # config says k8s-large-1, cluster has k8s-large
}

POLL_INTERVAL_SEC = 5  # How often to check if workflows are done
WORKFLOW_TIMEOUT_SEC = 1800  # 30 min max wait per config


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


def configure_cluster(selected_machine_ids: List[str]) -> bool:
    """
    Taint all non-selected worker nodes, untaint selected ones.
    Returns True if all operations succeeded.
    """
    # Map config machine IDs to actual node names
    selected_nodes = set()
    for mid in selected_machine_ids:
        actual = NODE_NAME_MAP.get(mid, mid)
        selected_nodes.add(actual)

    ok = True
    for node in ALL_WORKER_NODES:
        if node in selected_nodes:
            if not untaint_node(node):
                ok = False
        else:
            if not taint_node(node):
                ok = False

    return ok


def reset_cluster():
    """Remove all eval taints from all worker nodes."""
    print("Resetting cluster (removing all eval taints)...")
    for node in ALL_WORKER_NODES:
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
            resp = requests.post(f"{api_url}/api/workflow/trigger/{wf}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            wf_id = data.get("workflowName", "")
            workflow_ids.append(wf_id)
        except Exception as e:
            print(f"  ERROR triggering {wf}: {e}")
            workflow_ids.append(None)
    return workflow_ids


def poll_completion(api_url: str, workflow_ids: List[str], timeout_sec: int = WORKFLOW_TIMEOUT_SEC) -> Tuple[bool, Optional[Dict]]:
    """
    Poll /api/workflow/executions until all workflows complete or timeout.
    Returns (success, aggregated_result).
    """
    start = time.time()
    valid_ids = [wid for wid in workflow_ids if wid]

    if not valid_ids:
        return False, None

    while (time.time() - start) < timeout_sec:
        try:
            resp = requests.get(f"{api_url}/api/workflow/executions", timeout=30)
            resp.raise_for_status()
            executions = resp.json()

            # Check completion status for our workflows
            statuses = {}
            for ex in executions:
                if ex.get("id") in valid_ids or ex.get("name") in valid_ids:
                    state = ex.get("exec_state", {}).get("state", "unknown")
                    statuses[ex.get("id") or ex.get("name")] = state

            completed_count = sum(1 for s in statuses.values() if s == "completed")
            running_count = sum(1 for s in statuses.values() if s in ("running", "pending"))
            total_found = len(statuses)

            elapsed = time.time() - start
            print(f"\r  Polling [{elapsed:.0f}s]: {completed_count}/{len(valid_ids)} complete, "
                  f"{running_count} active, {total_found} found", end="", flush=True)

            if completed_count >= len(valid_ids):
                print()  # newline after polling
                # Calculate actual makespan from execution logs
                return True, _extract_makespan(executions, valid_ids)

        except Exception as e:
            print(f"\n  Poll error: {e}")

        time.sleep(POLL_INTERVAL_SEC)

    print(f"\n  TIMEOUT after {timeout_sec}s")
    return False, None


def _extract_makespan(executions: List[Dict], workflow_ids: List[str]) -> Dict:
    """Extract makespan and stats from execution data."""
    earliest_start = float('inf')
    latest_end = 0

    all_queue_times = []
    all_exec_times = []

    for ex in executions:
        ex_id = ex.get("id") or ex.get("name")
        if ex_id not in workflow_ids:
            continue

        logs = ex.get("logs", [])
        for task in logs:
            s = task.get("start")
            e = task.get("end")
            if s is not None:
                earliest_start = min(earliest_start, s)
            if e is not None:
                latest_end = max(latest_end, e)

        # Try logv2 for more detailed timing
        logv2 = ex.get("logv2", {})
        for job_name, events in logv2.items():
            if isinstance(events, list):
                for ev in events:
                    ts = ev.get("timestamp")
                    if ts is not None:
                        if ev.get("event") in ("ADDED", "SUBMITTED"):
                            earliest_start = min(earliest_start, ts)
                        elif ev.get("event") == "COMPLETED":
                            latest_end = max(latest_end, ts)

    makespan_ms = latest_end - earliest_start if earliest_start < float('inf') else 0

    return {
        "makespan_ms": makespan_ms,
        "makespan_min": makespan_ms / (1000 * 60) if makespan_ms > 0 else 0,
        "earliest_start": earliest_start if earliest_start < float('inf') else None,
        "latest_end": latest_end if latest_end > 0 else None,
    }


def clear_execution_history(api_url: str):
    """Try to clear execution history via API (if endpoint exists), otherwise just note it."""
    try:
        # Try common reset endpoints
        for endpoint in ["/api/simulation/reset", "/api/workflow/reset", "/api/reset"]:
            resp = requests.post(f"{api_url}{endpoint}", timeout=10)
            if resp.status_code == 200:
                return True
    except Exception:
        pass
    return False


# ============================================================================
# MAIN EVALUATION LOGIC
# ============================================================================

def load_results(filepath: str) -> Dict:
    """Load optimization results JSON."""
    with open(filepath) as f:
        return json.load(f)


def extract_unique_configs(data: Dict) -> List[Dict]:
    """Extract unique configurations from optimization results."""
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
    """Get list of selected machine IDs from binary machine_counts vector."""
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
):
    """Main evaluation loop."""

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

    # Load data
    data = load_results(results_file)
    available_machines = data.get("config", {}).get("availableMachines", [])
    workflows = data.get("config", {}).get("workflows", [])
    unique_configs = extract_unique_configs(data)

    if max_configs:
        unique_configs = unique_configs[:max_configs]

    print(f"\n  Total evaluations in file : {len(data.get('optimization_results', {}).get('all_evaluations', []))}")
    print(f"  Unique configurations     : {len(unique_configs)}")
    print(f"  Workflows per run         : {len(workflows)}")
    print(f"  Available machines        : {len(available_machines)}")
    print()

    eval_results: List[ConfigEval] = []

    try:
        for i, config in enumerate(unique_configs):
            config_num = i + 1
            mc = config["machine_counts"]
            selected_ids = get_selected_machine_ids(mc, available_machines)
            selected_nodes = [NODE_NAME_MAP.get(mid, mid) for mid in selected_ids]
            sim_makespan = config["makespan_minutes"]
            sim_cost = config["total_cost"]
            iteration = config["iteration"]

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
                print("  SKIP: No machines selected")
                continue

            if dry_run:
                ce.status = "dry_run"
                eval_results.append(ce)
                print("  DRY RUN: Would taint/untaint and run workflows")
                continue

            # 1. Configure cluster
            print("  Configuring cluster (taint/untaint)...")
            if not configure_cluster(selected_ids):
                print("  WARN: Some taint operations failed, proceeding anyway")

            # Brief pause for taints to propagate
            time.sleep(2)

            # 2. Clear previous state if possible
            clear_execution_history(api_url)
            time.sleep(1)

            # 3. Trigger workflows
            print(f"  Triggering {len(workflows)} workflows...")
            ce.status = "running"
            wf_start = time.time()
            workflow_ids = trigger_workflows(api_url, workflows)
            triggered = sum(1 for w in workflow_ids if w)
            print(f"  Triggered {triggered}/{len(workflows)} workflows")

            if triggered == 0:
                ce.status = "failed"
                ce.failure_reason = "No workflows triggered successfully"
                eval_results.append(ce)
                continue

            # 4. Poll for completion
            print("  Waiting for completion...")
            success, result = poll_completion(api_url, workflow_ids)

            if success and result:
                actual_ms = result["makespan_ms"]
                actual_min = result["makespan_min"]

                ce.actual_makespan_ms = actual_ms
                ce.actual_makespan_min = actual_min
                ce.error_min = actual_min - sim_makespan
                ce.error_pct = ((actual_min - sim_makespan) / actual_min * 100) if actual_min > 0 else 0
                ce.abs_error_pct = abs(ce.error_pct)
                ce.status = "completed"

                wall_time = time.time() - wf_start
                print(f"  DONE in {wall_time:.1f}s wall clock")
                print(f"  Actual   : {actual_min:.2f} min")
                print(f"  Simulated: {sim_makespan:.2f} min")
                print(f"  Error    : {ce.error_min:+.2f} min ({ce.error_pct:+.1f}%)")
            else:
                ce.status = "failed"
                ce.failure_reason = "Timeout or poll failure"
                print(f"  FAILED: Could not get completion data")

            eval_results.append(ce)

            # 5. Clean up between runs
            print("  Cleaning up pods...")
            drain_running_pods()
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
    finally:
        # Always reset cluster
        if not dry_run and not skip_reset:
            reset_cluster()

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    completed = [r for r in eval_results if r.status == "completed"]
    failed = [r for r in eval_results if r.status == "failed"]

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

        summary.mean_error_pct = statistics.mean(errors_pct)
        summary.median_error_pct = statistics.median(errors_pct)
        summary.mean_abs_error_pct = statistics.mean(abs_errors)
        summary.max_abs_error_pct = max(abs_errors)
        summary.min_abs_error_pct = min(abs_errors)
        summary.std_error_pct = statistics.stdev(errors_pct) if len(errors_pct) > 1 else 0.0
        summary.overestimates = sum(1 for e in errors_pct if e < 0)   # sim > actual
        summary.underestimates = sum(1 for e in errors_pct if e > 0)  # sim < actual

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

        # Per-config table
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

    # Save results
    summary.results = [asdict(r) for r in eval_results]
    output = {
        "metadata": {
            "source_file": results_file,
            "api_url": api_url,
            "timestamp": datetime.now().isoformat(),
            "total_configs": summary.total_configs,
        },
        "summary": {
            "completed": summary.completed,
            "failed": summary.failed,
            "mean_error_pct": summary.mean_error_pct,
            "median_error_pct": summary.median_error_pct,
            "mean_abs_error_pct": summary.mean_abs_error_pct,
            "max_abs_error_pct": summary.max_abs_error_pct,
            "min_abs_error_pct": summary.min_abs_error_pct,
            "std_error_pct": summary.std_error_pct,
            "overestimates": summary.overestimates,
            "underestimates": summary.underestimates,
        },
        "evaluations": summary.results,
    }

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
    parser.add_argument("--results", required=True, help="Path to optimization_results.json")
    parser.add_argument("--api-url", default="http://localhost:3333", help="GenScale API URL (default: http://localhost:3333)")
    parser.add_argument("--output", default="eval_accuracy.json", help="Output file (default: eval_accuracy.json)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs to evaluate")
    parser.add_argument("--skip-reset", action="store_true", help="Don't reset taints on exit")
    parser.add_argument("--timeout", type=int, default=WORKFLOW_TIMEOUT_SEC, help=f"Per-config timeout in seconds (default: {WORKFLOW_TIMEOUT_SEC})")

    args = parser.parse_args()

    global WORKFLOW_TIMEOUT_SEC
    WORKFLOW_TIMEOUT_SEC = args.timeout

    run_evaluation(
        results_file=args.results,
        api_url=args.api_url,
        dry_run=args.dry_run,
        max_configs=args.max_configs,
        output_file=args.output,
        skip_reset=args.skip_reset,
    )


if __name__ == "__main__":
    main()