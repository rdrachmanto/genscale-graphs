#!/usr/bin/env python3
"""
Bayesian Optimization for Cluster Sizing
Uses simulation API to evaluate cluster configurations
"""

import json
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Try to import scikit-optimize, install if needed
try:
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective
except ImportError:
    print("Installing scikit-optimize...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-optimize', '--break-system-packages'])
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'matplotlib', '--break-system-packages'])
    import matplotlib.pyplot as plt


# ============================================================================
# EDITABLE CONFIGURATION
# ============================================================================

CONFIG = {
    "apiUrl": "http://localhost:3333",
    
    "availableMachines": [
        {
            "machineId": "epyc-c1-m4-01",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.08,
            "allocatable": {
                "cpu": "1",
                "memory": "4Gi"
            }
        },
        {
            "machineId": "epyc-c3-m12-02",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.28,
            "allocatable": {
                "cpu": "3",
                "memory": "12Gi"
            }
        },
        {
            "machineId": "xeon-c4-m8-03",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.32,
            "allocatable": {
                "cpu": "4",
                "memory": "16Gi"
            }
        },
        {
            "machineId": "epyc-c4-m16-04",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.4,
            "allocatable": {
                "cpu": "4",
                "memory": "16Gi"
            }
        },
        {
            "machineId": "xeon-c5-m20-05",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.44,
            "allocatable": {
                "cpu": "5",
                "memory": "20Gi"
            }
        },
        {
            "machineId": "epyc-c6-m12-06",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.48,
            "allocatable": {
                "cpu": "6",
                "memory": "24Gi"
            }
        },
        {
            "machineId": "xeon-c6-m24-7",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.56,
            "allocatable": {
                "cpu": "6",
                "memory": "24Gi"
            }
        },
        {
            "machineId": "xeon-c8-m16-8",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.64,
            "allocatable": {
                "cpu": "8",
                "memory": "32Gi"
            }
        },
        {
            "machineId": "xeon-c16-m32-9",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 1.28,
            "allocatable": {
                "cpu": "16",
                "memory": "64Gi"
            }
        },
        {
            "machineId": "xeon-c36-m144-10",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.88,
            "allocatable": {
                "cpu": "36",
                "memory": "144Gi"
            }
        },
        {
            "machineId": "epyc-c40-m160-11",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 3.2,
            "allocatable": {
                "cpu": "40",
                "memory": "160Gi"
            }
        },
        {
            "machineId": "xeon-c44-m176-12",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 3.52,
            "allocatable": {
                "cpu": "44",
                "memory": "176Gi"
            }
        }
    ],
    "workflows": [
        "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml",
        "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml"
    ],
    
    "constraints": {
        "priority": "cost",  # "cost" or "deadline" - what to optimize for
        "deadline": 720,  # minutes - target deadline (soft constraint)
        "costCap": 50  # dollars - target cost cap (soft constraint)
    },
    
    "simulationConfig": {
        "deltaTime": 10000,
        "speedMultiplier": 500,
        "schedulingInterval": 20000,
        "schedulerPolicy": "fifo",
        "disableMockFilesystem": True
    },
    
    "optimizationConfig": {
        "nIterations": 300,  # Total optimization iterations
        "nInitialPoints": 10,  # Random exploration before optimization
        "acquisitionFunc": "EI",  # Expected Improvement
        "randomState": 42,
        "startingPoint": None  # Optional: e.g., [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        # "startingPoint": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    
    "outputConfig": {
        "saveResults": True,
        "resultsFile": "optimization_results.json",
        "plotConvergence": True,
        "plotFile": "convergence.png"
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_memory(memory_str: str) -> float:
    """Convert memory string (e.g., '12Gi') to bytes"""
    units = {
        'Ki': 1024,
        'Mi': 1024**2,
        'Gi': 1024**3,
        'Ti': 1024**4,
        'K': 1000,
        'M': 1000**2,
        'G': 1000**3,
        'T': 1000**4
    }
    
    for unit, multiplier in units.items():
        if memory_str.endswith(unit):
            value = float(memory_str[:-len(unit)])
            return value * multiplier
    
    # Assume bytes if no unit
    return float(memory_str)


def build_cluster_config(machine_counts: List[int], available_machines: List[Dict]) -> Dict:
    """Build cluster configuration from machine counts"""
    nodes = []
    
    for idx, count in enumerate(machine_counts):
        if count > 0:  # Only include machines with count > 0
            machine = available_machines[idx]
            # Each machine is unique, so we can only use it once
            # count will be binary (0 or 1)
            nodes.append({
                "name": machine["machineId"],
                "hardwareId": machine["hardwareId"],
                "cpu": machine["allocatable"]["cpu"],
                "memory": machine["allocatable"]["memory"]
            })
    
    return {"nodes": nodes}


def calculate_cluster_cost(machine_counts: List[int], available_machines: List[Dict], runtime_minutes: float) -> float:
    """Calculate total cost of cluster configuration"""
    total_cost = 0.0
    runtime_hours = runtime_minutes / 60.0
    
    for idx, count in enumerate(machine_counts):
        if count > 0:
            machine = available_machines[idx]
            total_cost += machine["costPerHour"] * runtime_hours
    
    return total_cost


def run_simulation(cluster_config: Dict, workflows: List[str], sim_config: Dict, api_url: str) -> Dict:
    """Run simulation via API and return results"""
    payload = {
        "workflows": workflows,
        "config": {
            **sim_config,
            "clusterConfig": cluster_config
        }
    }
    
    try:
        response = requests.post(
            f"{api_url}/api/simulation/run",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            return result["data"]
        else:
            print(f"Simulation failed: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None


def calculate_penalty(value: float, target: float, priority: str) -> float:
    """Calculate penalty for exceeding constraints (soft constraints)"""
    if priority == "cost":
        # When optimizing for cost, heavily penalize deadline violations
        if value > target:
            return (value - target) / target * 10.0  # 10x penalty multiplier
        return 0.0
    else:  # priority == "deadline"
        # When optimizing for deadline, heavily penalize cost violations
        if value > target:
            return (value - target) / target * 10.0
        return 0.0


# ============================================================================
# OPTIMIZATION OBJECTIVE
# ============================================================================

class OptimizationState:
    """Track optimization progress"""
    def __init__(self):
        self.iteration = 0
        self.evaluations = []
        self.best_score = float('inf')
        self.best_config = None
        self.start_time = datetime.now()


optimization_state = OptimizationState()


def objective_function(machine_counts: List[int]) -> float:
    """
    Objective function for Bayesian Optimization
    Returns score to minimize (lower is better)
    """
    optimization_state.iteration += 1
    iteration = optimization_state.iteration
    
    # Skip if no machines selected
    if sum(machine_counts) == 0:
        print(f"[{iteration}] No machines selected - skipping")
        return 1e10  # Very high penalty
    
    # Build cluster configuration
    cluster_config = build_cluster_config(machine_counts, CONFIG["availableMachines"])
    num_machines = len(cluster_config["nodes"])
    
    print(f"\n{'='*80}")
    print(f"[{iteration}] Evaluating configuration with {num_machines} machines")
    print(f"Machine selection: {machine_counts}")
    
    # Run simulation
    result = run_simulation(
        cluster_config,
        CONFIG["workflows"],
        CONFIG["simulationConfig"],
        CONFIG["apiUrl"]
    )
    
    if result is None:
        print(f"[{iteration}] Simulation failed")
        return 1e10
    
    # Extract metrics
    makespan_ms = result["summary"]["makespan"]
    makespan_minutes = makespan_ms / (1000 * 60)
    
    # Calculate cost
    total_cost = calculate_cluster_cost(
        machine_counts,
        CONFIG["availableMachines"],
        makespan_minutes
    )
    
    # Get constraints
    priority = CONFIG["constraints"]["priority"]
    deadline_target = CONFIG["constraints"]["deadline"]
    cost_target = CONFIG["constraints"]["costCap"]
    
    # Calculate objective based on priority
    if priority == "cost":
        # Minimize cost while meeting deadline
        objective = total_cost
        deadline_penalty = calculate_penalty(makespan_minutes, deadline_target, "cost")
        objective += deadline_penalty * cost_target  # Scale penalty by cost
        
        print(f"[{iteration}] Cost: ${total_cost:.2f}, Makespan: {makespan_minutes:.2f}m, Deadline Penalty: {deadline_penalty:.2f}")
        
    else:  # priority == "deadline"
        # Minimize makespan while meeting cost cap
        objective = makespan_minutes
        cost_penalty = calculate_penalty(total_cost, cost_target, "deadline")
        objective += cost_penalty * deadline_target  # Scale penalty by deadline
        
        print(f"[{iteration}] Makespan: {makespan_minutes:.2f}m, Cost: ${total_cost:.2f}, Cost Penalty: {cost_penalty:.2f}")
    
    print(f"[{iteration}] Objective Score: {objective:.2f}")
    
    # Track evaluation (convert numpy types to native Python types for JSON serialization)
    evaluation = {
        "iteration": int(iteration),
        "machine_counts": [int(x) for x in machine_counts],
        "num_machines": int(num_machines),
        "makespan_minutes": float(makespan_minutes),
        "total_cost": float(total_cost),
        "objective_score": float(objective),
        "meets_deadline": bool(makespan_minutes <= deadline_target),
        "meets_cost_cap": bool(total_cost <= cost_target),
        "cluster_config": cluster_config,
        "full_result": {
            "summary": result["summary"],
            "metadata": result["metadata"]
        }
    }
    optimization_state.evaluations.append(evaluation)
    
    # Track best
    if objective < optimization_state.best_score:
        optimization_state.best_score = objective
        optimization_state.best_config = evaluation
        print(f"[{iteration}] *** NEW BEST CONFIGURATION ***")
    
    return objective


# ============================================================================
# MAIN OPTIMIZATION
# ============================================================================

def run_optimization():
    """Run Bayesian Optimization for cluster sizing"""
    print("="*80)
    print("BAYESIAN OPTIMIZATION FOR CLUSTER SIZING")
    print("="*80)
    print(f"Start Time: {optimization_state.start_time}")
    print(f"API URL: {CONFIG['apiUrl']}")
    print(f"Workflows: {len(CONFIG['workflows'])} workflows")
    print(f"Available Machines: {len(CONFIG['availableMachines'])} machine types")
    print(f"Priority: {CONFIG['constraints']['priority']}")
    print(f"Deadline Target: {CONFIG['constraints']['deadline']} minutes")
    print(f"Cost Cap Target: ${CONFIG['constraints']['costCap']}")
    print(f"Optimization Iterations: {CONFIG['optimizationConfig']['nIterations']}")
    print("="*80)
    
    # Define search space - binary choice for each machine (0 or 1)
    # Since each machine is unique, we can only include it or not
    search_space = [Integer(0, 1, name=f"machine_{i}") 
                    for i in range(len(CONFIG["availableMachines"]))]
    
    # Run Bayesian Optimization
    print("\nStarting optimization...\n")
    
    # Prepare x0 parameter if starting point is provided
    x0 = None
    starting_point = CONFIG["optimizationConfig"].get("startingPoint")
    if starting_point is not None:
        x0 = [starting_point]
    
    result = gp_minimize(
        objective_function,
        search_space,
        n_calls=CONFIG["optimizationConfig"]["nIterations"],
        n_initial_points=CONFIG["optimizationConfig"]["nInitialPoints"],
        acq_func=CONFIG["optimizationConfig"]["acquisitionFunc"],
        random_state=CONFIG["optimizationConfig"]["randomState"],
        x0=x0,
        verbose=False
    )
    
    # Get best configuration
    best_eval = optimization_state.best_config
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now()}")
    print(f"Total Duration: {datetime.now() - optimization_state.start_time}")
    print(f"Total Evaluations: {len(optimization_state.evaluations)}")
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND")
    print("="*80)
    print(f"Objective Score: {best_eval['objective_score']:.2f}")
    print(f"Makespan: {best_eval['makespan_minutes']:.2f} minutes")
    print(f"Total Cost: ${best_eval['total_cost']:.2f}")
    print(f"Number of Machines: {best_eval['num_machines']}")
    print(f"Meets Deadline: {best_eval['meets_deadline']}")
    print(f"Meets Cost Cap: {best_eval['meets_cost_cap']}")
    print("\nSelected Machines:")
    for node in best_eval['cluster_config']['nodes']:
        machine_info = next(m for m in CONFIG["availableMachines"] if m["machineId"] == node["name"])
        print(f"  - {node['name']}: CPU={node['cpu']}, Memory={node['memory']}, Cost=${machine_info['costPerHour']:.2f}/hr")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Filter out failed evaluations
    successful_evals = [e for e in optimization_state.evaluations if 'makespan_minutes' in e]
    failed_count = len(optimization_state.evaluations) - len(successful_evals)
    
    if failed_count > 0:
        print(f"Note: {failed_count} evaluations failed during optimization")
    
    if successful_evals:
        all_makespans = [e['makespan_minutes'] for e in successful_evals]
        all_costs = [e['total_cost'] for e in successful_evals]
        
        print(f"Successful Evaluations: {len(successful_evals)}/{len(optimization_state.evaluations)}")
        print(f"Makespan Range: {min(all_makespans):.2f} - {max(all_makespans):.2f} minutes")
        print(f"Cost Range: ${min(all_costs):.2f} - ${max(all_costs):.2f}")
        print(f"Configurations Meeting Deadline: {sum(e['meets_deadline'] for e in successful_evals)}/{len(successful_evals)}")
        print(f"Configurations Meeting Cost Cap: {sum(e['meets_cost_cap'] for e in successful_evals)}/{len(successful_evals)}")
    else:
        print("No successful evaluations to summarize")
    
    # Save results
    if CONFIG["outputConfig"]["saveResults"]:
        # Filter successful evaluations for statistics
        successful_evals = [e for e in optimization_state.evaluations if 'makespan_minutes' in e]
        
        summary_stats = {
            "total_evaluations": len(optimization_state.evaluations),
            "successful_evaluations": len(successful_evals),
            "failed_evaluations": len(optimization_state.evaluations) - len(successful_evals)
        }
        
        if successful_evals:
            all_makespans = [e['makespan_minutes'] for e in successful_evals]
            all_costs = [e['total_cost'] for e in successful_evals]
            
            summary_stats.update({
                "makespan_range": [float(min(all_makespans)), float(max(all_makespans))],
                "cost_range": [float(min(all_costs)), float(max(all_costs))],
                "configurations_meeting_deadline": sum(e['meets_deadline'] for e in successful_evals),
                "configurations_meeting_cost_cap": sum(e['meets_cost_cap'] for e in successful_evals)
            })
        
        output_data = {
            "config": CONFIG,
            "optimization_results": {
                "best_configuration": best_eval,
                "all_evaluations": optimization_state.evaluations,
                "summary_stats": summary_stats,
                "start_time": optimization_state.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - optimization_state.start_time).total_seconds()
            }
        }
        
        with open(CONFIG["outputConfig"]["resultsFile"], 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to: {CONFIG['outputConfig']['resultsFile']}")
    
    # Plot convergence
    if CONFIG["outputConfig"]["plotConvergence"]:
        try:
            # Filter successful evaluations for plotting
            successful_evals = [e for e in optimization_state.evaluations if 'makespan_minutes' in e]
            
            if not successful_evals:
                print("No successful evaluations to plot")
                return best_eval
            
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Convergence
            plt.subplot(1, 2, 1)
            iterations = [e['iteration'] for e in successful_evals]
            scores = [e['objective_score'] for e in successful_evals]
            plt.plot(iterations, scores, 'b-', alpha=0.6, label='Objective Score')
            
            # Plot running minimum
            running_min = []
            current_min = float('inf')
            for score in scores:
                current_min = min(current_min, score)
                running_min.append(current_min)
            plt.plot(iterations, running_min, 'r-', linewidth=2, label='Best Score')
            
            plt.xlabel('Iteration')
            plt.ylabel('Objective Score (lower is better)')
            plt.title('Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Cost vs Makespan scatter
            plt.subplot(1, 2, 2)
            costs = [e['total_cost'] for e in successful_evals]
            makespans = [e['makespan_minutes'] for e in successful_evals]
            
            # Color by whether constraints are met
            colors = []
            for e in successful_evals:
                if e['meets_deadline'] and e['meets_cost_cap']:
                    colors.append('green')
                elif e['meets_deadline'] or e['meets_cost_cap']:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            plt.scatter(makespans, costs, c=colors, alpha=0.6, s=50)
            
            # Mark best configuration
            plt.scatter([best_eval['makespan_minutes']], [best_eval['total_cost']], 
                       c='blue', s=200, marker='*', edgecolors='black', linewidths=2,
                       label='Best Config', zorder=5)
            
            # Add constraint lines
            plt.axvline(x=CONFIG['constraints']['deadline'], color='red', linestyle='--', 
                       alpha=0.5, label=f"Deadline Target ({CONFIG['constraints']['deadline']}m)")
            plt.axhline(y=CONFIG['constraints']['costCap'], color='purple', linestyle='--', 
                       alpha=0.5, label=f"Cost Cap Target (${CONFIG['constraints']['costCap']})")
            
            plt.xlabel('Makespan (minutes)')
            plt.ylabel('Total Cost ($)')
            plt.title('Cost vs. Makespan Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(CONFIG["outputConfig"]["plotFile"], dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to: {CONFIG['outputConfig']['plotFile']}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    return best_eval


if __name__ == "__main__":
    best_configuration = run_optimization()