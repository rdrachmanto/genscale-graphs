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

DEADLINE_VIOLATION_PENALTY = 5.0  # Penalty multiplier for exceeding deadline when optimizing cost
COST_VIOLATION_PENALTY = 3.0  # Penalty multiplier for exceeding cost cap when optimizing deadline


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
            "machineId": "xeon-c2-m4-02",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.12,
            "allocatable": {
                "cpu": "2",
                "memory": "4Gi"
            }
        },
        {
            "machineId": "epyc-c1-m8-03",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.12,
            "allocatable": {
                "cpu": "1",
                "memory": "8Gi"
            }
        },
        {
            "machineId": "xeon-c3-m6-04",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.16,
            "allocatable": {
                "cpu": "3",
                "memory": "6Gi"
            }
        },
        {
            "machineId": "epyc-c2-m12-05",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.18,
            "allocatable": {
                "cpu": "2",
                "memory": "12Gi"
            }
        },
        {
            "machineId": "xeon-c4-m8-06",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.20,
            "allocatable": {
                "cpu": "4",
                "memory": "8Gi"
            }
        },
        {
            "machineId": "epyc-c3-m16-07",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.24,
            "allocatable": {
                "cpu": "3",
                "memory": "16Gi"
            }
        },
        {
            "machineId": "xeon-c5-m10-08",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.26,
            "allocatable": {
                "cpu": "5",
                "memory": "10Gi"
            }
        },
        {
            "machineId": "epyc-c4-m20-09",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.30,
            "allocatable": {
                "cpu": "4",
                "memory": "20Gi"
            }
        },
        {
            "machineId": "xeon-c6-m12-10",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.32,
            "allocatable": {
                "cpu": "6",
                "memory": "12Gi"
            }
        },
        {
            "machineId": "epyc-c5-m24-11",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.36,
            "allocatable": {
                "cpu": "5",
                "memory": "24Gi"
            }
        },
        {
            "machineId": "xeon-c7-m14-12",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.38,
            "allocatable": {
                "cpu": "7",
                "memory": "14Gi"
            }
        },
        {
            "machineId": "epyc-c6-m28-13",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.42,
            "allocatable": {
                "cpu": "6",
                "memory": "28Gi"
            }
        },
        {
            "machineId": "xeon-c8-m16-14",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.44,
            "allocatable": {
                "cpu": "8",
                "memory": "16Gi"
            }
        },
        {
            "machineId": "epyc-c7-m32-15",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.48,
            "allocatable": {
                "cpu": "7",
                "memory": "32Gi"
            }
        },
        {
            "machineId": "xeon-c9-m18-16",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.50,
            "allocatable": {
                "cpu": "9",
                "memory": "18Gi"
            }
        },
        {
            "machineId": "epyc-c8-m36-17",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.54,
            "allocatable": {
                "cpu": "8",
                "memory": "36Gi"
            }
        },
        {
            "machineId": "xeon-c10-m20-18",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.56,
            "allocatable": {
                "cpu": "10",
                "memory": "20Gi"
            }
        },
        {
            "machineId": "epyc-c9-m40-19",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 0.60,
            "allocatable": {
                "cpu": "9",
                "memory": "40Gi"
            }
        },
        {
            "machineId": "xeon-c11-m22-20",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 0.62,
            "allocatable": {
                "cpu": "11",
                "memory": "22Gi"
            }
        },
        {
            "machineId": "epyc-c16-m32-21",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 1.60,
            "allocatable": {
                "cpu": "16",
                "memory": "32Gi"
            }
        },
        {
            "machineId": "xeon-c12-m48-22",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 1.70,
            "allocatable": {
                "cpu": "12",
                "memory": "48Gi"
            }
        },
        {
            "machineId": "epyc-c18-m36-23",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 1.80,
            "allocatable": {
                "cpu": "18",
                "memory": "36Gi"
            }
        },
        {
            "machineId": "xeon-c14-m56-24",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 1.90,
            "allocatable": {
                "cpu": "14",
                "memory": "56Gi"
            }
        },
        {
            "machineId": "epyc-c20-m40-25",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 2.00,
            "allocatable": {
                "cpu": "20",
                "memory": "40Gi"
            }
        },
        {
            "machineId": "xeon-c16-m64-26",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.10,
            "allocatable": {
                "cpu": "16",
                "memory": "64Gi"
            }
        },
        {
            "machineId": "epyc-c22-m44-27",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 2.20,
            "allocatable": {
                "cpu": "22",
                "memory": "44Gi"
            }
        },
        {
            "machineId": "xeon-c18-m72-28",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.30,
            "allocatable": {
                "cpu": "18",
                "memory": "72Gi"
            }
        },
        {
            "machineId": "epyc-c24-m48-29",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 2.40,
            "allocatable": {
                "cpu": "24",
                "memory": "48Gi"
            }
        },
        {
            "machineId": "xeon-c20-m80-30",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.50,
            "allocatable": {
                "cpu": "20",
                "memory": "80Gi"
            }
        },
        {
            "machineId": "epyc-c26-m52-31",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 2.60,
            "allocatable": {
                "cpu": "26",
                "memory": "52Gi"
            }
        },
        {
            "machineId": "xeon-c22-m88-32",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.70,
            "allocatable": {
                "cpu": "22",
                "memory": "88Gi"
            }
        },
        {
            "machineId": "epyc-c28-m56-33",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 2.80,
            "allocatable": {
                "cpu": "28",
                "memory": "56Gi"
            }
        },
        {
            "machineId": "xeon-c24-m96-34",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 2.90,
            "allocatable": {
                "cpu": "24",
                "memory": "96Gi"
            }
        },
        {
            "machineId": "epyc-c30-m60-35",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 3.00,
            "allocatable": {
                "cpu": "30",
                "memory": "60Gi"
            }
        },
        {
            "machineId": "xeon-c26-m104-36",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 3.10,
            "allocatable": {
                "cpu": "26",
                "memory": "104Gi"
            }
        },
        {
            "machineId": "epyc-c32-m64-37",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 3.20,
            "allocatable": {
                "cpu": "32",
                "memory": "64Gi"
            }
        },
        {
            "machineId": "xeon-c28-m112-38",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 3.30,
            "allocatable": {
                "cpu": "28",
                "memory": "112Gi"
            }
        },
        {
            "machineId": "epyc-c34-m68-39",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 3.40,
            "allocatable": {
                "cpu": "34",
                "memory": "68Gi"
            }
        },
        {
            "machineId": "xeon-c30-m120-40",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 3.50,
            "allocatable": {
                "cpu": "30",
                "memory": "120Gi"
            }
        },
        {
            "machineId": "epyc-c48-m64-41",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 5.50,
            "allocatable": {
                "cpu": "48",
                "memory": "64Gi"
            }
        },
        {
            "machineId": "xeon-c36-m144-42",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 5.80,
            "allocatable": {
                "cpu": "36",
                "memory": "144Gi"
            }
        },
        {
            "machineId": "epyc-c52-m72-43",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 6.10,
            "allocatable": {
                "cpu": "52",
                "memory": "72Gi"
            }
        },
        {
            "machineId": "xeon-c40-m160-44",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 6.40,
            "allocatable": {
                "cpu": "40",
                "memory": "160Gi"
            }
        },
        {
            "machineId": "epyc-c56-m80-45",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 6.70,
            "allocatable": {
                "cpu": "56",
                "memory": "80Gi"
            }
        },
        {
            "machineId": "xeon-c44-m176-46",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 7.00,
            "allocatable": {
                "cpu": "44",
                "memory": "176Gi"
            }
        },
        {
            "machineId": "epyc-c60-m88-47",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 7.30,
            "allocatable": {
                "cpu": "60",
                "memory": "88Gi"
            }
        },
        {
            "machineId": "xeon-c48-m192-48",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 7.60,
            "allocatable": {
                "cpu": "48",
                "memory": "192Gi"
            }
        },
        {
            "machineId": "epyc-c64-m96-49",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 7.90,
            "allocatable": {
                "cpu": "64",
                "memory": "96Gi"
            }
        },
        {
            "machineId": "xeon-c52-m208-50",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 8.20,
            "allocatable": {
                "cpu": "52",
                "memory": "208Gi"
            }
        },
        {
            "machineId": "epyc-c68-m104-51",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 8.50,
            "allocatable": {
                "cpu": "68",
                "memory": "104Gi"
            }
        },
        {
            "machineId": "xeon-c56-m224-52",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 8.80,
            "allocatable": {
                "cpu": "56",
                "memory": "224Gi"
            }
        },
        {
            "machineId": "epyc-c72-m112-53",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 9.10,
            "allocatable": {
                "cpu": "72",
                "memory": "112Gi"
            }
        },
        {
            "machineId": "xeon-c60-m240-54",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 9.40,
            "allocatable": {
                "cpu": "60",
                "memory": "240Gi"
            }
        },
        {
            "machineId": "epyc-c76-m120-55",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 9.70,
            "allocatable": {
                "cpu": "76",
                "memory": "120Gi"
            }
        },
        {
            "machineId": "xeon-c64-m256-56",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 10.00,
            "allocatable": {
                "cpu": "64",
                "memory": "256Gi"
            }
        },
        {
            "machineId": "epyc-c80-m128-57",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 10.30,
            "allocatable": {
                "cpu": "80",
                "memory": "128Gi"
            }
        },
        {
            "machineId": "xeon-c72-m192-58",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 10.60,
            "allocatable": {
                "cpu": "72",
                "memory": "192Gi"
            }
        },
        {
            "machineId": "epyc-c88-m144-59",
            "hardwareId": "CPU:AMD EPYC 7352 24-Core Processor,Memory:257497MiB,Cores:96",
            "costPerHour": 10.90,
            "allocatable": {
                "cpu": "88",
                "memory": "144Gi"
            }
        },
        {
            "machineId": "xeon-c80-m224-60",
            "hardwareId": "CPU:Intel(R) Xeon(R) Gold 6240R,Memory:192002MiB,Cores:96",
            "costPerHour": 11.20,
            "allocatable": {
                "cpu": "80",
                "memory": "224Gi"
            }
        }
    ],
    "workflows": [
        "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml", "dna-2750.yaml",
        "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml", "dna-100.yaml",
        "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml", "dna-850.yaml"
    ],
    
    "constraints": {
        "priority": "cost",  # "cost" or "deadline" - what to optimize for
        "deadline": 200,  # minutes - target deadline (soft constraint)
        "costCap": 500  # dollars - target cost cap (soft constraint)
    },
    
    "simulationConfig": {
        "deltaTime": 10000,
        "speedMultiplier": 500,
        "schedulingInterval": 20000,
        "schedulerPolicy": "fifo",
        "disableMockFilesystem": True
    },
    
    "optimizationConfig": {
        "nIterations": 300,  # Total optimization iterations (increased from 200 for better convergence)
        "nInitialPoints": 60,  # Random exploration before optimization (matches dimensionality)
        "acquisitionFunc": "EI",  # Expected Improvement
        "xi": 0.01,  # Exploration parameter for EI (lower = more exploitation)
        "randomState": 42
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
            return (value - target) / target * DEADLINE_VIOLATION_PENALTY
        return 0.0
    else:  # priority == "deadline"
        # When optimizing for deadline, penalize cost violations
        if value > target:
            return (value - target) / target * COST_VIOLATION_PENALTY
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
    print(f"Initial Random Points: {CONFIG['optimizationConfig']['nInitialPoints']}")
    print(f"Acquisition Function: {CONFIG['optimizationConfig']['acquisitionFunc']} (xi={CONFIG['optimizationConfig']['xi']})")
    print("="*80)
    
    # Define search space - binary choice for each machine (0 or 1)
    # Since each machine is unique, we can only include it or not
    search_space = [Integer(0, 1, name=f"machine_{i}") 
                    for i in range(len(CONFIG["availableMachines"]))]
    
    # Run Bayesian Optimization
    print("\nStarting optimization...\n")
    
    result = gp_minimize(
        objective_function,
        search_space,
        n_calls=CONFIG["optimizationConfig"]["nIterations"],
        n_initial_points=CONFIG["optimizationConfig"]["nInitialPoints"],
        acq_func=CONFIG["optimizationConfig"]["acquisitionFunc"],
        xi=CONFIG["optimizationConfig"]["xi"],  # Explicit exploration parameter
        random_state=CONFIG["optimizationConfig"]["randomState"],
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