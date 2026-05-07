#!/usr/bin/env bash
###############################################################################
# list-vms.sh — Show status of all k8s VMs (reads from cluster-config.json)
#
# Usage:
#   sudo bash list-vms.sh [config.json]
###############################################################################
set -euo pipefail

CONFIG_FILE="${1:-cluster-config.json}"
[[ ! -f "$CONFIG_FILE" ]] && { echo "Config not found: ${CONFIG_FILE}"; exit 1; }

SSH_KEY="/root/.ssh/id_k8s"
VM_USER=$(jq -r '.vm_user // "ubuntu"' "$CONFIG_FILE")
VM_PREFIX=$(jq -r '.vm_prefix // "k8s"' "$CONFIG_FILE")
SUBNET=$(jq -r '.network.subnet // "10.10.10"' "$CONFIG_FILE")

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=3 -o LogLevel=ERROR -i "$SSH_KEY")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── Expand VMs from config (same logic as builder) ─────────────────────────
declare -a NAMES=() IPS=() ROLES=() VCPUS=() RAMS=() DISKS=()
IP_COUNTER=10

VM_COUNT=$(jq '.vms | length' "$CONFIG_FILE")
for pass in control_plane worker; do
    for (( vi=0; vi<VM_COUNT; vi++ )); do
        role=$(jq -r ".vms[$vi].role" "$CONFIG_FILE")
        [[ "$role" != "$pass" ]] && continue

        name=$(jq -r  ".vms[$vi].name"    "$CONFIG_FILE")
        vcpus=$(jq -r ".vms[$vi].vcpus"   "$CONFIG_FILE")
        ram=$(jq -r   ".vms[$vi].ram_mb"  "$CONFIG_FILE")
        disk=$(jq -r  ".vms[$vi].disk_gb" "$CONFIG_FILE")
        count=$(jq -r ".vms[$vi].count"   "$CONFIG_FILE")

        for (( ci=1; ci<=count; ci++ )); do
            if [[ $count -eq 1 ]]; then
                vm_name="${VM_PREFIX}-${name}"
            else
                vm_name="${VM_PREFIX}-${name}-${ci}"
            fi
            ip="${SUBNET}.${IP_COUNTER}"

            NAMES+=("$vm_name")
            IPS+=("$ip")
            ROLES+=("$role")
            VCPUS+=("$vcpus")
            RAMS+=("$ram")
            DISKS+=("$disk")

            IP_COUNTER=$((IP_COUNTER + 1))
        done
    done
done

TOTAL=${#NAMES[@]}
CP_IP="${IPS[0]}"

# ─── Display table ──────────────────────────────────────────────────────────
echo ""
printf "${BOLD}%-22s %-14s %-16s %-8s %-6s %-15s %-8s %-10s %-10s${NC}\n" \
    "VM" "ROLE" "IP" "vCPU" "RAM" "LIBVIRT" "SSH" "CPU%" "MEM"
printf '%.0s─' {1..115}; echo ""

for (( i=0; i<TOTAL; i++ )); do
    vm="${NAMES[$i]}"
    ip="${IPS[$i]}"
    role="${ROLES[$i]}"
    vcpu="${VCPUS[$i]}"
    ram="${RAMS[$i]}"

    role_short="worker"
    [[ "$role" == "control_plane" ]] && role_short="control-plane"

    # Libvirt state
    lib_state=$(virsh domstate "$vm" 2>/dev/null || echo "not found")
    lib_state=$(echo "$lib_state" | tr -d '\n')
    case "$lib_state" in
        running)    lib_color="${GREEN}${lib_state}${NC}" ;;
        "shut off") lib_color="${RED}${lib_state}${NC}" ;;
        *)          lib_color="${YELLOW}${lib_state}${NC}" ;;
    esac

    # SSH + resource usage
    ssh_ok="—"
    cpu_pct="—"
    mem_use="—"
    if [[ "$lib_state" == "running" ]]; then
        if stats=$(ssh "${SSH_OPTS[@]}" "${VM_USER}@${ip}" \
            "echo OK; grep 'cpu ' /proc/stat | awk '{u=\$2+\$4; t=\$2+\$3+\$4+\$5+\$6+\$7+\$8; printf \"%.0f%%\n\", u/t*100}'; free -m | awk '/Mem:/{printf \"%dM/%dM\", \$3, \$2}'" 2>/dev/null); then
            ssh_ok="${GREEN}✓${NC}"
            cpu_pct=$(echo "$stats" | sed -n '2p')
            mem_use=$(echo "$stats" | sed -n '3p')
        else
            ssh_ok="${RED}✗${NC}"
        fi
    fi

    printf "%-22s ${CYAN}%-14s${NC} %-16s %-8s %-6s %-22b %-15b %-10s %-10s\n" \
        "$vm" "$role_short" "$ip" "$vcpu" "${ram}M" "$lib_color" "$ssh_ok" "$cpu_pct" "$mem_use"
done

# ─── Summary line ───────────────────────────────────────────────────────────
printf '%.0s─' {1..115}; echo ""
total_vcpu=0; total_ram=0
for (( i=0; i<TOTAL; i++ )); do
    total_vcpu=$((total_vcpu + VCPUS[i]))
    total_ram=$((total_ram + RAMS[i]))
done
printf "${BOLD}%-22s %-14s %-16s %-8s %-6s${NC}\n" \
    "TOTAL: ${TOTAL} VMs" "" "" "$total_vcpu" "$((total_ram))M"

# ─── Kubernetes cluster status ───────────────────────────────────────────────
echo ""
if ssh "${SSH_OPTS[@]}" "${VM_USER}@${CP_IP}" "kubectl get nodes" &>/dev/null 2>&1; then
    printf "${BOLD}Kubernetes Cluster Status:${NC}\n"
    ssh "${SSH_OPTS[@]}" "${VM_USER}@${CP_IP}" "kubectl get nodes -o wide" 2>/dev/null
else
    printf "${YELLOW}Kubernetes cluster not reachable from control plane.${NC}\n"
fi
echo ""
