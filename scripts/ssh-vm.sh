#!/usr/bin/env bash
###############################################################################
# ssh-vm.sh — SSH into a k8s VM interactively (reads from cluster-config.json)
#
# Usage:
#   sudo bash ssh-vm.sh cp            # first control-plane
#   sudo bash ssh-vm.sh k8s-tiny-3    # by full VM name
#   sudo bash ssh-vm.sh 10.10.10.14   # by IP
#   sudo bash ssh-vm.sh 5             # by number (from list)
#   sudo bash ssh-vm.sh               # interactive picker
#   sudo bash ssh-vm.sh -c [config]   # specify config file
###############################################################################
set -euo pipefail

CONFIG_FILE="cluster-config.json"
TARGET=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CONFIG_FILE="$2"; shift 2 ;;
        *)  TARGET="$1"; shift ;;
    esac
done

[[ ! -f "$CONFIG_FILE" ]] && { echo "Config not found: ${CONFIG_FILE}"; exit 1; }

SSH_KEY="/root/.ssh/id_k8s"
VM_USER=$(jq -r '.vm_user // "ubuntu"' "$CONFIG_FILE")
VM_PREFIX=$(jq -r '.vm_prefix // "k8s"' "$CONFIG_FILE")
SUBNET=$(jq -r '.network.subnet // "10.10.10"' "$CONFIG_FILE")

BOLD='\033[1m'; CYAN='\033[0;36m'; NC='\033[0m'

# ─── Expand VMs from config (same logic as builder) ─────────────────────────
declare -a NAMES=() IPS=() ROLES=() SHORTCUTS=()
IP_COUNTER=10

VM_COUNT=$(jq '.vms | length' "$CONFIG_FILE")
for pass in control_plane worker; do
    for (( vi=0; vi<VM_COUNT; vi++ )); do
        role=$(jq -r ".vms[$vi].role" "$CONFIG_FILE")
        [[ "$role" != "$pass" ]] && continue

        name=$(jq -r  ".vms[$vi].name"  "$CONFIG_FILE")
        count=$(jq -r ".vms[$vi].count" "$CONFIG_FILE")

        for (( ci=1; ci<=count; ci++ )); do
            if [[ $count -eq 1 ]]; then
                vm_name="${VM_PREFIX}-${name}"
                shortcut="${name}"
            else
                vm_name="${VM_PREFIX}-${name}-${ci}"
                shortcut="${name}-${ci}"
            fi
            ip="${SUBNET}.${IP_COUNTER}"

            NAMES+=("$vm_name")
            IPS+=("$ip")
            ROLES+=("$role")
            SHORTCUTS+=("$shortcut")

            IP_COUNTER=$((IP_COUNTER + 1))
        done
    done
done

TOTAL=${#NAMES[@]}

# ─── Resolve target to an IP ────────────────────────────────────────────────
resolve_target() {
    local t="$1"
    # By number (1-indexed)
    if [[ "$t" =~ ^[0-9]+$ ]] && [[ "$t" -ge 1 ]] && [[ "$t" -le "$TOTAL" ]]; then
        echo "${IPS[$((t - 1))]}"
        return 0
    fi
    # By IP directly
    for (( i=0; i<TOTAL; i++ )); do
        if [[ "${IPS[$i]}" == "$t" ]]; then
            echo "$t"
            return 0
        fi
    done
    # By full name or shortcut
    for (( i=0; i<TOTAL; i++ )); do
        if [[ "${NAMES[$i]}" == "$t" ]] || [[ "${SHORTCUTS[$i]}" == "$t" ]]; then
            echo "${IPS[$i]}"
            return 0
        fi
    done
    return 1
}

# ─── Interactive picker if no argument ──────────────────────────────────────
if [[ -z "$TARGET" ]]; then
    echo ""
    echo -e "${BOLD}Select a VM to connect to:${NC}"
    echo ""
    for (( i=0; i<TOTAL; i++ )); do
        role_short="worker"
        [[ "${ROLES[$i]}" == "control_plane" ]] && role_short="control-plane"
        printf "  %2d)  ${CYAN}%-22s${NC}  %-14s  %s\n" \
            "$((i + 1))" "${NAMES[$i]}" "$role_short" "${IPS[$i]}"
    done
    echo ""
    read -rp "Enter choice [1-${TOTAL}] or name: " choice
    TARGET="$choice"
fi

IP=$(resolve_target "$TARGET" 2>/dev/null) || {
    echo "Unknown VM: ${TARGET}"
    echo ""
    echo "Valid options:"
    echo "  Numbers : 1-${TOTAL}"
    echo "  Names   : ${NAMES[*]}"
    echo "  Shorts  : ${SHORTCUTS[*]}"
    echo "  IPs     : ${IPS[*]}"
    exit 1
}

# Find the display name for the resolved IP
DISPLAY_NAME="$IP"
for (( i=0; i<TOTAL; i++ )); do
    if [[ "${IPS[$i]}" == "$IP" ]]; then
        DISPLAY_NAME="${NAMES[$i]}"
        break
    fi
done

echo "Connecting to ${DISPLAY_NAME} @ ${IP}..."
exec ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -i "$SSH_KEY" \
        "${VM_USER}@${IP}"
