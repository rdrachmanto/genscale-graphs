#!/usr/bin/env bash
###############################################################################
# install-nfs-clients.sh — Install nfs-common on all worker nodes
#
# Usage:
#   sudo bash install-nfs-clients.sh
#   sudo bash install-nfs-clients.sh -c cluster-config.json
###############################################################################
set -euo pipefail

CONFIG_FILE="cluster-config.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CONFIG_FILE="$2"; shift 2 ;;
        *)  shift ;;
    esac
done

[[ ! -f "$CONFIG_FILE" ]] && { echo "Config not found: ${CONFIG_FILE}"; exit 1; }

SSH_KEY="/root/.ssh/id_k8s"
VM_USER=$(jq -r '.vm_user // "ubuntu"' "$CONFIG_FILE")
VM_PREFIX=$(jq -r '.vm_prefix // "k8s"' "$CONFIG_FILE")
SUBNET=$(jq -r '.network.subnet // "10.10.10"' "$CONFIG_FILE")

BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# ─── Collect worker nodes (same expansion logic as ssh-vm.sh) ────────────────
declare -a WORKER_NAMES=() WORKER_IPS=()
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
            else
                vm_name="${VM_PREFIX}-${name}-${ci}"
            fi
            ip="${SUBNET}.${IP_COUNTER}"

            # Only collect workers
            if [[ "$role" == "worker" ]]; then
                WORKER_NAMES+=("$vm_name")
                WORKER_IPS+=("$ip")
            fi

            IP_COUNTER=$((IP_COUNTER + 1))
        done
    done
done

TOTAL=${#WORKER_NAMES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "No worker nodes found in ${CONFIG_FILE}"
    exit 1
fi

echo ""
echo -e "${BOLD}Installing nfs-common on ${TOTAL} worker node(s):${NC}"
echo ""

PASS=0
FAIL=0

for (( i=0; i<TOTAL; i++ )); do
    name="${WORKER_NAMES[$i]}"
    ip="${WORKER_IPS[$i]}"

    printf "  ${CYAN}%-25s${NC} (%s) ... " "$name" "$ip"

    if ssh -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           -o LogLevel=ERROR \
           -o ConnectTimeout=10 \
           -i "$SSH_KEY" \
           "${VM_USER}@${ip}" \
           "sudo apt-get install -y nfs-common > /dev/null 2>&1"; then
        echo -e "${GREEN}OK${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAILED${NC}"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo -e "${BOLD}Done: ${GREEN}${PASS} succeeded${NC}, ${RED}${FAIL} failed${NC}"
echo ""

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
