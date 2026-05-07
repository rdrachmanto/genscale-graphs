#!/usr/bin/env bash
###############################################################################
# one-click-k8s.sh
#
# Creates KVM virtual machines (Ubuntu 22.04) from a JSON config, networks
# them together, then bootstraps a Kubernetes cluster with kubeadm.
#
# Fully idempotent — safe to re-run at any point.
#
# Usage:
#   sudo bash one-click-k8s.sh [config.json]       # full run
#   sudo bash one-click-k8s.sh teardown [config.json]  # destroy everything
###############################################################################
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# ─────────────────────────── Colours / helpers ───────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
hdr()   { echo -e "\n${CYAN}${BOLD}═══ $* ═══${NC}"; }

# ─────────────────────────── Argument parsing ────────────────────────────────
ACTION="run"
CONFIG_FILE="cluster-config.json"

for arg in "$@"; do
    case "$arg" in
        teardown) ACTION="teardown" ;;
        *.json)   CONFIG_FILE="$arg" ;;
    esac
done

[[ ! -f "$CONFIG_FILE" ]] && err "Config file not found: ${CONFIG_FILE}"

# ─────────────────────────── Dependencies ────────────────────────────────────
[[ $EUID -ne 0 ]] && err "Please run as root (sudo)."

if ! command -v jq &>/dev/null; then
    info "Installing jq..."
    apt-get update -qq && apt-get install -y -qq jq >/dev/null 2>&1
fi

# ─────────────────────────── Parse JSON config ───────────────────────────────
hdr "Loading configuration from ${CONFIG_FILE}"

VM_PREFIX=$(jq -r '.vm_prefix // "k8s"'          "$CONFIG_FILE")
VM_USER=$(jq -r  '.vm_user // "ubuntu"'           "$CONFIG_FILE")
VM_PASSWORD=$(jq -r '.vm_password // "k8spass123"' "$CONFIG_FILE")
IMAGE_URL=$(jq -r '.image_url'                     "$CONFIG_FILE")
OS_VARIANT=$(jq -r '.os_variant // "ubuntu22.04"'  "$CONFIG_FILE")
K8S_VERSION=$(jq -r '.kubernetes_version // "1.29"' "$CONFIG_FILE")

NET_NAME=$(jq -r  '.network.name // "k8snet"'       "$CONFIG_FILE")
NET_BRIDGE=$(jq -r '.network.bridge // "virbr-k8s"' "$CONFIG_FILE")
SUBNET=$(jq -r    '.network.subnet // "10.10.10"'   "$CONFIG_FILE")
POD_CIDR=$(jq -r  '.network.pod_cidr // "10.244.0.0/16"' "$CONFIG_FILE")
GATEWAY="${SUBNET}.1"

IMAGE_DIR="/var/lib/libvirt/images"
SSH_KEY_PATH="/root/.ssh/id_k8s"

# ─────────────────────────── Expand VM specs ─────────────────────────────────
# Flatten the vms array (handling count > 1) into ordered parallel arrays.
# Control-plane VMs come first, workers after.
declare -a ALL_NAMES=() ALL_IPS=() ALL_MACS=() ALL_VCPUS=() ALL_RAM=()
declare -a ALL_DISK=() ALL_ROLES=()
declare -a CP_INDICES=() WORKER_INDICES=()

IP_COUNTER=10          # start from .10
MAC_COUNTER=16         # 0x10 in hex

VM_COUNT=$(jq '.vms | length' "$CONFIG_FILE")

# First pass: control_plane VMs, then workers (ensures CP is index 0)
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
            mac=$(printf "52:54:00:a8:%02x:%02x" $(( MAC_COUNTER / 256 )) $(( MAC_COUNTER % 256 )))
            idx=${#ALL_NAMES[@]}

            ALL_NAMES+=("$vm_name")
            ALL_IPS+=("$ip")
            ALL_MACS+=("$mac")
            ALL_VCPUS+=("$vcpus")
            ALL_RAM+=("$ram")
            ALL_DISK+=("$disk")
            ALL_ROLES+=("$role")

            if [[ "$role" == "control_plane" ]]; then
                CP_INDICES+=("$idx")
            else
                WORKER_INDICES+=("$idx")
            fi

            IP_COUNTER=$((IP_COUNTER + 1))
            MAC_COUNTER=$((MAC_COUNTER + 1))
        done
    done
done

TOTAL_VMS=${#ALL_NAMES[@]}
PRIMARY_CP_IDX=${CP_INDICES[0]}
CP_IP="${ALL_IPS[$PRIMARY_CP_IDX]}"
CP_NAME="${ALL_NAMES[$PRIMARY_CP_IDX]}"

info "Expanded ${TOTAL_VMS} VMs from config (${#CP_INDICES[@]} control-plane, ${#WORKER_INDICES[@]} workers)"
for (( i=0; i<TOTAL_VMS; i++ )); do
    printf "  %-20s  %2s vCPU  %6s MB RAM  %4s GB disk  %-14s  %s  %s\n" \
        "${ALL_NAMES[$i]}" "${ALL_VCPUS[$i]}" "${ALL_RAM[$i]}" \
        "${ALL_DISK[$i]}" "${ALL_ROLES[$i]}" "${ALL_IPS[$i]}" "${ALL_MACS[$i]}"
done

# ─────────────────────────── SSH config ──────────────────────────────────────
SSH_OPTS_ARR=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=10 -o LogLevel=ERROR -o ServerAliveInterval=15
    -o ServerAliveCountMax=3 -i "$SSH_KEY_PATH")

run_remote_script() {
    local ip="$1" script_path="$2" description="$3"
    info "  → ${description}"
    scp "${SSH_OPTS_ARR[@]}" "$script_path" "${VM_USER}@${ip}:/tmp/_remote_script.sh" >/dev/null 2>&1 \
        || err "Failed to SCP script to ${ip}"
    ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ip}" \
        "chmod +x /tmp/_remote_script.sh && sudo /tmp/_remote_script.sh" \
        || err "Remote script failed on ${ip}: ${description}"
}

# ─────────────────────────── Teardown ────────────────────────────────────────
teardown() {
    hdr "Tearing down cluster"
    for (( i=0; i<TOTAL_VMS; i++ )); do
        vm="${ALL_NAMES[$i]}"
        virsh destroy  "$vm" 2>/dev/null || true
        virsh undefine "$vm" --remove-all-storage 2>/dev/null || true
        info "  Removed ${vm}"
    done
    virsh net-destroy  "$NET_NAME" 2>/dev/null || true
    virsh net-undefine "$NET_NAME" 2>/dev/null || true
    rm -f "${SSH_KEY_PATH}" "${SSH_KEY_PATH}.pub"
    rm -f /tmp/k8s-*.sh
    info "Teardown complete."
    exit 0
}
[[ "$ACTION" == "teardown" ]] && teardown

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Validate host resources
# ═════════════════════════════════════════════════════════════════════════════
hdr "Validating host resources"

if ! grep -qE '(vmx|svm)' /proc/cpuinfo; then
    err "Hardware virtualisation (VT-x/AMD-V) not available."
fi

HOST_CPUS=$(nproc)
HOST_RAM_MB=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)
# Available disk in the image directory (or its parent mount)
HOST_DISK_AVAIL_GB=$(df -BG "${IMAGE_DIR}" 2>/dev/null \
    | awk 'NR==2{gsub("G",""); print $4}' || echo 0)

TOTAL_VCPUS=0; TOTAL_RAM_MB=0; TOTAL_DISK_GB=0
for (( i=0; i<TOTAL_VMS; i++ )); do
    TOTAL_VCPUS=$((TOTAL_VCPUS + ALL_VCPUS[i]))
    TOTAL_RAM_MB=$((TOTAL_RAM_MB + ALL_RAM[i]))
    TOTAL_DISK_GB=$((TOTAL_DISK_GB + ALL_DISK[i]))
done

info "Host:      ${HOST_CPUS} CPUs,  ${HOST_RAM_MB} MB RAM,  ${HOST_DISK_AVAIL_GB} GB disk available (${IMAGE_DIR})"
info "Requested: ${TOTAL_VCPUS} vCPUs, ${TOTAL_RAM_MB} MB RAM, ${TOTAL_DISK_GB} GB disk"

ERRORS=0

# RAM is a hard constraint
if [[ $TOTAL_RAM_MB -gt $HOST_RAM_MB ]]; then
    echo -e "${RED}  ✗ RAM: need ${TOTAL_RAM_MB} MB but host only has ${HOST_RAM_MB} MB${NC}"
    ERRORS=$((ERRORS + 1))
else
    HEADROOM=$(( (HOST_RAM_MB - TOTAL_RAM_MB) * 100 / HOST_RAM_MB ))
    if [[ $HEADROOM -lt 5 ]]; then
        warn "  RAM: only ${HEADROOM}% headroom — host may become unresponsive"
    else
        echo -e "${GREEN}  ✓ RAM: ${TOTAL_RAM_MB} / ${HOST_RAM_MB} MB (${HEADROOM}% headroom)${NC}"
    fi
fi

# Disk is a hard constraint
if [[ $TOTAL_DISK_GB -gt $HOST_DISK_AVAIL_GB ]]; then
    echo -e "${RED}  ✗ Disk: need ${TOTAL_DISK_GB} GB but only ${HOST_DISK_AVAIL_GB} GB available${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}  ✓ Disk: ${TOTAL_DISK_GB} / ${HOST_DISK_AVAIL_GB} GB available${NC}"
fi

# CPU is soft (oversubscription is normal for VMs)
if [[ $TOTAL_VCPUS -gt $HOST_CPUS ]]; then
    RATIO=$(awk "BEGIN{printf \"%.1f\", $TOTAL_VCPUS/$HOST_CPUS}")
    warn "  CPU: ${TOTAL_VCPUS} vCPUs on ${HOST_CPUS} physical cores (${RATIO}x oversubscription)"
    if (( TOTAL_VCPUS > HOST_CPUS * 4 )); then
        echo -e "${RED}  ✗ CPU oversubscription >4x — this will be very slow${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${GREEN}  ✓ CPU: ${TOTAL_VCPUS} / ${HOST_CPUS} cores${NC}"
fi

if [[ $ERRORS -gt 0 ]]; then
    err "Host resource validation failed (${ERRORS} error(s)). Adjust your config or provision a bigger machine."
fi
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Infrastructure setup
# ═════════════════════════════════════════════════════════════════════════════
hdr "Setting up infrastructure"

info "Installing host dependencies..."
apt-get update -qq
apt-get install -y -qq \
    qemu-kvm libvirt-daemon-system libvirt-clients \
    virtinst bridge-utils cloud-image-utils \
    genisoimage wget openssh-client \
    dnsmasq-base >/dev/null 2>&1
systemctl enable --now libvirtd

# ────── SSH Key ──────────────────────────────────────────────────────────────
if [[ ! -f "$SSH_KEY_PATH" ]]; then
    info "Generating SSH key pair..."
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -q
else
    info "SSH key already exists, reusing."
fi
SSH_PUB=$(cat "${SSH_KEY_PATH}.pub")

# ────── Libvirt Network ─────────────────────────────────────────────────────
setup_network() {
    if virsh net-info "$NET_NAME" &>/dev/null; then
        local state
        state=$(virsh net-info "$NET_NAME" 2>/dev/null | awk '/^Active:/{print $2}')
        if [[ "$state" == "yes" ]]; then
            info "Network ${NET_NAME} already active."
            return 0
        fi
        info "Network ${NET_NAME} inactive, starting..."
        if virsh net-start "$NET_NAME" 2>/dev/null; then return 0; fi
        warn "Could not start, recreating..."
        virsh net-destroy  "$NET_NAME" 2>/dev/null || true
        virsh net-undefine "$NET_NAME" 2>/dev/null || true
    fi

    info "Creating NAT network ${NET_NAME} (${SUBNET}.0/24)..."

    # Build DHCP host reservations
    DHCP_HOSTS=""
    for (( i=0; i<TOTAL_VMS; i++ )); do
        DHCP_HOSTS+="      <host mac='${ALL_MACS[$i]}' ip='${ALL_IPS[$i]}'/>"$'\n'
    done

    cat > /tmp/${NET_NAME}.xml <<NETEOF
<network>
  <name>${NET_NAME}</name>
  <forward mode='nat'/>
  <bridge name='${NET_BRIDGE}' stp='on' delay='0'/>
  <ip address='${GATEWAY}' netmask='255.255.255.0'>
    <dhcp>
      <range start='${SUBNET}.100' end='${SUBNET}.200'/>
${DHCP_HOSTS}    </dhcp>
  </ip>
</network>
NETEOF
    virsh net-define /tmp/${NET_NAME}.xml || err "Failed to define network"
    virsh net-start "$NET_NAME"           || err "Failed to start network"
    virsh net-autostart "$NET_NAME"
    rm -f /tmp/${NET_NAME}.xml
}
setup_network

# ────── Download base image ─────────────────────────────────────────────────
BASE_IMG="${IMAGE_DIR}/jammy-base.qcow2"
if [[ ! -f "$BASE_IMG" ]]; then
    info "Downloading Ubuntu 22.04 cloud image..."
    wget -q --show-progress -O "${BASE_IMG}.tmp" "$IMAGE_URL"
    mv "${BASE_IMG}.tmp" "$BASE_IMG"
else
    info "Base image already exists, reusing."
fi

# ────── Build /etc/hosts content for all VMs ────────────────────────────────
ETC_HOSTS="127.0.0.1 localhost"
for (( i=0; i<TOTAL_VMS; i++ )); do
    ETC_HOSTS+=$'\n'"      ${ALL_IPS[$i]} ${ALL_NAMES[$i]}"
done

# ────── K8s setup script (runs inside each VM via cloud-init) ────────────────
K8S_SETUP_SCRIPT=$(cat <<K8SEOF
#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

[[ -f /tmp/k8s_setup_done ]] && exit 0

cat > /etc/modules-load.d/k8s.conf <<EOF2
overlay
br_netfilter
EOF2
modprobe overlay
modprobe br_netfilter

cat > /etc/sysctl.d/k8s.conf <<EOF2
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF2
sysctl --system >/dev/null 2>&1

swapoff -a
sed -i '/swap/d' /etc/fstab

apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg apt-transport-https >/dev/null 2>&1

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq containerd.io >/dev/null 2>&1

mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
systemctl restart containerd
systemctl enable containerd

curl -fsSL "https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/Release.key" | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/ /" > /etc/apt/sources.list.d/kubernetes.list

apt-get update -qq
apt-get install -y -qq kubelet kubeadm kubectl >/dev/null 2>&1
apt-mark hold kubelet kubeadm kubectl
systemctl enable kubelet

echo "K8S_SETUP_DONE" > /tmp/k8s_setup_done
K8SEOF
)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1b — Create VMs
# ═════════════════════════════════════════════════════════════════════════════
hdr "Creating ${TOTAL_VMS} virtual machines"

for (( i=0; i<TOTAL_VMS; i++ )); do
    VM="${ALL_NAMES[$i]}"
    IP="${ALL_IPS[$i]}"
    MAC="${ALL_MACS[$i]}"
    VCPUS="${ALL_VCPUS[$i]}"
    RAM="${ALL_RAM[$i]}"
    DISK="${ALL_DISK[$i]}"

    if virsh dominfo "$VM" &>/dev/null; then
        vm_state=$(virsh domstate "$VM" 2>/dev/null || echo "unknown")
        if [[ "$vm_state" == "running" ]]; then
            info "VM $VM already running, skipping."
            continue
        elif [[ "$vm_state" == "shut off" ]]; then
            info "VM $VM stopped, starting..."
            virsh start "$VM"
            continue
        else
            warn "VM $VM in state '$vm_state', recreating..."
            virsh destroy  "$VM" 2>/dev/null || true
            virsh undefine "$VM" --remove-all-storage 2>/dev/null || true
        fi
    fi

    info "Creating VM: ${VM}  (${VCPUS} vCPU, ${RAM} MB, ${DISK} GB, IP: ${IP})"

    VM_DISK="${IMAGE_DIR}/${VM}.qcow2"
    rm -f "$VM_DISK"
    cp "$BASE_IMG" "$VM_DISK"
    qemu-img resize "$VM_DISK" "${DISK}G" >/dev/null

    CIDIR=$(mktemp -d)

    cat > "${CIDIR}/meta-data" <<EOF
instance-id: ${VM}
local-hostname: ${VM}
EOF

    cat > "${CIDIR}/user-data" <<EOF
#cloud-config
hostname: ${VM}
manage_etc_hosts: false
users:
  - name: ${VM_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    plain_text_passwd: "${VM_PASSWORD}"
    ssh_authorized_keys:
      - ${SSH_PUB}
package_update: true
package_upgrade: false
write_files:
  - path: /etc/hosts
    content: |
      ${ETC_HOSTS}
    permissions: '0644'
  - path: /root/k8s-setup.sh
    permissions: '0755'
    content: |
$(echo "$K8S_SETUP_SCRIPT" | sed 's/^/      /')
runcmd:
  - bash /root/k8s-setup.sh
EOF

    cat > "${CIDIR}/network-config" <<EOF
version: 2
ethernets:
  enp1s0:
    dhcp4: false
    addresses:
      - ${IP}/24
    gateway4: ${GATEWAY}
    nameservers:
      addresses:
        - 8.8.8.8
        - 8.8.4.4
EOF

    SEED_ISO="${IMAGE_DIR}/${VM}-seed.iso"
    rm -f "$SEED_ISO"
    cloud-localds -v --network-config="${CIDIR}/network-config" \
        "$SEED_ISO" "${CIDIR}/user-data" "${CIDIR}/meta-data" 2>/dev/null || \
    genisoimage -output "$SEED_ISO" -volid cidata -joliet -rock \
        "${CIDIR}/user-data" "${CIDIR}/meta-data" "${CIDIR}/network-config" 2>/dev/null

    rm -rf "$CIDIR"

    virt-install \
        --name "$VM" \
        --vcpus "$VCPUS" \
        --memory "$RAM" \
        --disk "path=${VM_DISK},format=qcow2,bus=virtio" \
        --disk "path=${SEED_ISO},device=cdrom" \
        --os-variant "$OS_VARIANT" \
        --network "network=${NET_NAME},mac=${MAC},model=virtio" \
        --graphics none \
        --console pty,target_type=serial \
        --noautoconsole \
        --import \
        --quiet

    info "  VM ${VM} launched."
done

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Wait for VMs + bootstrap Kubernetes
# ═════════════════════════════════════════════════════════════════════════════
hdr "Waiting for all ${TOTAL_VMS} VMs to boot and install packages"
info "(This typically takes 3-6 minutes depending on VM count. Be patient.)"
echo ""
PHASE2_START=$(date +%s)

for (( i=0; i<TOTAL_VMS; i++ )); do
    ip="${ALL_IPS[$i]}"
    name="${ALL_NAMES[$i]}"

    # Wait for SSH
    retries=0
    while [[ $retries -lt 60 ]]; do
        elapsed=$(( $(date +%s) - PHASE2_START ))
        printf "\r  [%3ds] %-20s: waiting for SSH...          " "$elapsed" "$name"
        if ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ip}" "true" 2>/dev/null; then
            elapsed=$(( $(date +%s) - PHASE2_START ))
            printf "\r  [%3ds] %-20s: SSH up ✓                    \n" "$elapsed" "$name"
            break
        fi
        sleep 5
        retries=$((retries + 1))
    done
    [[ $retries -ge 60 ]] && err "SSH timed out on ${name} after 5 min"

    # Wait for cloud-init k8s setup
    retries=0
    while [[ $retries -lt 60 ]]; do
        elapsed=$(( $(date +%s) - PHASE2_START ))
        ci_status=$(ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ip}" \
            "cloud-init status 2>/dev/null | head -1 || echo 'unknown'" 2>/dev/null || echo "unknown")
        printf "\r  [%3ds] %-20s: installing packages (%s)...          " "$elapsed" "$name" "$ci_status"
        if ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ip}" "test -f /tmp/k8s_setup_done" 2>/dev/null; then
            elapsed=$(( $(date +%s) - PHASE2_START ))
            printf "\r  [%3ds] %-20s: kubeadm + containerd ready ✓          \n" "$elapsed" "$name"
            break
        fi
        sleep 10
        retries=$((retries + 1))
    done
    [[ $retries -ge 60 ]] && err "Timed out on ${name}. SSH in and check /var/log/cloud-init-output.log"
done
echo ""
PHASE2_ELAPSED=$(( $(date +%s) - PHASE2_START ))
info "All ${TOTAL_VMS} VMs ready in ${PHASE2_ELAPSED}s ✓"

# ────── Verify inter-VM connectivity ─────────────────────────────────────────
info "Verifying inter-VM connectivity (spot check)..."
# Only ping each CP from a sample of workers (full mesh would be slow with many VMs)
for ci in "${CP_INDICES[@]}"; do
    for wi in "${WORKER_INDICES[@]}"; do
        ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ALL_IPS[$wi]}" \
            "ping -c 1 -W 2 ${ALL_IPS[$ci]}" >/dev/null 2>&1 \
            || warn "${ALL_NAMES[$wi]} → ${ALL_NAMES[$ci]}: FAIL"
    done
    # Only spot-check from first 3 workers
    break_count=0
    for wi in "${WORKER_INDICES[@]}"; do
        break_count=$((break_count + 1))
        [[ $break_count -ge 3 ]] && break
    done
done
info "Connectivity check done ✓"

# ────── Skip if cluster already healthy ──────────────────────────────────────
CLUSTER_READY=false
if ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" \
    "test -f /home/${VM_USER}/.kube/config" 2>/dev/null; then
    READY_COUNT=$(ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" \
        "kubectl get nodes --no-headers 2>/dev/null | grep -c ' Ready '" 2>/dev/null || echo 0)
    if [[ "$READY_COUNT" -ge "$TOTAL_VMS" ]]; then
        info "Cluster already has ${READY_COUNT} Ready nodes — skipping bootstrap."
        CLUSTER_READY=true
    fi
fi

if [[ "$CLUSTER_READY" == "false" ]]; then

    # ────── Reset prior state on all VMs ─────────────────────────────────────
    hdr "Bootstrapping Kubernetes cluster"
    info "Resetting any prior kubeadm state..."
    for (( i=0; i<TOTAL_VMS; i++ )); do
        ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${ALL_IPS[$i]}" \
            "sudo kubeadm reset -f 2>/dev/null; sudo rm -rf /etc/kubernetes /var/lib/kubelet/config.yaml /etc/cni/net.d ~/.kube" \
            2>/dev/null || true
    done

    # ────── Determine if HA (multiple control planes) ────────────────────────
    HA_MODE=false
    if [[ ${#CP_INDICES[@]} -gt 1 ]]; then
        HA_MODE=true
        info "HA mode: ${#CP_INDICES[@]} control-plane nodes detected"
    fi

    # ────── kubeadm init on primary CP ───────────────────────────────────────
    info "Initialising primary control plane on ${CP_NAME} (${CP_IP})..."

    HA_FLAGS=""
    if [[ "$HA_MODE" == "true" ]]; then
        HA_FLAGS="--control-plane-endpoint=${CP_IP}:6443 --upload-certs"
    fi

    cat > /tmp/k8s-init-cp.sh <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

echo "[CP] Running kubeadm init..."
sudo kubeadm init \\
    --apiserver-advertise-address=${CP_IP} \\
    --pod-network-cidr=${POD_CIDR} \\
    --node-name=${CP_NAME} \\
    ${HA_FLAGS}

echo "[CP] Setting up kubeconfig..."
mkdir -p /home/${VM_USER}/.kube
sudo cp /etc/kubernetes/admin.conf /home/${VM_USER}/.kube/config
sudo chown ${VM_USER}:${VM_USER} /home/${VM_USER}/.kube/config

echo "[CP] Verifying kubectl works..."
export KUBECONFIG=/home/${VM_USER}/.kube/config
kubectl cluster-info || { echo "[CP] kubectl cluster-info FAILED"; exit 1; }

echo "[CP] Installing Flannel CNI..."
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

echo "[CP] Generating join command..."
sudo kubeadm token create --print-join-command > /tmp/k8s_join_cmd.txt

SCRIPT

    # If HA, also capture the certificate key
    if [[ "$HA_MODE" == "true" ]]; then
        cat >> /tmp/k8s-init-cp.sh <<'SCRIPT'
echo "[CP] Generating certificate key for HA..."
CERT_KEY=$(sudo kubeadm init phase upload-certs --upload-certs 2>/dev/null | tail -1)
echo "$CERT_KEY" > /tmp/k8s_cert_key.txt
SCRIPT
    fi

    echo 'echo "[CP] Control plane init COMPLETE"' >> /tmp/k8s-init-cp.sh

    run_remote_script "$CP_IP" /tmp/k8s-init-cp.sh "kubeadm init + kubeconfig + Flannel"

    # Verify
    info "Verifying control plane..."
    ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "test -f /etc/kubernetes/admin.conf" \
        || err "/etc/kubernetes/admin.conf missing — kubeadm init failed"
    ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "test -f /tmp/k8s_join_cmd.txt" \
        || err "Join command file missing"
    info "Control plane verified ✓"

    # Retrieve join command
    JOIN_CMD=$(ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "cat /tmp/k8s_join_cmd.txt" 2>/dev/null)
    [[ -z "$JOIN_CMD" || "$JOIN_CMD" != *"kubeadm join"* ]] && \
        err "Invalid join command: ${JOIN_CMD}"

    CERT_KEY=""
    if [[ "$HA_MODE" == "true" ]]; then
        CERT_KEY=$(ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "cat /tmp/k8s_cert_key.txt" 2>/dev/null)
        [[ -z "$CERT_KEY" ]] && err "Failed to retrieve certificate key for HA"
        info "Certificate key retrieved for HA join ✓"
    fi

    # ────── Join additional control-plane nodes ──────────────────────────────
    if [[ "$HA_MODE" == "true" ]]; then
        for (( ci=1; ci<${#CP_INDICES[@]}; ci++ )); do
            idx="${CP_INDICES[$ci]}"
            W_IP="${ALL_IPS[$idx]}"
            W_NAME="${ALL_NAMES[$idx]}"

            cat > /tmp/k8s-join-cp.sh <<CPSCRIPT
#!/usr/bin/env bash
set -euo pipefail
echo "[CP-HA] Joining cluster as control-plane node ${W_NAME}..."
sudo ${JOIN_CMD} --node-name=${W_NAME} --control-plane --certificate-key ${CERT_KEY}

echo "[CP-HA] Setting up kubeconfig..."
mkdir -p /home/${VM_USER}/.kube
sudo cp /etc/kubernetes/admin.conf /home/${VM_USER}/.kube/config
sudo chown ${VM_USER}:${VM_USER} /home/${VM_USER}/.kube/config

echo "[CP-HA] Join COMPLETE"
CPSCRIPT

            run_remote_script "$W_IP" /tmp/k8s-join-cp.sh "Join ${W_NAME} as HA control-plane"
        done
    fi

    # ────── Join workers ─────────────────────────────────────────────────────
    info "Joining ${#WORKER_INDICES[@]} worker nodes..."
    for idx in "${WORKER_INDICES[@]}"; do
        W_IP="${ALL_IPS[$idx]}"
        W_NAME="${ALL_NAMES[$idx]}"

        cat > /tmp/k8s-join-worker.sh <<WSCRIPT
#!/usr/bin/env bash
set -euo pipefail
echo "[WORKER] Joining cluster as ${W_NAME}..."
sudo ${JOIN_CMD} --node-name=${W_NAME}
echo "[WORKER] Join COMPLETE"
WSCRIPT

        run_remote_script "$W_IP" /tmp/k8s-join-worker.sh "Join worker ${W_NAME}"
    done
fi

# ────── Wait for all nodes Ready ─────────────────────────────────────────────
hdr "Waiting for ${TOTAL_VMS} nodes to reach Ready"
RETRIES=0
READY_COUNT=0
while [[ $RETRIES -lt 60 ]]; do
    READY_COUNT=$(ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" \
        "kubectl get nodes --no-headers 2>/dev/null | grep -c ' Ready '" 2>/dev/null || echo 0)
    if [[ "$READY_COUNT" -ge "$TOTAL_VMS" ]]; then
        break
    fi
    printf "\r  Nodes ready: %s/%s ...    " "$READY_COUNT" "$TOTAL_VMS"
    sleep 10
    RETRIES=$((RETRIES + 1))
done
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================================"
if [[ "$READY_COUNT" -ge "$TOTAL_VMS" ]]; then
    info "🎉  Kubernetes cluster is UP!  (${READY_COUNT}/${TOTAL_VMS} nodes Ready)"
else
    warn "⚠️  Cluster partially up (${READY_COUNT}/${TOTAL_VMS} Ready). Nodes may still be converging."
fi
echo "========================================================================"
echo ""
ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "kubectl get nodes -o wide" 2>/dev/null || true
echo ""
ssh "${SSH_OPTS_ARR[@]}" "${VM_USER}@${CP_IP}" "kubectl get pods -A" 2>/dev/null || true
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Control Plane(s):"
for ci in "${CP_INDICES[@]}"; do
    echo "   ${ALL_NAMES[$ci]}  @  ${ALL_IPS[$ci]}"
done
echo ""
echo " Workers:"
for wi in "${WORKER_INDICES[@]}"; do
    echo "   ${ALL_NAMES[$wi]}  @  ${ALL_IPS[$wi]}"
done
echo ""
echo " SSH:     ssh -i ${SSH_KEY_PATH} ${VM_USER}@<IP>"
echo " Destroy: sudo bash $0 teardown ${CONFIG_FILE}"
echo "────────────────────────────────────────────────────────────────────────"

rm -f /tmp/k8s-init-cp.sh /tmp/k8s-join-worker.sh /tmp/k8s-join-cp.sh
