#!/bin/bash
# net_scenarios_host.sh
# Usage:
#   sudo ./net_scenarios_host.sh <case> [mtu]
#   case = 1 | 2 | 3 | reset
#   mtu  = optional, e.g. 1500 or 9001 (default keeps current)

HOST_IF="ens6"  # Change to ens5 or ens6 depending on host
MTU_ARG=$2

# Reset any existing qdisc rules
reset_qdisc() {
    echo ">>> Resetting qdisc on $HOST_IF"
    sudo tc qdisc del dev $HOST_IF root 2>/dev/null
}

# Set MTU if specified
set_mtu() {
    local IF=$1
    local MTU=$2
    if [[ -n "$MTU" ]]; then
        echo ">>> Setting MTU=$MTU on $IF"
        sudo ip link set dev $IF mtu $MTU
    fi
}

# Calculate burst dynamically
get_burst() {
    local IF=$1
    local MTU=$(ip link show "$IF" | awk '/mtu/ {print $5; exit}')
    echo $(( MTU + MTU / 2 ))
}

case "$1" in
    1)
        echo ">>> Host Case 1: RTT=10ms, Loss=1%, Rate=100Mbit"
        reset_qdisc
        set_mtu $HOST_IF $MTU_ARG
        sudo tc qdisc add dev $HOST_IF root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $HOST_IF)
        sudo tc qdisc add dev $HOST_IF parent 1:1 handle 10: netem delay 5ms loss 1%
        ;;
    2)
        echo ">>> Host Case 2: RTT=200ms, Loss=20%, Rate=100Mbit"
        reset_qdisc
        set_mtu $HOST_IF $MTU_ARG
        sudo tc qdisc add dev $HOST_IF root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $HOST_IF)
        sudo tc qdisc add dev $HOST_IF parent 1:1 handle 10: netem delay 100ms loss 20%
        ;;
    3)
        echo ">>> Host Case 3: RTT=200ms, Loss=0%, Rate=100Mbit (Host)"
        reset_qdisc
        set_mtu $HOST_IF $MTU_ARG
        sudo tc qdisc add dev $HOST_IF root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $HOST_IF)
        sudo tc qdisc add dev $HOST_IF parent 1:1 handle 10: netem delay 100ms
        ;;
    reset)
        reset_qdisc
        echo ">>> All qdisc rules cleared on $HOST_IF."
        ;;
    *)
        echo "Usage: $0 {1|2|3|reset} [mtu]"
        exit 1
        ;;
esac
