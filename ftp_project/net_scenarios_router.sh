#!/bin/bash
# net_scenarios_router.sh
# Usage:
#   sudo ./net_scenarios_router.sh <case> [mtu]
#   case = 1 | 2 | 3 | reset
#   mtu  = optional, e.g. 1500 or 9001 (default keeps current)

#LAN-SERVER
ROUTER_IF1="eth1"
#LAN-CLIENT
ROUTER_IF2="eth2"
MTU_ARG=$2

reset_qdisc() {
    echo ">>> Resetting qdisc on $ROUTER_IF1 and $ROUTER_IF2"
    sudo tc qdisc del dev $ROUTER_IF1 root 2>/dev/null
    sudo tc qdisc del dev $ROUTER_IF2 root 2>/dev/null
}

set_mtu() {
    local IF=$1
    local MTU=$2
    if [[ -n "$MTU" ]]; then
        echo ">>> Setting MTU=$MTU on $IF"
        sudo ip link set dev $IF mtu $MTU
    fi
}

get_burst() {
    local IF=$1
    local MTU=$(ip link show "$IF" | awk '/mtu/ {print $5; exit}')
    echo $(( MTU + MTU / 2 ))
}

case "$1" in
    1)
        echo ">>> Router Case 1: RTT=10ms, Loss=1%, Rate=100Mbit"
        reset_qdisc
        set_mtu $ROUTER_IF1 $MTU_ARG
        set_mtu $ROUTER_IF2 $MTU_ARG
        sudo tc qdisc add dev $ROUTER_IF1 root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $ROUTER_IF1)
        sudo tc qdisc add dev $ROUTER_IF2 root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $ROUTER_IF2)
        sudo tc qdisc add dev $ROUTER_IF1 parent 1:1 handle 10: netem delay 5ms loss 1%
        sudo tc qdisc add dev $ROUTER_IF2 parent 1:1 handle 10: netem delay 5ms loss 1%
        ;;
    2)
        echo ">>> Router Case 2: RTT=200ms, Loss=20%, Rate=100Mbit"
        reset_qdisc
        set_mtu $ROUTER_IF1 $MTU_ARG
        set_mtu $ROUTER_IF2 $MTU_ARG
        sudo tc qdisc add dev $ROUTER_IF1 root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $ROUTER_IF1)
        sudo tc qdisc add dev $ROUTER_IF2 root handle 1:0 tbf rate 100mbit latency 0.001ms burst $(get_burst $ROUTER_IF2)
        sudo tc qdisc add dev $ROUTER_IF1 parent 1:1 handle 10: netem delay 100ms loss 20%
        sudo tc qdisc add dev $ROUTER_IF2 parent 1:1 handle 10: netem delay 100ms loss 20%
        ;;
    3)
        echo ">>> Router Case 3: RTT=200ms, Loss=0%, Throughput=80Mbit"
        reset_qdisc
        set_mtu $ROUTER_IF1 $MTU_ARG
        set_mtu $ROUTER_IF2 $MTU_ARG
        sudo tc qdisc add dev $ROUTER_IF1 root handle 1:0 tbf rate 80mbit latency 0.001ms burst $(get_burst $ROUTER_IF1)
        sudo tc qdisc add dev $ROUTER_IF2 root handle 1:0 tbf rate 80mbit latency 0.001ms burst $(get_burst $ROUTER_IF2)
        sudo tc qdisc add dev $ROUTER_IF1 parent 1:1 handle 10: netem delay 100ms
        sudo tc qdisc add dev $ROUTER_IF2 parent 1:1 handle 10: netem delay 100ms
        ;;
    reset)
        reset_qdisc
        echo ">>> All qdisc rules cleared on router interfaces."
        ;;
    *)
        echo "Usage: $0 {1|2|3|reset} [mtu]"
        exit 1
        ;;
esac
