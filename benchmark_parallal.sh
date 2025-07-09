#!/bin/bash

# Configuration parameters
NUM_PROCESSES=12
QUERY_COUNT=10000

redis_hosts=( "localhost"  "redis-13000.internal.cluster.kamran-default.demo.redislabs.com" "redis-14000.internal.cluster.kamran-default.demo.redislabs.com" "redis-15000.internal.cluster.kamran-default.demo.redislabs.com" "redis-16000.internal.cluster.kamran-default.demo.redislabs.com")
redis_ports=(6380  13000 14000 15000 16000)

for i in "${!redis_hosts[@]}"; do
    echo "Starting ${NUM_PROCESSES} benchmark processes for ${redis_hosts[i]}:${redis_ports[i]} with ${QUERY_COUNT} queries each"
    for j in $(seq 1 ${NUM_PROCESSES}); do
        python benchmarkvl.py query --redis-host "${redis_hosts[i]}" --redis-port "${redis_ports[i]}" --max-workers 1 --algorithm hnsw --query-count ${QUERY_COUNT} &
    done
    wait
    echo "Completed benchmarks for ${redis_hosts[i]}:${redis_ports[i]}"
done