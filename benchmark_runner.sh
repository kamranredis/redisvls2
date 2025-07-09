#!/bin/bash

# Simple bash script to iterate over Redis hosts, ports, algorithms, operations, and worker counts
# for Redis Vector Library benchmarking

 

# Define arrays for iteration - paired host:port combinations
redis_hosts=("localhost" "localhost" "redis-12000.internal.cluster.kamran-default.demo.redislabs.com" "redis-13000.internal.cluster.kamran-default.demo.redislabs.com" "redis-14000.internal.cluster.kamran-default.demo.redislabs.com" "redis-15000.internal.cluster.kamran-default.demo.redislabs.com" "redis-16000.internal.cluster.kamran-default.demo.redislabs.com")
redis_ports=(6379 6380 12000 13000 14000 15000 16000)
algorithms=("flat" "hnsw")
operations=("load" "query")
workers=(1 16)

# Calculate total number of operations (paired combinations)
total_ops=$((${#redis_hosts[@]} * ${#algorithms[@]} * ${#operations[@]} * ${#workers[@]}))
current_op=0

echo "=== Starting Benchmark Suite ==="
echo "Total operations to run: $total_ops"
echo ""

# Loop through paired host:port combinations
for i in "${!redis_hosts[@]}"; do
    redis_host="${redis_hosts[$i]}"
    redis_port="${redis_ports[$i]}"
    
    for algorithm in "${algorithms[@]}"; do
        for operation in "${operations[@]}"; do
            for worker_count in "${workers[@]}"; do
                current_op=$((current_op + 1))
                echo "=== Operation $current_op/$total_ops ==="
                echo "Running: host=$redis_host, port=$redis_port, algorithm=$algorithm, operation=$operation, workers=$worker_count"
                echo ""
                
                # Set query count based on algorithm
                if [[ "$algorithm" == "hnsw" ]]; then
                    query_count=10000
                else
                    query_count=1000
                fi
                
                # Build command arguments
                cmd_args=(
                    "$operation"
                    --algorithm "$algorithm"
                    --max-workers "$worker_count"
                    --redis-host "$redis_host"
                    --redis-port "$redis_port"
                    --query-count "$query_count"
                    --data-size 1000000
                )
                
                if python benchmarkvl.py "${cmd_args[@]}"; then
                    echo "✓ Operation $current_op completed successfully"
                else
                    echo "✗ Operation $current_op failed"
                fi
                echo ""
                sleep 1
            done
        done
    done
done

echo "=== Benchmark Complete ==="
