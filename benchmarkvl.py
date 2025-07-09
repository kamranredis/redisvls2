import os
import time
import logging
import queue
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Union, Iterator, Any
import typer
from redis import Redis
from redis.connection import ConnectionPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from utils import (
    print_table_header, 
    print_table_footer, 
    calculate_and_print_live_stats, 
    calculate_and_print_final_stats,
    print_benchmark_config
)
#change 
import threading

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_OPERATIONS = ["load", "query"]
SUPPORTED_ALGORITHMS = ["flat", "hnsw"]
SUPPORTED_DISTANCE_METRICS = ["cosine", "l2", "ip"]
SUPPORTED_DATATYPES = ["float32"]
DEFAULT_SEED = 42


@contextmanager
def timer(operation_name: str = "Operation") -> Iterator[None]:
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    logger.info(f"{operation_name} took: {end_time - start_time:.4f} seconds")


def validate_choice_argument(value: str, valid_choices: List[str], arg_name: str) -> None:
    """Validate that a value is in the list of valid choices."""
    if value not in valid_choices:
        logger.error(f"{arg_name} must be one of {valid_choices}, got: {value}")
        raise typer.Exit(1)


def get_numpy_dtype(datatype_str: str) -> np.dtype:
    """Convert datatype string to numpy dtype."""
    dtype_map: Dict[str, np.dtype] = {"float32": np.float32}
    if datatype_str not in dtype_map:
        raise ValueError(f"Unsupported datatype: {datatype_str}. Only {list(dtype_map.keys())} are supported.")
    return dtype_map[datatype_str]


def generate_fake_embeddings(num_embeddings: int, embedding_dim: int, datatype_str: str, seed: Optional[int] = DEFAULT_SEED, start_id: int = 0) -> np.ndarray:
    """Generate fake embeddings using random numbers with deterministic seeding."""
    # Use the start_id as an offset to ensure different workers generate different data
    # but still deterministically based on the base seed
    if seed is not None:
        np.random.seed(seed + start_id)

    dtype = get_numpy_dtype(datatype_str)
    embeddings = np.random.rand(num_embeddings, embedding_dim).astype(dtype)
    return embeddings


def create_schema(index_name: str, dimension: int, distance_metric: str, algorithm: str, datatype: str, include_id: bool = True) -> IndexSchema:
    """Create and return the index schema."""
    fields = [
        {
            "name": "vector",
            "type": "vector",
            "attrs": {
                "dims": dimension,
                "distance_metric": distance_metric,
                "algorithm": algorithm,
                "datatype": datatype,
            },
        }
    ]
    
    # Conditionally add ID field
    if include_id:
        fields.append({
            "name": "id",
            "type": "text",
        })
    
    return IndexSchema.from_dict({
        "index": {"name": index_name, "prefix": index_name, "storage_type": "hash"},
        "fields": fields,
    })


def _load_worker(connection_pool: ConnectionPool, index_name: str, start_id: int, count: int, dimension: int, datatype: str, include_id: bool, worker_id: int, batch_size: int = 1000) -> int:
    """Load worker using Redis pipeline without transactions for efficient batching."""
    
    # Get connection from pool
    redis_client = Redis(connection_pool=connection_pool)

    try:
        total_loaded = 0

        # Process in smaller batches to manage memory
        for batch_start in range(0, count, batch_size):
            batch_count = min(batch_size, count - batch_start)
            current_start_id = start_id + batch_start

            # Generate embeddings just-in-time for this batch
            batch_embeddings = generate_fake_embeddings(
                batch_count, dimension, datatype, start_id=current_start_id
            )

            # Create pipeline without transactions for better performance
            pipe = redis_client.pipeline(transaction=False)

            # Add all batch operations to pipeline
            for i, embedding in enumerate(batch_embeddings):
                doc_id = current_start_id + i

                # Prepare the hash fields
                hash_key = f"{index_name}:{doc_id}"
                fields = {"vector": embedding.tobytes()}

                if include_id:
                    fields["id"] = f"document:{doc_id}"

                # Add to pipeline
                pipe.hset(hash_key, mapping=fields)

            # Execute the batch
            pipe.execute()
            total_loaded += batch_count

            # Optional: Log progress for large batches
            if batch_start + batch_count < count:
                logger.debug(f"Worker {worker_id}: Loaded {total_loaded}/{count} records")

        logger.info(f"Worker {worker_id} completed: {total_loaded} records loaded")
        return total_loaded

    except Exception as e:
        logger.error(f"Worker {worker_id} error loading {count} records starting from ID {start_id}: {e}")
        raise

    finally:
        # Connection automatically returns to pool when redis_client goes out of scope
        pass


def _calculate_worker_ranges(data_size, max_workers):
    """Calculate ID ranges for each worker to ensure even distribution."""
    worker_ranges = []

    if max_workers >= data_size:
        # More workers than data points - each worker gets 1 item max
        for i in range(data_size):
            worker_ranges.append((i, 1))
    else:
        base_range_size = data_size // max_workers

        for i in range(max_workers):
            start_id = i * base_range_size
            if i == max_workers - 1:  # Last worker gets all remaining data
                count = data_size - start_id
            else:
                count = base_range_size
            worker_ranges.append((start_id, count))

    return worker_ranges


def _load_concurrent(client: Redis, schema: IndexSchema, data_size: int, dimension: int, datatype: str, include_id: bool, max_workers: int) -> None:
    """Load data concurrently using Redis pipelining for improved performance and memory usage."""

    # Create connection pool (reuse client's connection config)
    connection_kwargs = client.connection_pool.connection_kwargs.copy()
    pool = ConnectionPool(
        host=connection_kwargs.get('host', 'localhost'),
        port=connection_kwargs.get('port', 6379),
        password=connection_kwargs.get('password', None),
        db=connection_kwargs.get('db', 0),
        max_connections=max_workers + 2  # Extra connections for safety
    )
    
    # Calculate ID ranges for workers
    worker_ranges = _calculate_worker_ranges(data_size, max_workers)
    logger.info(f"Split {data_size} records into {len(worker_ranges)} ranges for {max_workers} workers")
    
    # Track results
    successful_workers = 0
    failed_workers = 0
    total_loaded = 0
    errors: List[str] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all worker loading tasks
        future_to_worker = {
            executor.submit(
                _load_worker,
                pool,
                schema.index.name,  # Get index name from schema
                start_id,
                count,
                dimension,
                datatype,
                include_id,
                i  # worker_id
            ): (start_id, count, i)
            for i, (start_id, count) in enumerate(worker_ranges)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_worker):
            start_id, count, worker_id = future_to_worker[future]
            
            try:
                records_loaded = future.result()
                successful_workers += 1
                total_loaded += records_loaded
                logger.info(f"Worker {worker_id} (IDs {start_id}-{start_id + count - 1}) completed: {records_loaded} records loaded")
                
            except Exception as e:
                failed_workers += 1
                error_msg = f"Worker {worker_id} (IDs {start_id}-{start_id + count - 1}) failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                # Continue with other workers - don't stop execution
    
    # Final reporting
    logger.info(f"Loading completed: {successful_workers}/{len(worker_ranges)} workers successful")
    logger.info(f"Total records loaded: {total_loaded}")
    
    if errors:
        logger.warning(f"Failed workers ({failed_workers}): {errors}")
    
    # Close connection pool
    pool.disconnect()


def _load_single_threaded(client: Redis, schema: IndexSchema, data_size: int, dimension: int, datatype: str, include_id: bool, batch_size: int = 1000) -> None:
    """Load data in single-threaded mode using Redis pipelining and just-in-time generation."""

    total_loaded = 0

    # Process in batches to manage memory
    for batch_start in range(0, data_size, batch_size):
        batch_count = min(batch_size, data_size - batch_start)

        # Generate embeddings just-in-time for this batch
        batch_embeddings = generate_fake_embeddings(
            batch_count, dimension, datatype, start_id=batch_start
        )

        # Create pipeline without transactions
        pipe = client.pipeline(transaction=False)

        # Add all batch operations to pipeline
        for i, embedding in enumerate(batch_embeddings):
            doc_id = batch_start + i

            # Prepare the hash fields
            hash_key = f"{schema.index.name}:{doc_id}"
            fields = {"vector": embedding.tobytes()}

            if include_id:
                fields["id"] = f"document:{doc_id}"

            # Add to pipeline
            pipe.hset(hash_key, mapping=fields)

        # Execute the batch
        pipe.execute()
        total_loaded += batch_count

        # Log progress for batches
        if batch_start + batch_count < data_size:
            logger.info(f"Single-threaded: Loaded {total_loaded}/{data_size} records")

    logger.info(f"Single-threaded loading completed: {total_loaded} records loaded")


def load_data(client: Redis, schema: IndexSchema, data_size: int, dimension: int, datatype: str, include_id: bool = True, max_workers: int = 1) -> None:
    """Load data operation: create index and load embeddings with improved memory efficiency."""
    logger.info("=== LOAD OPERATION ===")
    
    # Create the index
    with timer("Index creation"):
        index = SearchIndex(schema, client, validate_on_load=False)
        index.create(overwrite=True, drop=True)
    
    # Load data using just-in-time generation and Redis pipelining
    with timer("Data Loading with pipelining"):
        logger.info("Starting data loading with just-in-time generation and Redis pipelining.")
        
        if max_workers == 1:
            # Single-threaded path with pipelining
            logger.info("Using single-threaded mode with Redis pipelining")
            _load_single_threaded(client, schema, data_size, dimension, datatype, include_id)
        else:
            # Multi-threaded path with pipelining
            logger.info(f"Using concurrent mode with {max_workers} workers and Redis pipelining")
            _load_concurrent(client, schema, data_size, dimension, datatype, include_id, max_workers)


def execute_query(index: SearchIndex, query_embedding: np.ndarray, num_results: int, query_id: int) -> Dict[str, Any]:
    """Execute a single query and return timing/result info."""
    try:
        start_time = time.perf_counter()
        
        query = VectorQuery(
            vector=query_embedding.tobytes(),
            vector_field_name="vector",
            num_results=num_results,
            return_score=True,
        )
        results = index.query(query)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            'query_id': query_id,
            'latency_ms': latency_ms,
            'success': True,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Query {query_id} error: {e}")
        return {
            'query_id': query_id,
            'latency_ms': None,
            'success': False,
            'error': str(e)
        }


def _create_optimized_worker_pool(schema: IndexSchema, client: Redis, max_workers: int) -> Tuple[List[SearchIndex], ConnectionPool]:
    """Create optimized connection pool and SearchIndex objects with retry logic."""
    
    # Size connection pool generously for concurrent access
    # Each SearchIndex may need multiple connections for different operations
    connection_kwargs = client.connection_pool.connection_kwargs.copy()
    optimized_pool = ConnectionPool(
        host=connection_kwargs.get('host', 'localhost'),
        port=connection_kwargs.get('port', 6379),
        password=connection_kwargs.get('password', None),
        db=connection_kwargs.get('db', 0),
        max_connections=max_workers * 2,  # More generous sizing
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={}
    )
    
    # Create SearchIndex objects with dedicated connections
    search_indexes: List[SearchIndex] = []
    for i in range(max_workers):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Each worker gets its own Redis client from the pool
                worker_redis = Redis(connection_pool=optimized_pool)
                
                # Test connection
                worker_redis.ping()
                
                # Create SearchIndex with dedicated connection
                worker_index = SearchIndex(schema, worker_redis, validate_on_load=False)
                search_indexes.append(worker_index)
                
                logger.debug(f"Created SearchIndex {i} with dedicated connection")
                break
                
            except Exception as e:
                logger.warning(f"SearchIndex {i} creation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Close pool on final failure
                    optimized_pool.disconnect()
                    raise
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    logger.info(f"Created {len(search_indexes)} SearchIndex objects with optimized connection pool")
    return search_indexes, optimized_pool


def _process_query_results(latencies: List[float], successful_queries: int, completed_queries: int, total_queries: int, start_time: float, last_print_time: float) -> float:
    """Helper function to calculate and print live statistics during query execution."""
    current_time = time.perf_counter()
    should_print = (current_time - last_print_time >= 1.0) or (completed_queries == total_queries)
    
    if should_print:
        elapsed_time = current_time - start_time
        success_rate = (successful_queries / completed_queries) * 100 if completed_queries > 0 else 0
        qps = successful_queries / elapsed_time if elapsed_time > 0 else 0
        calculate_and_print_live_stats(latencies, elapsed_time, completed_queries, success_rate, qps)
        return current_time
    
    return last_print_time


def _run_queries_single_threaded(index: SearchIndex, query_embeddings: np.ndarray, num_results: int) -> Tuple[List[float], int, int, float]:
    """Execute queries in single-threaded mode."""
    latencies: List[float] = []
    successful_queries = 0
    start_time = time.perf_counter()
    last_print_time = start_time
    
    for i, embedding in enumerate(query_embeddings):
        result = execute_query(index, embedding, num_results, i)
        
        if result['success']:
            latencies.append(result['latency_ms'])
            successful_queries += 1
        
        # Update live statistics
        last_print_time = _process_query_results(
            latencies, successful_queries, i + 1, len(query_embeddings), start_time, last_print_time
        )
    
    total_time = time.perf_counter() - start_time
    failed_queries = len(query_embeddings) - successful_queries
    
    return latencies, successful_queries, failed_queries, total_time


def _run_queries_concurrent(client: Redis, schema: IndexSchema, query_embeddings: np.ndarray, num_results: int, max_workers: int) -> Tuple[List[float], int, int, float]:
    """Execute queries using queue-based approach with optimized connection management."""
    
    # Create optimized worker pool
    search_indexes, connection_pool = _create_optimized_worker_pool(schema, client, max_workers)
    
    try:
        # Create and populate queue
        query_queue: queue.Queue = queue.Queue()
        for i, embedding in enumerate(query_embeddings):
            query_queue.put((i, embedding))
        
        # Add sentinel values for clean shutdown
        for _ in range(max_workers):
            query_queue.put(None)
        
        logger.info(f"Queued {len(query_embeddings)} queries for processing by {max_workers} workers")
        
        # Thread-safe result collection
        results_dict: Dict[int, Dict[str, Any]] = {}
        results_lock = threading.Lock()
        
        # Track timing for live statistics
        start_time = time.perf_counter()
        last_print_time = start_time
        
        # Launch workers with their own RedisVL index connections
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _query_worker,
                    i,  # worker_id
                    query_queue,
                    results_dict,
                    results_lock,
                    search_indexes[i],  # worker-specific SearchIndex
                    num_results
                ) for i in range(max_workers)
            ]
            
            # Monitor progress while workers process queries
            while True:
                # Check if all workers have finished
                all_done = all(future.done() for future in futures)
                
                # Update live statistics
                current_time = time.perf_counter()
                if current_time - last_print_time >= 1.0 or all_done:
                    with results_lock:
                        completed_queries = len(results_dict)
                        successful_queries = sum(1 for result in results_dict.values() if result['success'])
                        latencies = [result['latency_ms'] for result in results_dict.values() if result['success']]
                    
                    if completed_queries > 0:
                        elapsed_time = current_time - start_time
                        success_rate = (successful_queries / completed_queries) * 100
                        qps = successful_queries / elapsed_time if elapsed_time > 0 else 0
                        calculate_and_print_live_stats(latencies, elapsed_time, completed_queries, success_rate, qps)
                    
                    last_print_time = current_time
                
                if all_done:
                    break
                    
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Wait for all workers to finish
            for future in futures:
                future.result()
        
        # Calculate final statistics
        total_time = time.perf_counter() - start_time
        successful_queries = sum(1 for result in results_dict.values() if result['success'])
        failed_queries = len(query_embeddings) - successful_queries
        latencies = [result['latency_ms'] for result in results_dict.values() if result['success']]
        
        return latencies, successful_queries, failed_queries, total_time
    
    finally:
        # Clean up connection pool
        connection_pool.disconnect()
        logger.debug("Connection pool disconnected")


def _query_worker(worker_id: int, query_queue: queue.Queue, results_dict: Dict[int, Dict[str, Any]], results_lock: threading.Lock, search_index: SearchIndex, num_results: int) -> None:
    """Worker function that pulls queries from queue until sentinel received."""
    processed_count = 0
    
    while True:
        try:
            # Block until item available (no timeout needed)
            query_item = query_queue.get()
            
            # Check for sentinel value (clean shutdown)
            if query_item is None:
                query_queue.task_done()
                logger.debug(f"Worker {worker_id} received shutdown signal after processing {processed_count} queries")
                break
                
            query_id, query_embedding = query_item
            
            # Execute the query
            result = execute_query(search_index, query_embedding, num_results, query_id)
            
            # Store result thread-safely
            with results_lock:
                results_dict[query_id] = result
            
            processed_count += 1
            
            # Mark task as done
            query_queue.task_done()
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error processing query: {e}")
            # Still mark as done to prevent queue.join() from hanging
            query_queue.task_done()
    
    logger.debug(f"Worker {worker_id} finished processing {processed_count} queries")


def query_data(client: Redis, schema: IndexSchema, dimension: int, datatype: str, query_count: int = 100, num_results: int = 10, max_workers: int = 1) -> None:
    """Query operation: perform vector search queries and measure latency."""
    logger.info(f"Running {query_count} queries, returning {num_results} results each")
    
    # Connect to existing index
    index = SearchIndex(schema, client, validate_on_load=False)
    
    # Generate query embeddings ahead of time
    with timer("Generating query embeddings"):
        query_embeddings = generate_fake_embeddings(query_count, dimension, datatype)
        logger.info(f"Generated {query_count} query embeddings.")
    
    # Print header for real-time results
    print_table_header()
    
    # Execute queries based on worker count
    if max_workers == 1:
        #logger.info("Running queries in single-threaded mode")
        latencies, successful_queries, failed_queries, total_time = _run_queries_single_threaded(
            index, query_embeddings, num_results
        )
    else:
        #logger.info(f"Running queries in concurrent mode with {max_workers} workers")
        latencies, successful_queries, failed_queries, total_time = _run_queries_concurrent(
            client, schema, query_embeddings, num_results, max_workers
        )
    
    # Calculate and display final statistics
    calculate_and_print_final_stats(latencies, successful_queries, query_count, total_time)
    logger.info("Query operation completed successfully!")


def main(
    operation: str = typer.Argument(..., help="Operation to perform: 'load' to create index and load data, 'query' to perform vector queries"),
    redis_host: str = typer.Option("localhost", "--redis-host", help="Redis server hostname"),
    redis_port: int = typer.Option(6379, "--redis-port", help="Redis server port"),
    redis_password: str = typer.Option("", "--redis-password", help="Redis server password"),
    index_name: str = typer.Option("redisvl", "--index-name", help="Name of the Redis index"),
    dimension: int = typer.Option(960, "--dimension", help="Vector dimension"),
    algorithm: str = typer.Option("flat", "--algorithm", help="Vector search algorithm (choices: flat, hnsw)"),
    distance_metric: str = typer.Option("cosine", "--distance-metric", help="Distance metric for vector similarity (choices: cosine, l2, ip)"),
    datatype: str = typer.Option("float32", "--datatype", help="Data type for vectors (choices: float32)"),
    data_size: int = typer.Option(1000000, "--data-size", help="Number of embeddings to generate and load (used for 'load' operation)"),
    query_count: int = typer.Option(100, "--query-count", help="Number of queries to run (used for 'query' operation)"),
    num_results: int = typer.Option(3, "--num-results", help="Number of results to return per query (used for 'query' operation)"),
    include_id: bool = typer.Option(True, "--include-id/--no-id", help="Include ID field in schema (default: True)"),
    max_workers: int = typer.Option(1, "--max-workers", help="Number of workers for concurrent operation - applies to both 'load' and 'query' operations (default: 1 for single-threaded)"),
):
    """Redis Vector Search Benchmark Tool"""
    
    # Validate all arguments
    validate_choice_argument(operation, SUPPORTED_OPERATIONS, "operation")
    validate_choice_argument(algorithm, SUPPORTED_ALGORITHMS, "algorithm")
    validate_choice_argument(distance_metric, SUPPORTED_DISTANCE_METRICS, "distance_metric")
    validate_choice_argument(datatype, SUPPORTED_DATATYPES, "datatype")
    
    # Validate query-specific parameters
    if operation == "query":
        if query_count <= 0:
            logger.error("query_count must be greater than 0")
            raise typer.Exit(1)
        if num_results <= 0:
            logger.error("num_results must be greater than 0")
            raise typer.Exit(1)
    
    # Validate load-specific parameters  
    if operation == "load":
        if data_size <= 0:
            logger.error("data_size must be greater than 0")
            raise typer.Exit(1)
    
    # Validate max_workers for both operations
    if max_workers <= 0:
        logger.error("max_workers must be greater than 0")
        raise typer.Exit(1)
    
    # Setup Redis connection
    redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
    
    try:
        client = Redis.from_url(redis_url)
        # Test the connection with a ping
        client.ping()
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}")
        raise typer.Exit(1)
    
    # Print configuration
    print_benchmark_config(
        operation, redis_host, redis_port, index_name, dimension,
        algorithm, distance_metric, datatype, data_size, query_count, num_results, include_id, max_workers
    )
    
    # Create schema
    schema = create_schema(index_name, dimension, distance_metric, algorithm, datatype, include_id)
    
    # Execute operation
    if operation == "load":
        load_data(client, schema, data_size, dimension, datatype, include_id, max_workers)
    elif operation == "query":
        query_data(client, schema, dimension, datatype, query_count, num_results, max_workers)


if __name__ == "__main__":
    typer.run(main)
