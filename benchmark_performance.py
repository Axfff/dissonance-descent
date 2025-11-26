"""
Quick performance benchmark to demonstrate speedup from optimizations.
"""
import time
import numpy as np
from src.dissonance import calculate_total_dissonance
from src.dissonance_fast import calculate_total_dissonance_fast

def benchmark_dissonance():
    """Compare performance of different backends."""
    
    # Create test data: simulate large optimization space
    n_partials = 100  # Typical for dense grid
    frequencies = np.random.uniform(100, 1000, n_partials)
    amplitudes = np.random.uniform(0.1, 1.0, n_partials)
    
    model_params = {'a': 0.5, 'b': 14, 's1': 0.021, 's2': 19.0}
    
    print("=" * 60)
    print("DISSONANCE CALCULATION PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Test configuration: {n_partials} partials")
    print(f"Pairwise calculations: {n_partials * (n_partials - 1) // 2:,}")
    print()
    
    # Benchmark original implementation
    print("1. Original Python implementation...")
    start = time.time()
    for _ in range(10):
        result_orig = calculate_total_dissonance(frequencies, amplitudes, model_params)
    time_orig = (time.time() - start) / 10
    print(f"   Average time: {time_orig*1000:.2f} ms")
    print(f"   Result: {result_orig:.6f}")
    print()
    
    # Benchmark NumPy vectorized
    print("2. NumPy vectorized implementation...")
    start = time.time()
    for _ in range(10):
        result_numpy = calculate_total_dissonance_fast(frequencies, amplitudes, model_params, backend='numpy')
    time_numpy = (time.time() - start) / 10
    speedup_numpy = time_orig / time_numpy
    print(f"   Average time: {time_numpy*1000:.2f} ms")
    print(f"   Speedup: {speedup_numpy:.1f}x")
    print(f"   Result: {result_numpy:.6f}")
    print()
    
    # Benchmark Numba JIT
    print("3. Numba JIT compiled implementation...")
    # Warmup
    _ = calculate_total_dissonance_fast(frequencies, amplitudes, model_params, backend='numba')
    start = time.time()
    for _ in range(10):
        result_numba = calculate_total_dissonance_fast(frequencies, amplitudes, model_params, backend='numba')
    time_numba = (time.time() - start) / 10
    speedup_numba = time_orig / time_numba
    print(f"   Average time: {time_numba*1000:.2f} ms (after warmup)")
    print(f"   Speedup: {speedup_numba:.1f}x")
    print(f"   Result: {result_numba:.6f}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"NumPy vectorized:  {speedup_numpy:6.1f}x faster")
    print(f"Numba JIT:         {speedup_numba:6.1f}x faster")
    print()
    print(f"For a 480-param optimization with ~700 evaluations:")
    estimated_orig = time_orig * 700 * 2021 / 60  # 2021 slices
    estimated_fast = time_numba * 700 * 2021 / 60
    print(f"  Original:  ~{estimated_orig:.1f} minutes")
    print(f"  Optimized: ~{estimated_fast:.1f} minutes")
    print(f"  Time saved: ~{estimated_orig - estimated_fast:.1f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_dissonance()
