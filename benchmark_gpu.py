#!/usr/bin/env python3
"""
Quick benchmark to test GPU speedup vs CPU.
"""

if __name__ == "__main__":
    from src.dissonance_gpu import benchmark_gpu_speedup
    
    print("\n🚀 Running GPU Speedup Benchmark...\n")
    print("This will compare CPU (NumPy) vs GPU (PyTorch) performance")
    print("for dissonance calculations at different problem sizes.\n")
    
    benchmark_gpu_speedup()
    
    print("\n✅ Benchmark complete!")
    print("\nTo enable GPU in your optimization:")
    print('  1. Set "gpu.enabled": true in config.json')
    print('  2. Set "gpu.device": "cuda" for V100, or "auto" for auto-detect')
    print('  3. Run: python run_experiment.py')
