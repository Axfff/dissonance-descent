"""
Multi-restart optimization wrapper for global search.
Runs optimization multiple times with different random initializations.
"""
import numpy as np
import json
from datetime import datetime
from src.randomization import create_random_initial_partials

def optimize_timbre_multi_restart(slices, config, model_params=None, n_restarts=3,
                                  strategies=None, verbose=True):
    """
    Run optimization multiple times with different random initializations.
    Keep the best result.
    
    Args:
        slices: MIDI slices
        config: Full config dict
        model_params: Plomp-Levelt parameters
        n_restarts: Number of optimization runs
        strategies: List of strategies to use ('perturb', 'smart', 'random')
        verbose: Print progress
        
    Returns:
        dict: Best optimization result with additional multi-restart info
    """
    from src.optimizer_enhanced import optimize_timbre_enhanced
    
    if strategies is None:
        # Default: mix of perturbation and smart randomization
        strategies = ['perturb', 'smart'] * (n_restarts // 2 + 1)
        strategies = strategies[:n_restarts]
    
    # Ensure we have enough strategies
    while len(strategies) < n_restarts:
        strategies.append('smart')
    
    results = []
    best_result = None
    best_cost = float('inf')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-RESTART OPTIMIZATION: {n_restarts} runs")
        print(f"{'='*60}")
    
    # Create a modified config for each restart
    for restart_idx in range(n_restarts):
        strategy = strategies[restart_idx]
        
        if verbose:
            print(f"\n--- Restart {restart_idx + 1}/{n_restarts} (strategy: {strategy}) ---")
        
        # Create randomized initial partials
        opt_config = config['optimization']
        timbre_config = config['timbre']
        
        # Set random seed for reproducibility
        np.random.seed(restart_idx + 42)
        
        # Modify config to use randomized initialization
        config_copy = {**config}
        config_copy['_restart_idx'] = restart_idx
        config_copy['_restart_strategy'] = strategy
        config_copy['_randomization'] = {
            'amp_perturbation': config.get('optimization', {}).get('amp_perturbation', 0.2),
            'adsr_perturbation': config.get('optimization', {}).get('adsr_perturbation', 0.2),
            'strategy': strategy
        }
        
        # Update progress directory for this restart
        base_dir = opt_config.get('progress_dir', 'experiments/optimization_progress')
        config_copy['optimization'] = {**opt_config}
        config_copy['optimization']['progress_dir'] = f"{base_dir}_restart{restart_idx}"
        
        try:
            # Run optimization
            result = optimize_timbre_enhanced(slices, config_copy, model_params)
            
            results.append({
                'restart_idx': restart_idx,
                'strategy': strategy,
                'cost': result['cost'],
                'success': result['success'],
                'result': result
            })
            
            # Track best result
            if result['success'] and result['cost'] < best_cost:
                best_cost = result['cost']
                best_result = result
                best_result['_restart_info'] = {
                    'restart_idx': restart_idx,
                    'strategy': strategy,
                    'n_restarts': n_restarts
                }
            
            if verbose:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} Cost: {result['cost']:.6f}")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            results.append({
                'restart_idx': restart_idx,
                'strategy': strategy,
                'cost': None,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-RESTART SUMMARY")
        print(f"{'='*60}")
        successful = [r for r in results if r['success']]
        print(f"Successful runs: {len(successful)}/{n_restarts}")
        
        if successful:
            costs = [r['cost'] for r in successful]
            print(f"Best cost:  {min(costs):.6f}")
            print(f"Worst cost: {max(costs):.6f}")
            print(f"Mean cost:  {np.mean(costs):.6f}")
            print(f"Std cost:   {np.std(costs):.6f}")
            
            # Find which restart was best
            best_restart = best_result['_restart_info']
            print(f"\nBest result from restart {best_restart['restart_idx'] + 1} (strategy: {best_restart['strategy']})")
        print(f"{'='*60}\n")
    
    # Save multi-restart summary
    if best_result:
        base_dir = config['optimization'].get('progress_dir', 'experiments/optimization_progress')
        summary_file = base_dir + '/multi_restart_summary.json'
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f:
            # Prepare summary (exclude large objects)
            summary = {
                'n_restarts': n_restarts,
                'timestamp': datetime.now().isoformat(),
                'results': [
                    {
                        'restart_idx': int(r['restart_idx']),
                        'strategy': r['strategy'],
                        'cost': float(r['cost']) if r['cost'] is not None else None,
                        'success': bool(r['success'])
                    }
                    for r in results
                ],
                'best': {
                    'restart_idx': best_result['_restart_info']['restart_idx'],
                    'strategy': best_result['_restart_info']['strategy'],
                    'cost': best_result['cost']
                }
            }
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"Multi-restart summary saved to: {summary_file}")
    
    return best_result if best_result else results[0]['result'] if results else None
