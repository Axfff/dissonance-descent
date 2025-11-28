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
            
            # Add restart metadata to result
            result['_restart_idx'] = restart_idx
            result['_restart_strategy'] = strategy
            
            results.append({
                'restart_idx': restart_idx,
                'strategy': strategy,
                'cost': result['cost'],
                'success': result['success'],
                'result': result
            })
            
            # Generate outputs for this trial
            if result['success']:
                from src.trial_outputs import generate_trial_outputs
                
                trial_output_dir = f"{base_dir}_restart{restart_idx}/outputs"
                try:
                    generated_files = generate_trial_outputs(
                        result, restart_idx, slices, config_copy, trial_output_dir
                    )
                    result['_generated_files'] = generated_files
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Output generation failed: {e}")
            
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
    
    # Save multi-restart summary with file paths
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
                        'success': bool(r['success']),
                        'output_dir': f"{base_dir}_restart{r['restart_idx']}/outputs" if r['success'] else None,
                        'files': r.get('result', {}).get('_generated_files', {}) if r['success'] else {}
                    }
                    for r in results
                ],
                'best': {
                    'restart_idx': best_result['_restart_info']['restart_idx'],
                    'strategy': best_result['_restart_info']['strategy'],
                    'cost': best_result['cost'],
                    'output_dir': best_result.get('_generated_files', {})
                }
            }
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"Multi-restart summary saved to: {summary_file}")
        
        # Generate comparison HTML
        try:
            _generate_comparison_html(results, base_dir, n_restarts)
            if verbose:
                print(f"Comparison page saved to: {base_dir}/comparison.html")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Comparison page generation failed: {e}")
    
    return best_result if best_result else results[0]['result'] if results else None

def _generate_comparison_html(results, base_dir, n_restarts):
    """Generate HTML comparison page for all trials."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Restart Optimization Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .trial-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
        .trial-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .trial-header {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
        .best-badge {{ background: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }}
        .cost {{ font-size: 24px; color: #3498db; margin: 10px 0; }}
        .files {{ margin-top: 15px; }}
        .file-link {{ display: block; padding: 8px; margin: 4px 0; background: #ecf0f1; border-radius: 4px; text-decoration: none; color: #2c3e50; }}
        .file-link:hover {{ background: #bdc3c7; }}
        .audio {{ margin-top: 10px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>🎯 Multi-Restart Optimization Results</h1>
    <div class="summary">
        <p><strong>Total Trials:</strong> {n_restarts}</p>
        <p><strong>Successful:</strong> {sum(1 for r in results if r['success'])}/{n_restarts}</p>
    </div>
    
    <div class="trial-grid">
"""
    
    # Find best trial
    successful = [r for r in results if r['success']]
    best_idx = None
    if successful:
        best_idx = min(successful, key=lambda r: r['cost'])['restart_idx']
    
    for r in results:
        if not r['success']:
            continue
        
        is_best = (r['restart_idx'] == best_idx)
        result_obj = r['result']
        restart_idx = r['restart_idx']
        
        html += f"""
        <div class="trial-card">
            <div class="trial-header">
                Restart {restart_idx + 1} ({r['strategy']})
                {f'<span class="best-badge">BEST</span>' if is_best else ''}
            </div>
            <div class="cost">Cost: {r['cost']:.6f}</div>
            <p><strong>Active Partials:</strong> {len(result_obj['active_partials'])}</p>
            
            <div class="files">
                <strong>Visualizations:</strong>
"""
        
        # Add visualization links
        output_dir = f"{base_dir}_restart{restart_idx}/outputs"
        viz_files = [
            ('frequency_migration.png', '📊 Frequency Migration'),
            ('harmonicity_map.png', '🎵 Harmonicity Map'),
            ('dissonance_landscape.png', '🌄 Dissonance Landscape'),
            ('adsr_comparison.png', '⏱️  ADSR Comparison')
        ]
        
        for filename, label in viz_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                rel_path = os.path.relpath(filepath, base_dir)
                html += f'<a href="{rel_path}" class="file-link" target="_blank">{label}</a>\n'
        
        html += """
                <br><strong>Audio Files:</strong>
"""
        
        # Add audio links
        audio_files = [
            ('output_standard.wav', '🔊 Standard Timbre'),
            ('output_optimized.wav', '✨ Optimized Timbre')
        ]
        
        for filename, label in audio_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                rel_path = os.path.relpath(filepath, base_dir)
                html += f'<a href="{rel_path}" class="file-link" download>{label}</a>\n'
                html += f'<audio controls class="audio"><source src="{rel_path}" type="audio/wav"></audio>\n'
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    # Save HTML
    html_path = os.path.join(base_dir, 'comparison.html')
    with open(html_path, 'w') as f:
        f.write(html)

