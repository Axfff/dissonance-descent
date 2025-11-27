import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.dissonance import calculate_song_dissonance_enhanced
from datetime import datetime
import os

class OptimizationCallback:
    """
    Callback class to track optimization progress and save intermediate results.
    """
    def __init__(self, output_dir='experiments/optimization_progress', save_every=5):
        self.iteration = 0
        self.eval_count = 0  # Track function evaluations
        self.history = {
            'iterations': [],
            'costs': [],
            'timestamps': []
        }
        self.output_dir = output_dir
        self.save_every = save_every
        self.start_time = datetime.now()
        self.last_cost = None  # Track last cost
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Progress tracking enabled: saving to {output_dir}")
        print(f"Saving every {save_every} iterations")
    
    def __call__(self, xk):
        """
        Called by scipy.optimize.minimize at each iteration.
        
        Args:
            xk: Current parameter vector
        """
        self.iteration += 1
        
        # Record iteration data
        self.history['iterations'].append(self.iteration)
        self.history['timestamps'].append((datetime.now() - self.start_time).total_seconds())
        
        # Record the last cost (from the cost function)
        if self.last_cost is not None:
            self.history['costs'].append(self.last_cost)
        
        # Print progress
        if self.iteration % self.save_every == 0:
            elapsed = self.history['timestamps'][-1]
            if len(self.history['costs']) > 0:
                current_cost = self.history['costs'][-1]
                print(f"  Iteration {self.iteration}: cost={current_cost:.6f}, elapsed={elapsed:.1f}s")
    
    def record_cost(self, cost):
        """Record the cost for tracking. Called from cost function."""
        self.last_cost = cost
        self.eval_count += 1
    
    def save_progress(self, metadata, decode_fn, params):
        """
        Save current optimization state.
        
        Args:
            metadata: Optimization metadata for decoding
            decode_fn: Function to decode parameters into partials
            params: Current parameter vector
        """
        # Save history as JSON
        history_file = os.path.join(self.output_dir, 'history.json')
        save_data = {
            'iterations': self.history['iterations'],
            'costs': self.history['costs'],
            'timestamps': self.history['timestamps'],
            'total_iterations': self.iteration,
            'total_evaluations': self.eval_count
        }
        with open(history_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save latest parameters
        partials = decode_fn(params, metadata)
        
        partials_file = os.path.join(self.output_dir, 'latest_partials.json')
        with open(partials_file, 'w') as f:
            json.dump(partials, f, indent=2)
        
        # Create convergence plot
        self.plot_convergence()
    
    def plot_convergence(self):
        """Create and save convergence plot."""
        if len(self.history['costs']) < 2:
            return
        
        # Make sure we have matching lengths
        n_points = min(len(self.history['iterations']), len(self.history['costs']))
        if n_points < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cost vs iteration
        ax1.plot(
            self.history['iterations'][:n_points], 
            self.history['costs'][:n_points], 
            'b-', linewidth=2, alpha=0.7
        )
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Dissonance Cost')
        ax1.set_title('Optimization Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Cost vs time
        ax2.plot(
            self.history['timestamps'][:n_points], 
            self.history['costs'][:n_points], 
            'g-', linewidth=2, alpha=0.7
        )
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Dissonance Cost')
        ax2.set_title('Convergence vs Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'convergence.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
    
    def create_checkpoint(self, xk, metadata, decode_fn, iteration_label):
        """
        Create a detailed checkpoint with visualizations.
        
        Args:
            xk: Current parameter vector
            metadata: Optimization metadata
            decode_fn: Function to decode parameters
            iteration_label: Label for this checkpoint (e.g., 'iter_050')
        """
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint_{iteration_label}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Decode parameters
        partials = decode_fn(xk, metadata)
        
        # Save partials
        with open(os.path.join(checkpoint_dir, 'partials.json'), 'w') as f:
            json.dump(partials, f, indent=2)
        
        # Create frequency spectrum plot
        ratios = [p['ratio'] for p in partials]
        amplitudes = [p['amplitude'] for p in partials]
        
        plt.figure(figsize=(12, 5))
        plt.bar(ratios, amplitudes, width=0.08, alpha=0.7, color='purple')
        plt.xlabel('Frequency Ratio')
        plt.ylabel('Amplitude')
        plt.title(f'Frequency Spectrum - Iteration {self.iteration}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(checkpoint_dir, 'spectrum.png'), dpi=150)
        plt.close()
        
        # If ADSR is enabled, save envelope info
        if metadata.get('optimize_adsr', False):
            active_partials = [p for p in partials if p['amplitude'] > 0.05][:6]
            
            if len(active_partials) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                
                for i, partial in enumerate(active_partials):
                    if i >= 6:
                        break
                    
                    ax = axes[i]
                    env = partial['envelope']
                    
                    # Simple envelope visualization
                    x = [0, env['attack'], env['attack'] + env['decay'], 0.5, 0.5 + env['release']]
                    y = [0, 1, env['sustain'], env['sustain'], 0]
                    
                    ax.plot(x, y, 'b-', linewidth=2)
                    ax.fill_between(x, 0, y, alpha=0.3)
                    ax.set_title(f"Ratio {partial['ratio']:.2f} (Amp {partial['amplitude']:.3f})")
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([-0.05, 1.05])
                
                for i in range(len(active_partials), 6):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(os.path.join(checkpoint_dir, 'envelopes.png'), dpi=150)
                plt.close()
        
        print(f"  Checkpoint saved: {checkpoint_dir}")

def create_frequency_grid(min_ratio, max_ratio, step):
    """
    Creates a dense frequency grid.
    
    Args:
        min_ratio (float): Minimum frequency ratio (e.g., 1.0)
        max_ratio (float): Maximum frequency ratio (e.g., 10.0)
        step (float): Step size (e.g., 0.2)
        
    Returns:
        np.array: Array of frequency ratios
    """
    return np.arange(min_ratio, max_ratio + step/2, step)

def encode_parameters(partials, optimize_adsr):
    """
    Encodes structured partial parameters into a flat vector for optimization.
    
    Args:
        partials (list): List of dicts with 'ratio', 'amplitude', 'envelope'
        optimize_adsr (bool): Whether to include ADSR in optimization
        
    Returns:
        np.array: Flat parameter vector
        dict: Metadata for decoding
    """
    params = []
    metadata = {
        'num_partials': len(partials),
        'optimize_adsr': optimize_adsr,
        'ratios': []
    }
    
    for partial in partials:
        metadata['ratios'].append(partial['ratio'])
        params.append(partial['amplitude'])
        
        if optimize_adsr:
            env = partial['envelope']
            params.extend([
                env['attack'],
                env['decay'],
                env['sustain'],
                env['release']
            ])
    
    return np.array(params), metadata

def decode_parameters(params, metadata):
    """
    Decodes flat parameter vector back into structured partials.
    
    Args:
        params (np.array): Flat parameter vector
        metadata (dict): Metadata from encoding
        
    Returns:
        list: List of partial dicts with 'ratio', 'amplitude', 'envelope'
    """
    partials = []
    optimize_adsr = metadata['optimize_adsr']
    params_per_partial = 5 if optimize_adsr else 1
    
    for i in range(metadata['num_partials']):
        idx = i * params_per_partial
        
        partial = {
            'ratio': metadata['ratios'][i],
            'amplitude': params[idx]
        }
        
        if optimize_adsr:
            partial['envelope'] = {
                'attack': params[idx + 1],
                'decay': params[idx + 2],
                'sustain': params[idx + 3],
                'release': params[idx + 4]
            }
        else:
            # Keep original envelope (will be passed separately)
            partial['envelope'] = None
            
        partials.append(partial)
    
    return partials

def create_initial_partials(config_partials, frequency_grid, optimize_adsr):
    """
    Creates initial partial configuration for optimization.
    
    Args:
        config_partials (list): Original partials from config
        frequency_grid (np.array or None): Dense frequency grid, or None for fixed ratios
        optimize_adsr (bool): Whether ADSR is being optimized
        
    Returns:
        list: Initial partials for optimization
    """
    if frequency_grid is not None:
        # Map config partials to grid and fill gaps with zero amplitude
        initial_partials = []
        config_ratios = [p['ratio'] for p in config_partials]
        
        for ratio in frequency_grid:
            # Find closest config partial
            closest_idx = np.argmin([abs(ratio - cr) for cr in config_ratios])
            closest_partial = config_partials[closest_idx]
            
            # Use config amplitude if close, otherwise zero
            if abs(ratio - config_ratios[closest_idx]) < 0.05:
                amplitude = closest_partial['amplitude']
                envelope = closest_partial['envelope'].copy()
            else:
                amplitude = 0.0
                # Use average envelope from config
                envelope = {
                    'attack': 0.01,
                    'decay': 0.05,
                    'sustain': 0.5,
                    'release': 0.15
                }
            
            initial_partials.append({
                'ratio': ratio,
                'amplitude': amplitude,
                'envelope': envelope
            })
    else:
        # Use config partials as-is
        initial_partials = [p.copy() for p in config_partials]
        for p in initial_partials:
            p['envelope'] = p['envelope'].copy()
    
    return initial_partials

def optimize_timbre_enhanced(slices, config, model_params=None):
    """
    Enhanced optimization of timbre parameters including ADSR and dense frequency grid.
    
    Args:
        slices (list): List of time slices from MIDI parser.
        config (dict): Full config dict including optimization settings.
        model_params (dict, optional): Plomp-Levelt parameters.
        
    Returns:
        dict: Optimization results with 'partials', 'cost', 'success', etc.
    """
    opt_config = config['optimization']
    timbre_config = config['timbre']
    
    # Create frequency grid if enabled
    if opt_config['frequency_grid']['enabled']:
        frequency_grid = create_frequency_grid(
            opt_config['frequency_grid']['min_ratio'],
            opt_config['frequency_grid']['max_ratio'],
            opt_config['frequency_grid']['step']
        )
        print(f"Using dense frequency grid with {len(frequency_grid)} frequencies")
        print(f"Grid: {frequency_grid[:5]}...{frequency_grid[-5:]}")
    else:
        frequency_grid = None
        print("Using fixed frequency ratios from config")
    
    optimize_adsr = opt_config['optimize_adsr']['enabled']
    print(f"ADSR optimization: {optimize_adsr}")
    
    # Check if randomization is requested
    randomization_config = config.get('_randomization', None)
    if randomization_config:
        # Multi-restart mode: use randomized initialization
        from src.randomization import create_random_initial_partials
        
        strategy = randomization_config.get('strategy', 'perturb')
        amp_pert = randomization_config.get('amp_perturbation', 0.2)
        adsr_pert = randomization_config.get('adsr_perturbation', 0.2)
        
        print(f"Randomization: {strategy} (amp±{amp_pert*100:.0f}%, adsr±{adsr_pert*100:.0f}%)")
        
        initial_partials = create_random_initial_partials(
            timbre_config['partials'],
            frequency_grid,
            optimize_adsr,
            amp_perturbation=amp_pert,
            adsr_perturbation=adsr_pert,
            strategy=strategy
        )
    else:
        # Standard mode: use config-based initialization
        initial_partials = create_initial_partials(
            timbre_config['partials'],
            frequency_grid,
            optimize_adsr
        )
    
    # Encode parameters
    initial_params, metadata = encode_parameters(initial_partials, optimize_adsr)
    
    print(f"Total optimization parameters: {len(initial_params)}")
    print(f"Parameters per partial: {5 if optimize_adsr else 1}")
    
    # Calculate initial total amplitude for conservation
    initial_amplitudes = initial_params[::5 if optimize_adsr else 1]
    initial_total_amp = np.sum(initial_amplitudes)
    print(f"Initial total amplitude: {initial_total_amp:.4f}")
    
    # Create bounds
    bounds = []
    params_per_partial = 5 if optimize_adsr else 1
    
    for i in range(len(initial_partials)):
        # Amplitude bounds
        bounds.append((0.0, 1.0))
        
        if optimize_adsr:
            # ADSR bounds from config
            bounds.append(tuple(opt_config['optimize_adsr']['attack_bounds']))
            bounds.append(tuple(opt_config['optimize_adsr']['decay_bounds']))
            bounds.append(tuple(opt_config['optimize_adsr']['sustain_bounds']))
            bounds.append(tuple(opt_config['optimize_adsr']['release_bounds']))
    
    # Create constraints
    constraints = []
    
    # Amplitude conservation constraint
    if opt_config['constraints']['total_amplitude'] == 'conserve':
        def amplitude_sum_constraint(params):
            amplitudes = params[::params_per_partial]
            return np.sum(amplitudes) - initial_total_amp
        
        constraints.append({
            'type': 'eq',
            'fun': amplitude_sum_constraint
        })
        print("Constraint: Total amplitude conservation enabled")
    
    
    # Store original envelopes for non-optimized case
    original_envelopes = [p['envelope'].copy() for p in initial_partials]
    
    # Extract sparsity penalty
    sparsity_penalty = opt_config['constraints']['sparsity_penalty']
    
    # Create optimization callback for progress tracking
    checkpoint_interval = opt_config.get('checkpoint_interval', 10)  
    progress_dir = opt_config.get('progress_dir', 'experiments/optimization_progress')
    
    callback = OptimizationCallback(
        output_dir=progress_dir,
        save_every=5  # Print every 5 iterations
    )
    
    
    # Wrap cost function to record costs in callback
    def cost_function_with_tracking(params):
        # Decode parameters
        partials = decode_parameters(params, metadata)
        
        # If not optimizing ADSR, use original envelopes
        if not optimize_adsr:
            for i, partial in enumerate(partials):
                partial['envelope'] = original_envelopes[i]
        
        # Calculate dissonance
        d = calculate_song_dissonance_enhanced(partials, slices, model_params)
        
        # Add optional sparsity penalty
        if sparsity_penalty > 0:
            amplitudes = params[::params_per_partial]
            # Penalty for having many non-zero amplitudes
            non_zero_count = np.sum(amplitudes > 0.01)
            d += sparsity_penalty * non_zero_count
        
        # Record cost in callback
        callback.record_cost(d)
        
        return d
    
    # Wrapper callback that saves progress and creates checkpoints
    def combined_callback(xk):
        # Call the iteration callback first
        callback(xk)
        
        # Save progress periodically
        if callback.iteration % callback.save_every == 0:
            callback.save_progress(metadata, decode_parameters, xk)
        
        # Create detailed checkpoint at intervals
        if callback.iteration % checkpoint_interval == 0:
            iteration_label = f"{callback.iteration:04d}"
            callback.create_checkpoint(xk, metadata, decode_parameters, iteration_label)
    
    
    
    # Run optimization
    print("\n--- Starting Enhanced Optimization ---")
    
    # For large parameter spaces, L-BFGS-B is more efficient than SLSQP
    optimizer_method = 'L-BFGS-B' if len(initial_params) > 100 else 'SLSQP'
    print(f"Using optimizer: {optimizer_method}")
    
    result = minimize(
        cost_function_with_tracking,
        initial_params,
        method=optimizer_method,
        bounds=bounds,
        constraints=constraints if optimizer_method == 'SLSQP' else (),  # L-BFGS-B doesn't support eq constraints
        callback=combined_callback,
        options={'disp': True, 'maxiter': 200, 'ftol': 1e-6}
    )
    
    # Decode final parameters
    optimized_partials = decode_parameters(result.x, metadata)
    
    # If not optimizing ADSR, restore original envelopes
    if not optimize_adsr:
        for i, partial in enumerate(optimized_partials):
            partial['envelope'] = original_envelopes[i]
    
    
    # Filter out near-zero amplitude partials
    active_partials = [p for p in optimized_partials if p['amplitude'] > 0.001]
    
    # Final progress save
    callback.save_progress(metadata, decode_parameters, result.x)
    print(f"\nProgress saved to: {callback.output_dir}")
    
    print("\n--- Optimization Complete ---")
    print(f"Converged: {result.success}")
    print(f"Final cost: {result.fun:.6f}")
    print(f"Active partials: {len(active_partials)} / {len(optimized_partials)}")
    
    final_amplitudes = result.x[::params_per_partial]
    print(f"Final total amplitude: {np.sum(final_amplitudes):.4f}")
    
    return {
        'partials': optimized_partials,
        'active_partials': active_partials,
        'cost': result.fun,
        'success': result.success,
        'raw_result': result,
        'metadata': metadata,
        'callback': callback  # Include callback for access to history
    }
