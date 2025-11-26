import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_migration(initial_partials, optimized_partials, output_file='frequency_migration.png'):
    """
    Plots how amplitude migrated across the frequency spectrum.
    
    Args:
        initial_partials (list): Initial partial configuration from config
        optimized_partials (list): Optimized partials from optimizer
        output_file (str): Output filename
    """
    # Extract ratios and amplitudes
    initial_ratios = [p['ratio'] for p in initial_partials]
    initial_amps = [p['amplitude'] for p in initial_partials]
    
    optimized_ratios = [p['ratio'] for p in optimized_partials]
    optimized_amps = [p['amplitude'] for p in optimized_partials]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot initial distribution
    ax1.bar(initial_ratios, initial_amps, width=0.08, alpha=0.7, color='blue', label='Initial')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Initial Timbre Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot optimized distribution
    ax2.bar(optimized_ratios, optimized_amps, width=0.08, alpha=0.7, color='green', label='Optimized')
    ax2.set_xlabel('Frequency Ratio (×fundamental)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Optimized Timbre Spectrum')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved frequency migration plot to {output_file}")

def plot_adsr_comparison(initial_partials, optimized_partials, output_file='adsr_comparison.png'):
    """
    Plots ADSR envelope comparisons for key partials.
    
    Args:
        initial_partials (list): Initial partial configuration from config
        optimized_partials (list): Optimized partials from optimizer
        output_file (str): Output filename
    """
    # Filter to partials with significant amplitude
    significant_optimized = sorted(
        [p for p in optimized_partials if p['amplitude'] > 0.05],
        key=lambda p: p['amplitude'],
        reverse=True
    )[:6]  # Top 6 partials
    
    if len(significant_optimized) == 0:
        print("No significant partials to plot ADSR comparison")
        return
    
    # Create subplots
    n_partials = len(significant_optimized)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, opt_partial in enumerate(significant_optimized):
        if i >= 6:
            break
            
        ax = axes[i]
        ratio = opt_partial['ratio']
        
        # Find matching initial partial (closest ratio)
        initial_partial = min(initial_partials, key=lambda p: abs(p['ratio'] - ratio))
        
        # Generate envelope curves
        duration = 1.0  # 1 second note for visualization
        sample_rate = 100  # Low rate for visualization
        
        def generate_envelope_curve(env, duration, sr):
            """Generate ADSR envelope samples"""
            attack_samples = int(env['attack'] * sr)
            decay_samples = int(env['decay'] * sr)
            release_samples = int(env['release'] * sr)
            sustain_samples = max(1, int(duration * sr) - attack_samples - decay_samples)
            
            # Attack
            attack = np.linspace(0, 1, attack_samples) if attack_samples > 0 else []
            # Decay
            decay = np.linspace(1, env['sustain'], decay_samples) if decay_samples > 0 else []
            # Sustain
            sustain = np.ones(sustain_samples) * env['sustain']
            # Release
            release = np.linspace(env['sustain'], 0, release_samples) if release_samples > 0 else []
            
            envelope = np.concatenate([attack, decay, sustain, release])
            time = np.arange(len(envelope)) / sr
            
            return time, envelope
        
        # Plot initial envelope
        t_init, env_init = generate_envelope_curve(initial_partial['envelope'], duration, sample_rate)
        ax.plot(t_init, env_init, 'b-', alpha=0.6, linewidth=2, label='Initial')
        
        # Plot optimized envelope
        t_opt, env_opt = generate_envelope_curve(opt_partial['envelope'], duration, sample_rate)
        ax.plot(t_opt, env_opt, 'g-', alpha=0.8, linewidth=2, label='Optimized')
        
        ax.set_title(f'Ratio {ratio:.2f} (Amp: {opt_partial["amplitude"]:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([-0.05, 1.05])
    
    # Hide unused subplots
    for i in range(n_partials, 6):
        axes[i].set_visible(False)
    
    plt.suptitle('ADSR Envelope Comparison: Initial vs Optimized', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved ADSR comparison plot to {output_file}")

def plot_optimization_summary(initial_partials, optimized_partials, optimization_result, output_file='optimization_summary.png'):
    """
    Creates a comprehensive summary of the optimization results.
    
    Args:
        initial_partials (list): Initial partial configuration
        optimized_partials (list): Optimized partials
        optimization_result (dict): Result dict from optimizer
        output_file (str): Output filename
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Frequency spectrum comparison
    ax1 = fig.add_subplot(gs[0, :])
    initial_ratios = [p['ratio'] for p in initial_partials]
    initial_amps = [p['amplitude'] for p in initial_partials]
    optimized_ratios = [p['ratio'] for p in optimized_partials]
    optimized_amps = [p['amplitude'] for p in optimized_partials]
    
    x = np.arange(len(optimized_ratios))
    width = 0.35
    
    # For matching, we need to align by ratio
    # This is complex for dense grid, so just plot both separately
    ax1.bar(initial_ratios, initial_amps, width=0.08, alpha=0.5, label='Initial', color='blue')
    ax1.bar(optimized_ratios, optimized_amps, width=0.08, alpha=0.7, label='Optimized', color='green')
    ax1.set_xlabel('Frequency Ratio')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Spectral Distribution: Initial vs Optimized')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Active partials count
    ax2 = fig.add_subplot(gs[1, 0])
    active_initial = sum(1 for p in initial_partials if p['amplitude'] > 0.01)
    active_optimized = sum(1 for p in optimized_partials if p['amplitude'] > 0.01)
    ax2.bar(['Initial', 'Optimized'], [active_initial, active_optimized], color=['blue', 'green'], alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Active Partials (amplitude > 0.01)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total amplitude
    ax3 = fig.add_subplot(gs[1, 1])
    total_initial = sum(p['amplitude'] for p in initial_partials)
    total_optimized = sum(p['amplitude'] for p in optimized_partials)
    ax3.bar(['Initial', 'Optimized'], [total_initial, total_optimized], color=['blue', 'green'], alpha=0.7)
    ax3.set_ylabel('Total Amplitude')
    ax3.set_title('Total Amplitude (Energy Conservation)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimization info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    info_text = f"""
    Optimization Results:
    - Status: {'Success' if optimization_result.get('success') else 'Failed'}
    - Final Cost: {optimization_result.get('cost', 0):.6f}
    - Initial Partials: {len(initial_partials)}
    - Active Optimized Partials: {len(optimization_result.get('active_partials', []))}
    - Total Optimized Partials: {len(optimized_partials)}
    """
    ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.suptitle('Enhanced Timbre Optimization Summary', fontsize=16, fontweight='bold')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved optimization summary to {output_file}")
