import numpy as np
import matplotlib.pyplot as plt
import json
from src.dissonance import calculate_total_dissonance

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def plot_landscape_comparison(fixed_ratios, std_amps, opt_amps, output_file='dissonance_landscape.png'):
    """
    Plots the dissonance landscape (dissonance vs interval) for two different timbres.
    """
    print("Generating Dissonance Landscape Comparison...")
    
    # Load config for Plomp-Levelt params
    config = load_config()
    plomp_params = config.get('plomp_levelt', None)
    viz_config = config['visualization']
    
    base_freq = config['timbre']['base_freq']
    
    # 1. Define Probe Range
    range_ratio = viz_config.get('range_ratio', 2.1)
    resolution = viz_config.get('resolution', 500)
    
    probe_freqs_range = np.linspace(base_freq, base_freq * range_ratio, resolution)
    
    # Helper to calculate curve
    def calculate_curve(ratios, amps):
        # Full partials including fundamental
        full_ratios = [1.0] + list(ratios)
        full_amps = [1.0] + list(amps)
        
        # Base tone spectrum
        base_freqs = [base_freq * r for r in full_ratios]
        base_amps_vals = [a for a in full_amps] # Amplitude is relative
        
        scores = []
        for probe_f0 in probe_freqs_range:
            # Probe tone spectrum
            probe_freqs = [probe_f0 * r for r in full_ratios]
            probe_amps_vals = [a for a in full_amps]
            
            # Combine spectra
            combined_freqs = base_freqs + probe_freqs
            combined_amps = base_amps_vals + probe_amps_vals
            
            score = calculate_total_dissonance(combined_freqs, combined_amps, model_params=plomp_params)
            scores.append(score)
        return scores

    dissonance_std = calculate_curve(fixed_ratios, std_amps)
    dissonance_opt = calculate_curve(fixed_ratios, opt_amps)
            
    # 2. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(probe_freqs_range / base_freq, dissonance_std, label='Standard Timbre', alpha=0.8)
    plt.plot(probe_freqs_range / base_freq, dissonance_opt, label='Optimized Timbre', alpha=0.8, linewidth=2)
    
    # Mark standard intervals for reference
    intervals = {
        'Unison': 1.0,
        'm2': 16/15,
        'M2': 9/8,
        'm3': 6/5,
        'M3': 5/4,
        'P4': 4/3,
        'd5': 1.414, # approx tritone
        'P5': 3/2,
        'm6': 8/5,
        'M6': 5/3,
        'm7': 16/9,
        'M7': 15/8,
        'Octave': 2.0
    }
    
    max_val = max(max(dissonance_std), max(dissonance_opt))
    
    for name, ratio in intervals.items():
        if ratio <= range_ratio:
            plt.axvline(x=ratio, color='k', alpha=0.2, linestyle='--')
            plt.text(ratio, max_val, name, rotation=90, verticalalignment='top', fontsize=8)

    plt.title(f'Dissonance Landscape Comparison (Base: {base_freq}Hz)')
    plt.xlabel('Frequency Ratio (Probe / Base)')
    plt.ylabel('Roughness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    # Test run
    pass
