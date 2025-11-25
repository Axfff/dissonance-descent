import matplotlib.pyplot as plt
import numpy as np
from src.dissonance import calculate_total_dissonance

def plot_harmonicity_map(slices, fixed_ratios, standard_amplitudes, optimized_amplitudes, output_file='harmonicity_map.png'):
    """
    Plots the dissonance over time for both standard and optimized timbres.
    
    Args:
        slices (list): List of time slices.
        fixed_ratios (list): Fixed partial ratios [r2, r3, ...].
        standard_amplitudes (list): Standard partial amplitudes [a2, a3, ...].
        optimized_amplitudes (list): Optimized partial amplitudes [a2, a3, ...].
        output_file (str): Path to save the plot.
    """
    
    times = []
    dissonance_std = []
    dissonance_opt = []
    
    current_time = 0.0
    
    # Full ratios (including fundamental)
    ratios = [1.0] + list(fixed_ratios)
    
    # Full amplitudes (including fundamental)
    std_amps = [1.0] + list(standard_amplitudes)
    opt_amps = [1.0] + list(optimized_amplitudes)
    
    for duration, fundamentals in slices:
        if not fundamentals:
            # Silence
            times.append(current_time)
            dissonance_std.append(0)
            dissonance_opt.append(0)
            current_time += duration
            times.append(current_time)
            dissonance_std.append(0)
            dissonance_opt.append(0)
            continue
            
        # Calculate dissonance for this slice
        
        # Standard
        std_freqs = []
        std_a = []
        for f0, amp0 in fundamentals:
            for r, a in zip(ratios, std_amps):
                std_freqs.append(f0 * r)
                std_a.append(a * amp0)
        d_std = calculate_total_dissonance(std_freqs, std_a)
        
        # Optimized
        opt_freqs = []
        opt_a = []
        for f0, amp0 in fundamentals:
            for r, a in zip(ratios, opt_amps):
                opt_freqs.append(f0 * r)
                opt_a.append(a * amp0)
        d_opt = calculate_total_dissonance(opt_freqs, opt_a)
        
        # Add points for start and end of slice to create step plot
        times.append(current_time)
        dissonance_std.append(d_std)
        dissonance_opt.append(d_opt)
        
        current_time += duration
        times.append(current_time)
        dissonance_std.append(d_std)
        dissonance_opt.append(d_opt)
        
    plt.figure(figsize=(14, 6))
    plt.plot(times, dissonance_std, label='Standard Timbre (Harmonic)', alpha=0.7)
    plt.plot(times, dissonance_opt, label='Optimized Timbre', alpha=0.9, linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Perceptual Roughness')
    plt.title('Harmonicity Map: Dissonance Flow Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file)
    print(f"Harmonicity map saved to {output_file}")
