import numpy as np
from scipy.optimize import minimize
from src.dissonance import calculate_song_dissonance

def optimize_timbre(slices, fixed_ratios, initial_amplitudes=None, model_params=None):
    """
    Optimizes the timbre parameters (partial amplitudes) to minimize the global dissonance.
    
    Args:
        slices (list): List of time slices from MIDI parser.
        fixed_ratios (list): List of fixed partial ratios [r2, r3, ...].
        initial_amplitudes (list, optional): Initial guess for amplitudes [a2, a3, ...].
        model_params (dict, optional): Plomp-Levelt parameters.
        
    Returns:
        scipy.optimize.OptimizeResult: The result of the optimization.
    """
    
    # Default initial guess: All 0.5 if not provided
    if initial_amplitudes is None:
        initial_amplitudes = [0.5] * len(fixed_ratios)
        
    # Calculate initial total amplitude to conserve
    initial_total_amp = np.sum(initial_amplitudes)
    
    # Bounds: Amplitudes between 0.0 and 1.0
    bounds = [(0.0, 1.0) for _ in range(len(initial_amplitudes))]
    
    # Constraints: Sum of amplitudes must equal initial sum
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - initial_total_amp}
    ]
    
    print("Starting optimization (Amplitudes)...")
    print(f"Fixed Ratios: {fixed_ratios}")
    print(f"Initial Amplitudes: {initial_amplitudes}")
    print(f"Target Total Amplitude: {initial_total_amp}")
    
    # Wrapper for the cost function
    def cost_function(params):
        d = calculate_song_dissonance(params, slices, fixed_ratios, model_params)
        return d
        
    result = minimize(
        cost_function,
        initial_amplitudes,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 100}
    )
    
    print("Optimization complete.")
    print(f"Final amplitudes: {result.x}")
    print(f"Final total amplitude: {np.sum(result.x)}")
    print(f"Final cost: {result.fun}")
    
    return result
