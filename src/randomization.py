"""
Smart randomization utilities for optimization initialization.
Implements constraint-aware random parameter generation.
"""
import numpy as np

def randomize_amplitudes(base_amplitudes, perturbation=0.1, preserve_energy=True):
    """
    Randomize amplitudes with optional energy conservation.
    
    Args:
        base_amplitudes: Base amplitude values
        perturbation: Relative perturbation amount (0.1 = ±10%)
        preserve_energy: Maintain total amplitude sum
        
    Returns:
        Randomized amplitudes
    """
    amplitudes = np.array(base_amplitudes)
    
    # Add random perturbation
    noise = np.random.uniform(-perturbation, perturbation, len(amplitudes))
    randomized = amplitudes * (1 + noise)
    
    # Clip to valid range
    randomized = np.clip(randomized, 0.0, 1.0)
    
    # Preserve energy if requested
    if preserve_energy:
        original_sum = np.sum(amplitudes)
        current_sum = np.sum(randomized)
        if current_sum > 0:
            randomized = randomized * (original_sum / current_sum)
    
    return randomized

def randomize_envelope(base_envelope, perturbation=0.2, bounds=None):
    """
    Randomize ADSR envelope parameters within valid bounds.
    
    Args:
        base_envelope: Dict with attack, decay, sustain, release
        perturbation: Relative perturbation amount
        bounds: Dict of (min, max) tuples for each parameter
        
    Returns:
        Randomized envelope dict
    """
    if bounds is None:
        bounds = {
            'attack': (0.001, 0.1),
            'decay': (0.01, 0.3),
            'sustain': (0.2, 1.0),
            'release': (0.02, 0.5)
        }
    
    randomized = {}
    for key in ['attack', 'decay', 'sustain', 'release']:
        base_val = base_envelope[key]
        noise = np.random.uniform(-perturbation, perturbation)
        new_val = base_val * (1 + noise)
        
        # Clip to bounds
        min_val, max_val = bounds[key]
        randomized[key] = np.clip(new_val, min_val, max_val)
    
    return randomized

def create_random_initial_partials(config_partials, frequency_grid, optimize_adsr,
                                   amp_perturbation=0.2, adsr_perturbation=0.2,
                                   strategy='perturb'):
    """
    Create randomized initial partials using various strategies.
    
    Args:
        config_partials: Original partials from config
        frequency_grid: Dense frequency grid (or None)
        optimize_adsr: Whether ADSR is being optimized
        amp_perturbation: Amplitude randomization amount
        adsr_perturbation: ADSR randomization amount
        strategy: 'perturb', 'random', or 'smart'
        
    Returns:
        List of randomized partial dicts
    """
    if strategy == 'perturb':
        # Strategy A: Small perturbation around config
        return _perturb_partials(config_partials, frequency_grid, optimize_adsr,
                                amp_perturbation, adsr_perturbation)
    
    elif strategy == 'random':
        # Full randomization within bounds
        return _fully_random_partials(frequency_grid, optimize_adsr)
    
    elif strategy == 'smart':
        # Strategy C: Smart constraint-aware randomization
        return _smart_random_partials(config_partials, frequency_grid, optimize_adsr,
                                     amp_perturbation, adsr_perturbation)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _perturb_partials(config_partials, frequency_grid, optimize_adsr,
                     amp_perturbation, adsr_perturbation):
    """Strategy A: Perturb config-based initialization."""
    from src.optimizer_enhanced import create_initial_partials
    
    # Start with config-based initialization
    partials = create_initial_partials(config_partials, frequency_grid, optimize_adsr)
    
    # Extract amplitudes
    amps = [p['amplitude'] for p in partials]
    
    # Randomize with energy conservation
    randomized_amps = randomize_amplitudes(amps, amp_perturbation, preserve_energy=True)
    
    # Update partials
    for i, partial in enumerate(partials):
        partial['amplitude'] = randomized_amps[i]
        
        if optimize_adsr and partial.get('envelope'):
            partial['envelope'] = randomize_envelope(
                partial['envelope'],
                adsr_perturbation
            )
    
    return partials

def _smart_random_partials(config_partials, frequency_grid, optimize_adsr,
                          amp_perturbation, adsr_perturbation):
    """Strategy C: Smart constraint-aware randomization."""
    
    # Use frequency grid if available
    if frequency_grid is not None:
        ratios = frequency_grid
    else:
        ratios = [p['ratio'] for p in config_partials]
    
    # Calculate total energy from config
    total_energy = sum(p['amplitude'] for p in config_partials)
    
    # Generate random but normalized amplitudes
    n = len(ratios)
    random_amps = np.random.uniform(0.0, 1.0, n)
    
    # Normalize to preserve total energy
    if np.sum(random_amps) > 0:
        random_amps = random_amps * (total_energy / np.sum(random_amps))
    
    # Prefer lower frequencies (musical prior)
    # Apply decay based on frequency rank
    decay_factor = np.exp(-np.arange(n) * 0.1)
    random_amps = random_amps * decay_factor
    
    # Re-normalize
    if np.sum(random_amps) > 0:
        random_amps = random_amps * (total_energy / np.sum(random_amps))
    
    # Create partials
    partials = []
    for i, ratio in enumerate(ratios):
        partial = {
            'ratio': ratio,
            'amplitude': random_amps[i]
        }
        
        if optimize_adsr:
            # Random envelope within reasonable bounds
            partial['envelope'] = {
                'attack': np.random.uniform(0.001, 0.05),
                'decay': np.random.uniform(0.02, 0.15),
                'sustain': np.random.uniform(0.4, 0.9),
                'release': np.random.uniform(0.05, 0.3)
            }
        else:
            # Use config envelope
            if i < len(config_partials):
                partial['envelope'] = config_partials[i]['envelope'].copy()
            else:
                partial['envelope'] = {
                    'attack': 0.01,
                    'decay': 0.05,
                    'sustain': 0.7,
                    'release': 0.15
                }
        
        partials.append(partial)
    
    return partials

def _fully_random_partials(frequency_grid, optimize_adsr):
    """Full uniform randomization (less smart)."""
    if frequency_grid is not None:
        ratios = frequency_grid
    else:
        ratios = np.arange(1.0, 10.0, 0.5)
    
    partials = []
    for ratio in ratios:
        partial = {
            'ratio': ratio,
            'amplitude': np.random.uniform(0.0, 1.0)
        }
        
        if optimize_adsr:
            partial['envelope'] = {
                'attack': np.random.uniform(0.001, 0.1),
                'decay': np.random.uniform(0.01, 0.3),
                'sustain': np.random.uniform(0.2, 1.0),
                'release': np.random.uniform(0.02, 0.5)
            }
        
        partials.append(partial)
    
    return partials
