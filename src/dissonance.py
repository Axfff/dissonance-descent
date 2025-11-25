import numpy as np
from .synthesizer import calculate_envelope_average

def calculate_roughness_pair(f1, f2, a1, a2, params=None):
    """
    Calculates the roughness (dissonance) between two partials.
    Based on the Sethares (1993) implementation of Plomp-Levelt.
    params: dict containing 'a', 'b', 's1', 's2'
    """
    # Ensure f1 < f2
    f_min = min(f1, f2)
    f_max = max(f1, f2)
    
    # Default Sethares parameters
    if params is None:
        a = 3.5
        b = 5.75
        s1 = 0.021
        s2 = 19.0
    else:
        a = params.get('a', 3.5)
        b = params.get('b', 5.75)
        s1 = params.get('s1', 0.021)
        s2 = params.get('s2', 19.0)
    
    x = f_max - f_min
    
    # Avoid division by zero if f_min is 0 (though unlikely in music)
    if f_min == 0:
        return 0.0
        
    s = x / (s1 * f_min + s2)
    
    term1 = np.exp(-a * s)
    term2 = np.exp(-b * s)
    
    roughness = (a1 * a2) * (term1 - term2)
    
    return roughness

def calculate_total_dissonance(frequencies, amplitudes, model_params=None):
    """
    Calculates the total dissonance of a spectrum.
    frequencies: list or array of frequencies
    amplitudes: list or array of amplitudes
    model_params: dict of Plomp-Levelt parameters
    """
    frequencies = np.array(frequencies)
    amplitudes = np.array(amplitudes)
    
    # Flatten if necessary (e.g. if passed a list of lists)
    frequencies = frequencies.flatten()
    amplitudes = amplitudes.flatten()
    
    n = len(frequencies)
    total_dissonance = 0.0
    
    # Iterate over all unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            d = calculate_roughness_pair(frequencies[i], frequencies[j], amplitudes[i], amplitudes[j], params=model_params)
            total_dissonance += d
            
    # Normalization: Divide by the square of the sum of amplitudes
    # This makes the measure independent of the overall loudness, 
    # but sensitive to the relative distribution of energy.
    total_amp = np.sum(amplitudes)
    if total_amp > 0:
        total_dissonance /= (total_amp ** 2)
            
    return total_dissonance

def calculate_song_dissonance(timbre_params, slices, fixed_ratios, model_params=None):
    """
    Calculates the global dissonance of the song given a timbre.
    
    Args:
        timbre_params (list): List of partial amplitudes [a2, a3, ...]. 
                              a1 (fundamental) is always 1.0.
        slices (list): List of tuples (duration, [(freq1, amp1), ...]) from parse_midi_to_slices.
        fixed_ratios (list): List of fixed partial ratios [r2, r3, ...].
        model_params (dict): Plomp-Levelt parameters.
        
    Returns:
        float: The total integrated dissonance over time.
    """
    # Construct the full timbre model
    # timbre_params are now AMPLITUDES for partials 2..n
    # fixed_ratios are RATIOS for partials 2..n
    
    # Fundamental: ratio=1.0, amp=1.0
    ratios = [1.0] + list(fixed_ratios)
    amplitudes = [1.0] + list(timbre_params)
    
    total_song_dissonance = 0.0
    
    for duration, fundamentals in slices:
        if not fundamentals:
            continue
            
        # For this slice, calculate all active partials
        slice_freqs = []
        slice_amps = []
        
        for f0, amp0 in fundamentals:
            for r, a in zip(ratios, amplitudes):
                slice_freqs.append(f0 * r)
                slice_amps.append(a * amp0)
        
        # Calculate dissonance for this slice
        d = calculate_total_dissonance(slice_freqs, slice_amps, model_params)
        
        # Integrate over time
        total_song_dissonance += d * duration
        
    return total_song_dissonance

def calculate_song_dissonance_with_envelopes(timbre_params, slices, fixed_ratios, envelope_params, 
                                              brightness_decay=1.0, model_params=None):
    """
    Calculates the global dissonance of the song given a timbre with envelope information.
    Uses time-averaged amplitudes based on envelope shapes for more accurate perceptual modeling.
    
    Args:
        timbre_params (list): List of partial amplitudes [a2, a3, ...]. a1 (fundamental) is always 1.0.
        slices (list): List of tuples (duration, [(freq1, amp1), ...]) from parse_midi_to_slices.
        fixed_ratios (list): List of fixed partial ratios [r2, r3, ...].
        envelope_params (list): List of envelope parameter dicts for each partial (including fundamental).
        brightness_decay (float): Brightness decay factor for higher partials.
        model_params (dict): Plomp-Levelt parameters.
        
    Returns:
        float: The total integrated dissonance over time.
    """
    from .synthesizer import apply_brightness_decay
    
    # Construct the full timbre model
    ratios = [1.0] + list(fixed_ratios)
    amplitudes = [1.0] + list(timbre_params)
    
    total_song_dissonance = 0.0
    
    for duration, fundamentals in slices:
        if not fundamentals:
            continue
            
        # For this slice, calculate all active partials with envelope-weighted amplitudes
        slice_freqs = []
        slice_amps = []
        
        for f0, amp0 in fundamentals:
            for i, (r, a, env) in enumerate(zip(ratios, amplitudes, envelope_params)):
                # Apply brightness decay to envelope
                modified_env = apply_brightness_decay(env, i, brightness_decay)
                
                # Calculate time-averaged amplitude for this note duration
                avg_factor = calculate_envelope_average(modified_env, duration)
                
                slice_freqs.append(f0 * r)
                # Use time-averaged amplitude instead of peak amplitude
                slice_amps.append(a * amp0 * avg_factor)
        
        # Calculate dissonance for this slice
        d = calculate_total_dissonance(slice_freqs, slice_amps, model_params)
        
        # Integrate over time
        total_song_dissonance += d * duration
        
    return total_song_dissonance

