import numpy as np
import soundfile as sf

def generate_adsr_envelope(duration, attack, decay, sustain_level, release, sample_rate=44100):
    """
    Generates an ADSR envelope for a given duration.
    
    Args:
        duration: Note duration in seconds (not including release)
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain_level: Sustain level (0.0 to 1.0)
        release: Release time in seconds
        sample_rate: Sampling rate in Hz
        
    Returns:
        numpy array containing the envelope
    """
    # Total duration includes the release tail
    total_duration = duration + release
    num_samples = int(total_duration * sample_rate)
    envelope = np.ones(num_samples)
    
    # Calculate sample counts for each phase
    n_attack = int(attack * sample_rate)
    n_decay = int(decay * sample_rate)
    n_release = int(release * sample_rate)
    n_sustain = num_samples - n_attack - n_decay - n_release
    
    # Handle very short notes
    if n_sustain < 0:
        # Shorten attack and decay proportionally
        total_ad = n_attack + n_decay
        if total_ad > 0:
            scale = (num_samples - n_release) / total_ad
            n_attack = int(n_attack * scale)
            n_decay = int(n_decay * scale)
            n_sustain = 0
        else:
            # Extremely short note, just attack and release
            n_attack = min(n_attack, num_samples // 2)
            n_release = num_samples - n_attack
            n_decay = 0
            n_sustain = 0
    
    # Build envelope
    idx = 0
    
    # Attack phase
    if n_attack > 0:
        envelope[idx:idx+n_attack] = np.linspace(0, 1, n_attack)
        idx += n_attack
    
    # Decay phase
    if n_decay > 0:
        envelope[idx:idx+n_decay] = np.linspace(1, sustain_level, n_decay)
        idx += n_decay
    
    # Sustain phase
    if n_sustain > 0:
        envelope[idx:idx+n_sustain] = sustain_level
        idx += n_sustain
    
    # Release phase
    if n_release > 0:
        envelope[idx:idx+n_release] = np.linspace(sustain_level if n_sustain >= 0 else envelope[idx-1], 0, n_release)
    
    return envelope

def apply_brightness_decay(envelope_params, partial_index, brightness_decay):
    """
    Modifies envelope parameters based on partial index and brightness decay factor.
    Higher partials decay faster when brightness_decay > 1.0.
    
    Args:
        envelope_params: Dict with 'attack', 'decay', 'sustain', 'release'
        partial_index: Index of the partial (0 = fundamental)
        brightness_decay: Decay multiplier (1.0 = no effect, >1.0 = faster decay for higher partials)
        
    Returns:
        Modified envelope parameters dict
    """
    if brightness_decay == 1.0 or partial_index == 0:
        return envelope_params.copy()
    
    # Calculate decay factor: increases with partial index
    decay_factor = 1.0 / (1.0 + (partial_index * (brightness_decay - 1.0)))
    
    modified = envelope_params.copy()
    # Reduce release time for higher partials
    modified['release'] = envelope_params['release'] * decay_factor
    # Slightly reduce sustain level for higher partials
    modified['sustain'] = envelope_params['sustain'] * (0.7 + 0.3 * decay_factor)
    
    return modified

def generate_tone_with_envelopes(frequencies, amplitudes, envelopes, duration, phase_mode='zero', sample_rate=44100):
    """
    Generates an audio signal using additive synthesis with per-partial envelopes.
    
    Args:
        frequencies: List of frequencies (Hz) for each partial
        amplitudes: List of amplitudes (0.0 to 1.0) for each partial
        envelopes: List of envelope parameter dicts or None for each partial
        duration: Note duration in seconds (not including release)
        phase_mode: 'zero', 'random', or 'alternating'
        sample_rate: Sampling rate in Hz
        
    Returns:
        numpy array containing the audio signal
    """
    # Generate time array for the note duration (envelopes will extend this)
    # We need to find the longest envelope to determine final length
    max_duration = duration
    for env_params in envelopes:
        if env_params:
            max_duration = max(max_duration, duration + env_params.get('release', 0.2))
    
    t = np.linspace(0, max_duration, int(sample_rate * max_duration), endpoint=False)
    signal = np.zeros_like(t)
    
    # Generate initial phases
    phases = []
    for i in range(len(frequencies)):
        if phase_mode == 'zero':
            phases.append(0.0)
        elif phase_mode == 'random':
            phases.append(np.random.uniform(0, 2 * np.pi))
        elif phase_mode == 'alternating':
            phases.append(0.0 if i % 2 == 0 else np.pi)
        else:
            phases.append(0.0)
    
    # Generate each partial with its envelope
    for f, a, env_params, phase in zip(frequencies, amplitudes, envelopes, phases):
        # Generate sine wave
        partial = a * np.sin(2 * np.pi * f * t + phase)
        
        # Apply envelope if provided
        if env_params:
            envelope = generate_adsr_envelope(
                duration,
                env_params.get('attack', 0.02),
                env_params.get('decay', 0.1),
                env_params.get('sustain', 0.8),
                env_params.get('release', 0.2),
                sample_rate
            )
            
            # Ensure envelope matches signal length
            if len(envelope) > len(partial):
                envelope = envelope[:len(partial)]
            elif len(envelope) < len(partial):
                envelope = np.pad(envelope, (0, len(partial) - len(envelope)))
            
            partial *= envelope
        
        signal += partial
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val
    
    return signal

def generate_tone(frequencies, amplitudes, duration=2.0, sample_rate=44100):
    """
    Generates an audio signal using additive synthesis (legacy version).
    For backward compatibility. Consider using generate_tone_with_envelopes for better results.
    
    Args:
        frequencies: list of frequencies (Hz)
        amplitudes: list of amplitudes (0.0 to 1.0)
        duration: duration in seconds
        sample_rate: sampling rate in Hz
        
    Returns:
        numpy array containing the audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
        
    # Normalize signal to avoid clipping
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val
        
    return signal

def save_wav(filename, audio_data, sample_rate=44100):
    """
    Saves audio data to a .wav file.
    """
    sf.write(filename, audio_data, sample_rate)

def calculate_envelope_average(envelope_params, duration):
    """
    Calculates the time-averaged amplitude of an envelope over a note duration.
    This is useful for perceptually-weighted dissonance calculations.
    
    Args:
        envelope_params: Dict with 'attack', 'decay', 'sustain', 'release'
        duration: Note duration in seconds
        
    Returns:
        float: Time-averaged amplitude (0.0 to 1.0)
    """
    if not envelope_params:
        return 1.0  # Assume full amplitude if no envelope
    
    attack = envelope_params.get('attack', 0.02)
    decay = envelope_params.get('decay', 0.1)
    sustain = envelope_params.get('sustain', 0.8)
    
    # We don't include release in the average since it happens after note-off
    # Only consider the note duration itself
    
    total_time = duration
    
    # Attack contribution: average of triangle (0 to 1) = 0.5
    attack_contrib = 0.5 * min(attack, total_time)
    remaining_time = max(0, total_time - attack)
    
    # Decay contribution: average of trapezoid (1 to sustain)
    decay_time = min(decay, remaining_time)
    decay_contrib = ((1.0 + sustain) / 2.0) * decay_time
    remaining_time = max(0, remaining_time - decay_time)
    
    # Sustain contribution: constant
    sustain_contrib = sustain * remaining_time
    
    # Average amplitude
    if total_time > 0:
        avg_amplitude = (attack_contrib + decay_contrib + sustain_contrib) / total_time
    else:
        avg_amplitude = 0.5  # Default for very short notes
    
    return avg_amplitude
