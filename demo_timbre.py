"""
Demo script showing the enhanced timbre model capabilities.
This demonstrates per-partial ADSR envelopes, brightness decay, and phase control.
"""

import numpy as np
import json
from src.synthesizer import generate_tone_with_envelopes, apply_brightness_decay, save_wav

def demo_envelope_behavior():
    """
    Demonstrates the difference between uniform and per-partial envelopes.
    """
    print("Generating demonstration audio...")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    partials = config['timbre']['partials']
    brightness_decay = config['timbre']['brightness_decay']
    
    # Generate a test tone: A4 (440 Hz)
    base_freq = 440.0
    duration = 2.0
    
    # Extract parameters
    frequencies = [base_freq * p['ratio'] for p in partials]
    amplitudes = [p['amplitude'] for p in partials]
    envelope_params = [p['envelope'] for p in partials]
    
    # Demo 1: With per-partial envelopes and brightness decay
    print("Rendering with per-partial envelopes (brightness decay = 1.5)...")
    envelopes_with_decay = []
    for i, env in enumerate(envelope_params):
        modified_env = apply_brightness_decay(env, i, brightness_decay)
        envelopes_with_decay.append(modified_env)
    
    tone_enhanced = generate_tone_with_envelopes(
        frequencies, amplitudes, envelopes_with_decay, 
        duration, phase_mode='zero'
    )
    save_wav('demo_enhanced.wav', tone_enhanced)
    print("✓ Saved: demo_enhanced.wav (per-partial envelopes, brightness decay 1.5)")
    
    # Demo 2: Without brightness decay (all partials use their config envelopes)
    print("Rendering with per-partial envelopes (no brightness decay)...")
    tone_no_decay = generate_tone_with_envelopes(
        frequencies, amplitudes, envelope_params,
        duration, phase_mode='zero'
    )
    save_wav('demo_no_decay.wav', tone_no_decay)
    print("✓ Saved: demo_no_decay.wav (per-partial envelopes, no brightness decay)")
    
    # Demo 3: With uniform envelope (old method)
    print("Rendering with uniform envelope (comparison)...")
    from src.synthesizer import generate_tone, generate_adsr_envelope
    
    tone_uniform = generate_tone(frequencies, amplitudes, duration + 0.3, sample_rate=44100)
    envelope = generate_adsr_envelope(duration, 0.02, 0.1, 0.8, 0.3, sample_rate=44100)
    if len(envelope) > len(tone_uniform):
        envelope = envelope[:len(tone_uniform)]
    elif len(envelope) < len(tone_uniform):
        envelope = np.pad(envelope, (0, len(tone_uniform) - len(envelope)))
    tone_uniform *= envelope
    
    # Normalize
    max_val = np.max(np.abs(tone_uniform))
    if max_val > 0:
        tone_uniform /= max_val
    
    save_wav('demo_uniform.wav', tone_uniform)
    print("✓ Saved: demo_uniform.wav (uniform envelope, old method)")
    
    # Demo 4: Different phase modes
    print("Rendering with random phases...")
    tone_random = generate_tone_with_envelopes(
        frequencies, amplitudes, envelopes_with_decay,
        duration, phase_mode='random'
    )
    save_wav('demo_random_phase.wav', tone_random)
    print("✓ Saved: demo_random_phase.wav (random phases, softer attack)")
    
    print("\n=== Demo Complete ===")
    print("\nCompare the following files to hear the improvements:")
    print("1. demo_uniform.wav       - Old method: all partials have same envelope")
    print("2. demo_no_decay.wav      - Per-partial envelopes from config")
    print("3. demo_enhanced.wav      - Per-partial envelopes + brightness decay")
    print("4. demo_random_phase.wav  - Enhanced timbre with random phases")
    print("\nNotice how demo_enhanced.wav has:")
    print("  • More natural attack (high partials appear first)")
    print("  • Evolving brightness (high partials fade quickly)")
    print("  • Warmer, more realistic tone")

def show_envelope_details():
    """
    Prints detailed information about the configured envelopes.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    partials = config['timbre']['partials']
    brightness_decay = config['timbre']['brightness_decay']
    
    print("\n=== Envelope Configuration ===")
    print(f"Brightness Decay Factor: {brightness_decay}\n")
    
    print("Partial | Ratio | Amplitude | Attack | Decay  | Sustain | Release")
    print("--------|-------|-----------|--------|--------|---------|--------")
    
    for i, p in enumerate(partials):
        env = p['envelope']
        print(f"   {i}    | {p['ratio']:<5.1f} | {p['amplitude']:<9.2f} | "
              f"{env['attack']*1000:5.1f}ms | {env['decay']*1000:5.1f}ms | "
              f"{env['sustain']:<7.2f} | {env['release']*1000:5.1f}ms")
    
    print("\nWith Brightness Decay Applied:")
    print("Partial | Effective Sustain | Effective Release")
    print("--------|-------------------|------------------")
    
    from src.synthesizer import apply_brightness_decay
    for i, p in enumerate(partials):
        env = p['envelope']
        modified = apply_brightness_decay(env, i, brightness_decay)
        print(f"   {i}    | {modified['sustain']:17.2f} | {modified['release']*1000:15.1f}ms")

if __name__ == '__main__':
    show_envelope_details()
    demo_envelope_behavior()
