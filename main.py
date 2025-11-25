import numpy as np
import json
from src.dissonance import calculate_total_dissonance
from src.optimizer import optimize_scale
from src.synthesizer import generate_tone, save_wav

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("--- Adaptive Harmony Engine ---")
    config = load_config()
    timbre_config = config['timbre']
    plomp_params = config.get('plomp_levelt', None)
    
    # 1. Define Timbre
    partials = timbre_config['partials']
    timbre_model = [(p['ratio'], p['amplitude']) for p in partials]
    print(f"Timbre Model: {len(timbre_model)} partials")
    
    # 2. Define Initial Scale (Detuned Major Chord)
    # Ideal: 440, 550 (5/4), 660 (3/2)
    # Detuned: 440, 560, 650
    initial_freqs = [440.0, 560.0, 650.0]
    print(f"Initial Frequencies: {initial_freqs}")
    
    # 3. Calculate Initial Dissonance
    def get_full_spectrum(freqs, timbre):
        all_f = []
        all_a = []
        for f0 in freqs:
            for r, a in timbre:
                all_f.append(f0 * r)
                all_a.append(a)
        return all_f, all_a
        
    init_f, init_a = get_full_spectrum(initial_freqs, timbre_model)
    # Note: We need to update optimizer to pass params too, but for now let's just update main
    # Actually, optimizer calls calculate_total_dissonance internally. 
    # We need to update optimizer.py as well to accept params!
    init_diss = calculate_total_dissonance(init_f, init_a, model_params=plomp_params)
    print(f"Initial Dissonance: {init_diss:.4f}")
    
    # 4. Optimize
    print("\nOptimizing...")
    opt_freqs, opt_diss = optimize_scale(initial_freqs, timbre_model, bounds_semitones=2.0, model_params=plomp_params)
    
    print("\nOptimization Complete!")
    print(f"Optimized Frequencies: {opt_freqs}")
    print(f"Optimized Dissonance: {opt_diss:.4f}")
    
    # Check ratios relative to root
    root = opt_freqs[0]
    ratios = opt_freqs / root
    print(f"Ratios: {ratios}")
    
    # 5. Generate Audio
    print("\nGenerating Audio...")
    
    def render_chord(freqs, timbre, duration=2.0):
        mix = np.zeros(int(44100 * duration))
        for f0 in freqs:
            # Construct partials for this note
            note_f = [f0 * r for r, a in timbre]
            note_a = [a for r, a in timbre]
            mix += generate_tone(note_f, note_a, duration=duration)
        # Normalize
        if np.max(np.abs(mix)) > 0:
            mix /= np.max(np.abs(mix))
        return mix

    audio_initial = render_chord(initial_freqs, timbre_model)
    audio_optimized = render_chord(opt_freqs, timbre_model)
    
    save_wav("output_initial.wav", audio_initial)
    save_wav("output_optimized.wav", audio_optimized)
    
    print("Saved 'output_initial.wav' and 'output_optimized.wav'")

if __name__ == "__main__":
    main()
