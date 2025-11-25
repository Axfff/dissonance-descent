import os
import json
import mido
import numpy as np
import soundfile as sf
from src.midi_parser import parse_midi_to_slices, parse_midi_to_notes
from src.optimizer import optimize_timbre
from src.visualizer import plot_harmonicity_map
from src.synthesizer import generate_tone
from visualize_landscape import plot_landscape_comparison

def generate_bach_prelude_midi(filename='bach_prelude.mid'):
    """
    Generates a simplified MIDI file of the first few bars of Bach's Prelude in C.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Tempo: 120 BPM -> 500000 us/beat. 480 ticks/beat.
    # 16th notes. Duration = 120 ticks.
    
    # Bar 1: C E G C E G C E ...
    # Notes: C4(60), E4(64), G4(67), C5(72), E5(76)
    # Pattern: C E G C5 E5 G4 C5 E5
    
    notes = [60, 64, 67, 72, 76, 67, 72, 76] * 2 # Repeat twice for the measure
    
    # Let's make them overlap slightly to create harmony (simulating reverb/legato)
    # Note On at t=0, Note Off at t=1.5 * step
    
    step = 120
    duration = int(step * 1.5) # Overlap by 50%
    
    current_time = 0
    
    # We need to manage note_offs carefully if we want overlap in a single track
    # Mido uses delta times.
    # Easier way: Just sequence them as block chords for simplicity? 
    # No, the design says "arpeggios where notes overlap significantly".
    
    # Let's just do simple arpeggio with long release.
    # Actually, for the parser to catch "active notes", we just need Note On before previous Note Off.
    
    # Event list approach for generation
    events = []
    
    for i, note in enumerate(notes):
        start_time = i * step
        end_time = start_time + duration
        events.append(('on', start_time, note))
        events.append(('off', end_time, note))
        
    # Sort by time
    events.sort(key=lambda x: x[1])
    
    last_time = 0
    for type_, time, note in events:
        delta = time - last_time
        velocity = 64 if type_ == 'on' else 0
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
        last_time = time
        
    mid.save(filename)
    print(f"Generated {filename}")

def render_audio(slices, fixed_ratios, amplitudes, output_file='output.wav'):
    """
    Renders the audio for the given slices and timbre.
    """
    print(f"Rendering audio to {output_file}...")
    
    # Parameters
    fs = 44100
    
    # Ratios and Amplitudes (including fundamental)
    ratios = [1.0] + list(fixed_ratios)
    amps = [1.0] + list(amplitudes)
    
    # Calculate total duration
    total_duration = sum(s[0] for s in slices)
    
    # Create buffer
    audio = np.zeros(int(fs * (total_duration + 1.0))) # +1s for tail
    
    current_sample = 0
    
    for duration, fundamentals in slices:
        if not fundamentals:
            current_sample += int(duration * fs)
            continue
            
        # Generate sound for this slice
        # Note: This is a naive rendering (concatenating slices). 
        # Real synthesis should handle ADSR and overlapping notes properly.
        # But for "hearing the timbre", this might be enough if we crossfade?
        # Actually, the slices are "state of the world". 
        # If we just render the active notes for the duration of the slice, it should be accurate to the "state".
        
        num_samples = int(duration * fs)
        
        # Smooth transition window (10ms) to avoid clicks between slices
        # window = np.ones(num_samples)
        # fade_len = int(0.01 * fs)
        # if num_samples > 2 * fade_len:
        #     window[:fade_len] = np.linspace(0, 1, fade_len)
        #     window[-fade_len:] = np.linspace(1, 0, fade_len)
            
        slice_mix = np.zeros(num_samples)
        
        for f0, amp0 in fundamentals:
            # Generate complex tone
            # We use the synthesizer module's generate_tone but we need to sum partials manually 
            # or pass arrays if supported. generate_tone supports arrays.
            
            # Construct partials
            p_freqs = [f0 * r for r in ratios]
            p_amps = [a * amp0 for a in amps] # Scale by note velocity
            
            tone = generate_tone(p_freqs, p_amps, duration)
            
            # Truncate or pad if necessary (generate_tone might be slightly off due to rounding)
            if len(tone) > num_samples:
                tone = tone[:num_samples]
            elif len(tone) < num_samples:
                tone = np.pad(tone, (0, num_samples - len(tone)))
                
            slice_mix += tone
            
        # Add to main buffer
        # We add it to the current position. 
        # Since slices are contiguous in time, we just append?
        # Yes, slices partition the timeline.
        
        # Apply simple envelope to slice to avoid clicks? 
        # Ideally we want continuous phase but that's hard with this slice approach.
        # Let's just do a tiny crossfade or just accept clicks for the experiment prototype.
        # Better: Apply a tiny fade in/out to the slice mix.
        fade_len = min(100, num_samples // 2)
        slice_mix[:fade_len] *= np.linspace(0, 1, fade_len)
        slice_mix[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        audio[current_sample:current_sample+num_samples] += slice_mix
        current_sample += num_samples
        
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        
    sf.write(output_file, audio, fs)
    sf.write(output_file, audio, fs)
    print("Rendering complete.")

def render_audio_from_notes(notes, fixed_ratios, amplitudes, output_file='output.wav', 
                             envelope_params=None, brightness_decay=1.0, phase_mode='zero'):
    """
    Renders the audio for the given notes and timbre with per-partial envelopes.
    
    Args:
        notes: List of (start_time, duration, freq, amp) tuples
        fixed_ratios: List of partial ratios (excluding fundamental)
        amplitudes: List of partial amplitudes (excluding fundamental)
        output_file: Output WAV filename
        envelope_params: List of envelope dicts for each partial (including fundamental)
        brightness_decay: Brightness decay factor for higher partials
        phase_mode: Phase mode ('zero', 'random', 'alternating')
    """
    from src.synthesizer import generate_tone_with_envelopes, apply_brightness_decay
    
    print(f"Rendering audio from notes to {output_file}...")
    
    fs = 44100
    
    # Ratios and Amplitudes (including fundamental)
    ratios = [1.0] + list(fixed_ratios)
    amps = [1.0] + list(amplitudes)
    
    # Calculate total duration
    if not notes:
        print("No notes to render.")
        return
        
    last_note_end = max(n[0] + n[1] for n in notes)
    
    # Add extra time for release tail
    if envelope_params:
        max_release = max(env.get('release', 0.2) for env in envelope_params)
        total_duration = last_note_end + max_release + 1.0
    else:
        total_duration = last_note_end + 2.0
    
    audio = np.zeros(int(fs * total_duration))
    
    for start_time, duration, f0, amp0 in notes:
        # Construct partials
        p_freqs = [f0 * r for r in ratios]
        p_amps = [a * amp0 for a in amps]
        
        # Prepare envelopes for each partial
        if envelope_params:
            # Apply brightness decay to each partial's envelope
            p_envelopes = []
            for i, env in enumerate(envelope_params):
                modified_env = apply_brightness_decay(env, i, brightness_decay)
                p_envelopes.append(modified_env)
            
            # Generate tone with per-partial envelopes
            tone = generate_tone_with_envelopes(
                p_freqs, p_amps, p_envelopes, duration, 
                phase_mode=phase_mode, sample_rate=fs
            )
        else:
            # Fallback to old method with uniform envelope
            from src.synthesizer import generate_tone, generate_adsr_envelope
            
            release_time = 0.2
            full_duration = duration + release_time
            tone = generate_tone(p_freqs, p_amps, full_duration, sample_rate=fs)
            
            # Apply uniform envelope
            envelope = generate_adsr_envelope(duration, 0.02, 0.1, 0.8, release_time, fs)
            if len(envelope) > len(tone):
                envelope = envelope[:len(tone)]
            elif len(envelope) < len(tone):
                envelope = np.pad(envelope, (0, len(tone) - len(envelope)))
            tone *= envelope
        
        # Add to main buffer
        start_sample = int(start_time * fs)
        end_sample = start_sample + len(tone)
        
        if end_sample > len(audio):
            # Extend buffer if needed
            padding = np.zeros(end_sample - len(audio))
            audio = np.concatenate((audio, padding))
            
        audio[start_sample:end_sample] += tone
        
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        
    sf.write(output_file, audio, fs)
    print("Rendering complete.")

def main():
    # 1. Generate/Load MIDI
    midi_file = 'bach_prelude.mid'
    if not os.path.exists(midi_file):
        generate_bach_prelude_midi(midi_file)
        
    # 2. Parse MIDI
    print("Parsing MIDI...")
    slices = parse_midi_to_slices(midi_file)
    print(f"Generated {len(slices)} time slices.")
    
    # 3. Load Configuration
    print("\n--- Loading Configuration ---")
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Extract timbre parameters
    partials = config['timbre']['partials']
    brightness_decay = config['timbre'].get('brightness_decay', 1.0)
    phase_mode = config['timbre'].get('phase_mode', 'zero')
    
    # Extract ratios, amplitudes, and envelopes
    fixed_ratios = [p['ratio'] for p in partials[1:]]
    initial_amplitudes = [p['amplitude'] for p in partials[1:]]
    envelope_params = [p.get('envelope', {
        'attack': 0.02, 'decay': 0.1, 'sustain': 0.8, 'release': 0.2
    }) for p in partials]
    
    print(f"Loaded {len(partials)} partials")
    print(f"Brightness decay: {brightness_decay}")
    print(f"Phase mode: {phase_mode}")
    
    # 4. Optimization
    print("\n--- Starting Optimization ---")
    
    result = optimize_timbre(slices, fixed_ratios, initial_amplitudes)
    optimized_amplitudes = result.x
    
    print("\n--- Results ---")
    np.set_printoptions(precision=4, suppress=True)
    print(f"Fixed Ratios: {fixed_ratios}")
    print(f"Standard Amplitudes: {np.array(initial_amplitudes)}")
    print(f"Optimized Amplitudes: {optimized_amplitudes}")
    
    # 5. Visualization
    print("\nGenerating Visualization...")
    plot_harmonicity_map(slices, fixed_ratios, initial_amplitudes, optimized_amplitudes, 'harmonicity_map.png')
    plot_landscape_comparison(fixed_ratios, initial_amplitudes, optimized_amplitudes, 'dissonance_landscape.png')
    
    # 6. Rendering with Enhanced Timbre
    print("\nRendering Audio with Enhanced Timbre...")
    
    # Parse notes for rendering
    notes = parse_midi_to_notes(midi_file)
    print(f"Parsed {len(notes)} notes for rendering.")
    
    # Render with envelope parameters
    render_audio_from_notes(
        notes, fixed_ratios, initial_amplitudes, 
        'output_standard.wav',
        envelope_params=envelope_params,
        brightness_decay=brightness_decay,
        phase_mode=phase_mode
    )
    
    render_audio_from_notes(
        notes, fixed_ratios, optimized_amplitudes, 
        'output_optimized.wav',
        envelope_params=envelope_params,
        brightness_decay=brightness_decay,
        phase_mode=phase_mode
    )
    
    print("\nExperiment Complete!")

if __name__ == '__main__':
    main()
