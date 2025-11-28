"""
Generate complete outputs (visualizations, audio) for a single optimization trial.
"""
import os
import json

def generate_trial_outputs(result, restart_idx, slices, config, output_dir):
    """
    Generate all visualizations and audio for a single trial.
    
    Args:
        result: Optimization result dict
        restart_idx: Trial index
        slices: MIDI slices
        config: Configuration dict
        output_dir: Base output directory for this trial
        
    Returns:
        dict: Paths to generated files
    """
    # Import required modules from correct locations
    import sys
    import os
    # Add project root to path to import root-level modules
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from visualize_landscape import plot_landscape_comparison
    from src.visualizer import plot_harmonicity_map  
    from src.visualizer_enhanced import plot_frequency_migration, plot_adsr_comparison
    from src.synthesizer import render_notes_to_audio
    from src.midi_parser import parse_midi_to_notes
    
    print(f"\n  Generating outputs for restart {restart_idx}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    optimized_partials = result['active_partials']
    generated_files = {}
    
    # 1. Frequency migration plot
    try:
        freq_migration_path = os.path.join(output_dir, 'frequency_migration.png')
        plot_frequency_migration(
            config['timbre']['partials'],
            optimized_partials,
            freq_migration_path
        )
        generated_files['frequency_migration'] = freq_migration_path
        print(f"    ✓ Frequency migration → {freq_migration_path}")
    except Exception as e:
        print(f"    ✗ Frequency migration failed: {e}")
    
    # 2. ADSR comparison (if enabled)
    if config['optimization']['optimize_adsr']['enabled']:
        try:
            adsr_path = os.path.join(output_dir, 'adsr_comparison.png')
            plot_adsr_comparison(
                config['timbre']['partials'],
                optimized_partials,
                adsr_path
            )
            generated_files['adsr_comparison'] = adsr_path
            print(f"    ✓ ADSR comparison → {adsr_path}")
        except Exception as e:
            print(f"    ✗ ADSR comparison failed: {e}")
    
    # 3. Harmonicity map
    try:
        # Extract amplitudes
        initial_config_amps = [p['amplitude'] for p in config['timbre']['partials']]
        optimized_active_ratios = [p['ratio'] for p in optimized_partials]
        optimized_active_amps = [p['amplitude'] for p in optimized_partials]
        
        # Match frequencies
        initial_amps_matched = []
        config_ratios = [p['ratio'] for p in config['timbre']['partials']]
        
        for opt_ratio in optimized_active_ratios:
            if opt_ratio in config_ratios:
                idx = config_ratios.index(opt_ratio)
                initial_amps_matched.append(initial_config_amps[idx])
            else:
                initial_amps_matched.append(0.5)
        
        harmonicity_path = os.path.join(output_dir, 'harmonicity_map.png')
        plot_harmonicity_map(
            slices,
            optimized_active_ratios[1:],  # Exclude fundamental
            initial_amps_matched[1:],
            optimized_active_amps[1:],
            harmonicity_path
        )
        generated_files['harmonicity_map'] = harmonicity_path
        print(f"    ✓ Harmonicity map → {harmonicity_path}")
    except Exception as e:
        print(f"    ✗ Harmonicity map failed: {e}")
    
    # 4. Dissonance landscape (if not too many partials)
    if len(optimized_active_ratios) <= 20:
        try:
            landscape_path = os.path.join(output_dir, 'dissonance_landscape.png')
            plot_landscape_comparison(
                optimized_active_ratios[1:],
                initial_amps_matched[1:],
                optimized_active_amps[1:],
                landscape_path
            )
            generated_files['dissonance_landscape'] = landscape_path
            print(f"    ✓ Dissonance landscape → {landscape_path}")
        except Exception as e:
            print(f"    ✗ Dissonance landscape failed: {e}")
    else:
        print(f"    ⊘ Dissonance landscape skipped (too many partials: {len(optimized_active_ratios)})")
    
    # 5. Generate audio files
    try:
        # Parse MIDI to notes
        midi_file = config.get('midi_file', 'bach_prelude.mid')
        notes = parse_midi_to_notes(midi_file)
        
        # Standard timbre audio
        standard_path = os.path.join(output_dir, 'output_standard.wav')
        render_notes_to_audio(
            notes,
            config['timbre']['partials'],
            standard_path,
            config['timbre']['brightness_decay'],
            config['timbre']['phase_mode']
        )
        generated_files['audio_standard'] = standard_path
        print(f"    ✓ Standard audio → {standard_path}")
        
        # Optimized timbre audio
        optimized_path = os.path.join(output_dir, 'output_optimized.wav')
        render_notes_to_audio(
            notes,
            optimized_partials,
            optimized_path,
            config['timbre']['brightness_decay'],
            config['timbre']['phase_mode']
        )
        generated_files['audio_optimized'] = optimized_path
        print(f"    ✓ Optimized audio → {optimized_path}")
    except Exception as e:
        print(f"    ✗ Audio generation failed: {e}")
    
    # 6. Save trial metadata
    try:
        metadata = {
            'restart_idx': restart_idx,
            'strategy': result.get('_restart_strategy', 'unknown'),
            'cost': result['cost'],
            'success': result['success'],
            'active_partials': len(optimized_partials),
            'optimized_ratios': [p['ratio'] for p in optimized_partials],
            'optimized_amplitudes': [p['amplitude'] for p in optimized_partials],
            'generated_files': generated_files
        }
        
        metadata_path = os.path.join(output_dir, 'trial_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    ✓ Metadata → {metadata_path}")
    except Exception as e:
        print(f"    ✗ Metadata save failed: {e}")
    
    return generated_files
