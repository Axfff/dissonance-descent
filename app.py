import os
import json
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

from src.dissonance import calculate_total_dissonance
from src.synthesizer import generate_tone

app = Flask(__name__)

# Load default config to get default params
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calculate_dissonance', methods=['POST'])
def api_calculate_dissonance():
    data = request.json
    partials = data.get('partials', [])
    # partials is a list of {'ratio': float, 'amplitude': float}
    
    # We want to visualize the dissonance of this timbre against itself (or a sweep)
    # Usually, "dissonance curve" means: keep one tone fixed at f, sweep another tone f*alpha
    # and calculate dissonance vs alpha.
    
    # Let's implement the standard "Dissonance Curve" visualization:
    # Fixed tone at f=440 (with these partials)
    # Variable tone at f_var = 440 * alpha (with these partials)
    # Calculate dissonance between Fixed + Variable
    
    base_freq = 440.0
    timbre_model = [(p['ratio'], p['amplitude']) for p in partials]
    
    # Construct the fixed tone spectrum
    fixed_f = [base_freq * r for r, a in timbre_model]
    fixed_a = [a for r, a in timbre_model]
    
    # Sweep alpha from 1.0 to 2.2 (just over an octave)
    alphas = np.linspace(1.0, 2.2, 200)
    dissonance_values = []
    
    plomp_params = load_config().get('plomp_levelt', None)

    for alpha in alphas:
        # Construct variable tone spectrum
        var_f = [base_freq * alpha * r for r, a in timbre_model]
        var_a = [a for r, a in timbre_model]
        
        # Combine spectra
        all_f = fixed_f + var_f
        all_a = fixed_a + var_a
        
        diss = calculate_total_dissonance(all_f, all_a, model_params=plomp_params)
        dissonance_values.append(diss)
        
    return jsonify({
        'alphas': alphas.tolist(),
        'dissonance': dissonance_values
    })

@app.route('/api/generate_audio', methods=['POST'])
def api_generate_audio():
    data = request.json
    partials = data.get('partials', [])
    freqs = data.get('freqs', [440.0]) # List of frequencies to play
    duration = data.get('duration', 2.0)
    
    timbre_model = [(p['ratio'], p['amplitude']) for p in partials]
    
    mix = np.zeros(int(44100 * duration))
    for f0 in freqs:
        # Construct partials for this note
        note_f = [f0 * r for r, a in timbre_model]
        note_a = [a for r, a in timbre_model]
        mix += generate_tone(note_f, note_a, duration=duration)
        
    # Normalize
    if np.max(np.abs(mix)) > 0:
        mix /= np.max(np.abs(mix))
        
    # Write to buffer
    buffer = BytesIO()
    sf.write(buffer, mix, 44100, format='WAV')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
