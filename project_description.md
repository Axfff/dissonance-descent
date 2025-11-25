Project Title: Adaptive Harmony Engine

Physics-Based Timbre and Scale Optimization System

1. Executive Summary

This project aims to build a computational engine that simulates the relationship between musical timbre (spectral content) and harmony (consonance/dissonance). Based on the psychoacoustic research of Plomp, Levelt, and William Sethares (and popularized by minutephysics), the system will use numerical optimization (Gradient Descent) to automatically generate "perfect" scales for arbitrary instruments, or "perfect" instruments for arbitrary scales.

2. Theoretical Background & Context

Traditional music theory assumes "in-tune" notes are fixed universal ratios (e.g., Perfect Fifth = 3:2). However, psychoacoustic physics demonstrates that consonance is not an inherent property of intervals, but a result of the interference between the overtones (harmonics) of specific sounds.

The Plomp-Levelt Curve: A model describing the "roughness" (dissonance) between two pure sine waves based on their frequency difference.

Sethares' Extension: The dissonance of complex tones (like a violin or a gong) is the sum of the roughness of every pair of overtones present in the sound.

The Hypothesis:

If you change the instrument (timbre), the points of minimum dissonance (valleys) shift, creating new "natural" scales.

If you change the scale, you can shift the instrument's partials to eliminate "beating," creating a "harmonic timbre" for that specific music.

3. Project Goals

Core Objective:

Develop a Python-based system that treats "Harmoniousness" as a cost function to be minimized via optimization algorithms.

Sub-Goals:

Dissonance Calculator (The Cost Function): Implement a vectorized function that accepts a set of active frequencies (fundamentals + overtones) and returns a single scalar float representing total "Perceptual Roughness."

Mode A: Scale/Composition Optimization:

Input: A fixed Timbre (e.g., a "Gamelan" with non-integer harmonics) and a rough musical idea (random notes or standard 12-TET).

Process: Freeze timbre spectra. Use Gradient Descent to adjust the fundamental frequency (pitch) of every note in the song.

Output: A retuned song/scale that sounds mathematically "pure" on that specific instrument.

Mode B: Timbre Optimization:

Input: A fixed Composition (e.g., a C Major chord) and a base timbre type.

Process: Freeze note pitches. Use Gradient Descent to adjust the relative frequencies and amplitudes of the instrument's overtones.

Output: A synthetic instrument designed specifically to play that song with zero auditory roughness.

4. Technical Specifications

A. Data Structures

Timbre Vector ($T$): A collection of partials defining the instrument.

Format: [(freq_ratio_1, amplitude_1), (freq_ratio_2, amplitude_2), ...]

Example (Ideal String): [(1.0, 1.0), (2.0, 0.5), (3.0, 0.33), ...]

Song/State Representation ($S$):

A list of active fundamental frequencies at time $t$: [f1, f2, f3...]

B. The Cost Function (Plomp-Levelt Model)

The system must implement the standard parameterized Plomp-Levelt formula:


$$d(f_1, f_2) = e^{-a \cdot \Delta f} - e^{-b \cdot \Delta f}$$


Where $\Delta f$ is the critical bandwidth difference. The total dissonance is the sum of $d(p_i, p_j)$ for all pairs of partials $p$ in the current chord.

C. Optimization Algorithm

Scipy Minimize: Use scipy.optimize.minimize (L-BFGS-B or Nelder-Mead) to handle the multi-dimensional optimization landscape.

Constraints: Pitch shifts should be constrained (e.g., a note shouldn't drift more than a semitone) to preserve the melody's identity.

D. Audio Synthesis (Output)

The system must include an Additive Synthesizer module to render the result.

It should take the optimized frequency vectors and generate a .wav file so the user can hear the "before" and "after" comparison.

5. Implementation Roadmap for AI

Setup: Initialize Python environment with numpy, scipy, and soundfile (or pyaudio).

Module 1: dissonance.py: Implement the Plomp-Levelt calculation between two complex tones.

Module 2: optimizer.py: Create the optimization loop that perturbs frequencies to lower the dissonance score.

Module 3: synthesizer.py: A simple engine to generate sine-wave sums based on the definitions.

Main Script: A workflow that defines a "Dissonant Chord" + "Standard Timbre," runs the optimizer, and saves the audio.

6. Success Criteria

The system can demonstrate the "Stretched Octave" phenomenon (tuning a scale to match stretched partials).

The system can take a dissonant cluster of notes and automatically "resolve" them into a consonant chord by moving their pitches.

7. Plomp-Levelt Model Parameters

The system uses the Sethares approximation of the Plomp-Levelt dissonance curve. The parameters in `config.json` control the shape of this curve and the model of the human ear's critical bandwidth.

### Critical Bandwidth Parameters (s1, s2)
These define the "Critical Bandwidth" — the frequency range within which two tones interfere to cause roughness.
Formula: `bandwidth = s1 * frequency + s2`

- **`s1` (0.021)**: The slope. Makes the critical bandwidth wider for higher frequencies (approx 2.1% of the frequency).
- **`s2` (19.0)**: The intercept. The base bandwidth offset.

### Curve Shape Parameters (a, b)
These define the shape of the dissonance curve for a normalized frequency difference `s`.
Formula: `dissonance = e^(-a * s) - e^(-b * s)`

- **`a` (3.5)**: Controls the **fall-off** rate. Determines how quickly dissonance fades as notes move apart.
- **`b` (5.75)**: Controls the **rise** rate. Determines how sharply dissonance spikes near unison.

The default values (3.5, 5.75, 0.021, 19.0) are derived from Sethares' fit to the original Plomp-Levelt experimental data.