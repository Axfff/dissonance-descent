Experiment: Continuous Timbre Optimization (Whole Song)

1. The Philosophy

Instead of optimizing for static block chords, we will optimize the instrument for the continuous harmonic flow of the music. This ensures the timbre handles dissonance, passing tones, and suspensions gracefully, creating an instrument evolved specifically for the entire performance of the piece.

2. The Musical Target: Bach Prelude in C (MIDI)

We will use a MIDI representation of J.S. Bach's Prelude in C Major (BWV 846).

Why MIDI? MIDI allows us to determine exactly which notes are overlapping at any micro-second.

The Structure: The prelude consists of arpeggios where notes overlap significantly. The dissonance occurs not just between simultaneous strikes, but between a new note and the decaying tail of previous notes.

3. The Efficient "Vertical Slice" Algorithm

To optimize the whole song efficiently, we cannot use a fixed time-step (which is slow). We will use Event-Based Slicing.

The Process:

Flatten MIDI: Convert MIDI "Note On/Off" messages into a timeline of events.

Generate Slices: Divide the song into segments where the set of active pitches is constant.

Slice A (0.0s to 0.25s): Active Notes = {C4, E4}

Slice B (0.25s to 0.50s): Active Notes = {C4, E4, G4}

Weighted Cost: The cost of a slice is its Dissonance * Duration.

4. The Objective Function (Global Dissonance)

We define the Global Dissonance ($D_{global}$) as the time-integral of roughness over the entire piece:

$$D_{global} = \sum_{i=0}^{N} ( D(S_i, \text{Timbre}) \times \text{Duration}_i )$$

$S_i$: The set of active notes in Slice $i$.

$D(...)$: The Plomp-Levelt roughness score for that set of notes given the current Timbre.

$\text{Duration}_i$: How long that specific harmonic clash lasts.

Note: This naturally prioritizes sustained dissonances. A passing dissonant note that lasts 0.1s matters less than a dissonant chord held for 2.0s.

5. The Instrument (The Variable)

We optimize the Overtone Series of the synthesizer.

Base: 6 Partial Synthesizer.

Parameters: [p2_ratio, p3_ratio, p4_ratio, p5_ratio, p6_ratio]

Constraints:

Partials must remain ordered (p2 < p3 < p4...).

Bounds: p2 between 1.5 and 2.5, p3 between 2.5 and 3.5, etc. (To prevent the optimizer from just collapsing everything to Unison/Octaves immediately).

6. Implementation Steps for the AI

Step 1: MIDI Parsing Engine

Write a function parse_midi_to_slices(midi_file) that returns a list of tuples: [(duration, [freq1, freq2, ...]), ...].

Simulate sustain: If working from a raw score, assume a small overlap (legato) to ensure we catch harmonies.

Step 2: Vectorized Cost Function

Write calculate_song_dissonance(timbre_params, slices).

Optimization Tip: Pre-calculate the note frequencies for every slice. Inside the optimization loop, only apply the timbre_params (multipliers) to those pre-calculated fundamentals.

Step 3: Optimization Loop

Use scipy.optimize.minimize (Method: 'Nelder-Mead' or 'L-BFGS-B').

Input: Randomly initialized partial ratios.

Target: Minimize D_{global}.

Step 4: "Harmonicity Map" Visualization (Optional but Cool)

Once optimized, plot the Dissonance vs. Time for the whole song.

Compare the "Standard Harmonic Timbre" plot vs. the "Optimized Timbre" plot.

Expectation: The Optimized Timbre should show a lower "average" dissonance line, smoothing out the spikes where Bach uses tension.

Step 5: Render Result

Synthesize the MIDI file using the new "Bach-Optimized Instrument" to a .wav file.