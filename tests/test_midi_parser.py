import unittest
import mido
import os
from src.midi_parser import parse_midi_to_slices

class TestMidiParser(unittest.TestCase):
    def setUp(self):
        self.test_midi_path = 'test_scale.mid'
        # Create a simple MIDI file: C4 (0.5s) -> E4 (0.5s) -> G4 (0.5s)
        # Overlap them slightly to test polyphony? 
        # Let's do:
        # 0.0s: C4 on
        # 0.5s: E4 on
        # 1.0s: C4 off
        # 1.5s: E4 off
        
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Delta times in ticks. Default ticks_per_beat=480.
        # Let's assume 120 bpm -> 1 beat = 0.5s.
        # So 0.5s = 480 ticks.
        
        # Note: mido messages use delta time.
        
        # 0.0s: C4 on (note 60)
        track.append(mido.Message('note_on', note=60, velocity=64, time=0))
        
        # 0.5s: E4 on (note 64). Delta = 480
        track.append(mido.Message('note_on', note=64, velocity=64, time=480))
        
        # 1.0s: C4 off. Delta = 480 (0.5s after previous event)
        track.append(mido.Message('note_off', note=60, velocity=64, time=480))
        
        # 1.5s: E4 off. Delta = 480
        track.append(mido.Message('note_off', note=64, velocity=64, time=480))
        
        mid.save(self.test_midi_path)
        
    def tearDown(self):
        if os.path.exists(self.test_midi_path):
            os.remove(self.test_midi_path)
            
    def test_parse_midi(self):
        slices = parse_midi_to_slices(self.test_midi_path)
        
        # Expected slices:
        # 1. 0.0s - 0.5s: {C4} (60)
        # 2. 0.5s - 1.0s: {C4, E4} (60, 64)
        # 3. 1.0s - 1.5s: {E4} (64)
        
        self.assertEqual(len(slices), 3)
        
        # Slice 1
        dur1, freqs1 = slices[0]
        # Duration might be approx 0.5s depending on tempo default.
        # Mido default tempo is 500000 microseconds per beat (120bpm).
        # So 480 ticks = 1 beat = 0.5s.
        self.assertAlmostEqual(dur1, 0.5, places=2)
        self.assertEqual(len(freqs1), 1)
        self.assertAlmostEqual(freqs1[0], 261.63, places=1) # C4
        
        # Slice 2
        dur2, freqs2 = slices[1]
        self.assertAlmostEqual(dur2, 0.5, places=2)
        self.assertEqual(len(freqs2), 2)
        # C4 and E4
        self.assertAlmostEqual(freqs2[0], 261.63, places=1)
        self.assertAlmostEqual(freqs2[1], 329.63, places=1) # E4
        
        # Slice 3
        dur3, freqs3 = slices[2]
        self.assertAlmostEqual(dur3, 0.5, places=2)
        self.assertEqual(len(freqs3), 1)
        self.assertAlmostEqual(freqs3[0], 329.63, places=1)

if __name__ == '__main__':
    unittest.main()
