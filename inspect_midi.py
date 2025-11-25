import mido

def inspect_midi(filename):
    mid = mido.MidiFile(filename)
    print(f"Filename: {filename}")
    print(f"Type: {mid.type}")
    print(f"Ticks per beat: {mid.ticks_per_beat}")
    print(f"Number of tracks: {len(mid.tracks)}")
    
    for i, track in enumerate(mid.tracks):
        print(f"Track {i}: {track.name} (len: {len(track)})")
        
    print("\nFirst 50 merged events:")
    count = 0
    for msg in mid:
        print(msg)
        count += 1
        if count >= 50:
            break

if __name__ == "__main__":
    inspect_midi("bach_prelude.mid")
