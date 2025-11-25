import mido
import numpy as np

def parse_midi_to_slices(midi_file_path):
    """
    Parses a MIDI file into a list of time slices, where each slice
    contains the duration and the set of active notes (frequencies).
    
    Args:
        midi_file_path (str): Path to the MIDI file.
        
    Returns:
        list of tuples: [(duration, [freq1, freq2, ...]), ...]
    """
    mid = mido.MidiFile(midi_file_path)
    
    # Convert MIDI to a list of absolute-time events
    # Event: (time, type, note, velocity)
    events = []
    current_time = 0.0
    
    events = []
    current_time = 0.0
    
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' or msg.type == 'note_off':
            velocity = msg.velocity if msg.type == 'note_on' else 0
            msg_type = 'note_on' if velocity > 0 else 'note_off'
            events.append({
                'time': current_time,
                'type': msg_type,
                'note': msg.note,
                'velocity': velocity,
                'channel': msg.channel
            })
        elif msg.type == 'control_change' and msg.control == 64:
            events.append({
                'time': current_time,
                'type': 'pedal',
                'value': msg.value
            })
            
    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    slices = []
    held_notes = {} # (channel, note) -> velocity
    sustaining_notes = {} # (channel, note) -> velocity (notes released but held by pedal)
    pedal_down = False
    
    last_time = 0.0
    
    for event in events:
        time = event['time']
        if time > last_time:
            duration = time - last_time
            
            # Combine held and sustaining notes
            active_map = held_notes.copy()
            active_map.update(sustaining_notes)
            
            if active_map:
                slice_content = []
                for (chan, note), vel in active_map.items():
                    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
                    amp = vel / 127.0
                    slice_content.append((freq, amp))
                
                # Sort by frequency
                slice_content.sort(key=lambda x: x[0])
                slices.append((duration, slice_content))
            else:
                # Silence
                slices.append((duration, []))
            
        if event['type'] == 'note_on':
            held_notes[(event['channel'], event['note'])] = event['velocity']
            # If it was sustaining, it's now re-struck (held)
            if (event['channel'], event['note']) in sustaining_notes:
                del sustaining_notes[(event['channel'], event['note'])]
                
        elif event['type'] == 'note_off':
            key = (event['channel'], event['note'])
            if key in held_notes:
                vel = held_notes[key]
                del held_notes[key]
                if pedal_down:
                    sustaining_notes[key] = vel
            elif key in sustaining_notes:
                # Already sustaining, ignore
                pass
                
        elif event['type'] == 'pedal':
            if event['value'] >= 64:
                pedal_down = True
            else:
                pedal_down = False
                # Release all sustaining notes
                sustaining_notes.clear()
                
        last_time = time
        
    return slices

def parse_midi_to_notes(midi_file_path):
    """
    Parses a MIDI file into a list of notes.
    
    Args:
        midi_file_path (str): Path to the MIDI file.
        
    Returns:
        list of tuples: [(start_time, duration, frequency, amplitude), ...]
    """
    mid = mido.MidiFile(midi_file_path)
    
    events = []
    current_time = 0.0
    
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' or msg.type == 'note_off':
            velocity = msg.velocity if msg.type == 'note_on' else 0
            msg_type = 'note_on' if velocity > 0 else 'note_off'
            events.append({
                'time': current_time,
                'type': msg_type,
                'note': msg.note,
                'velocity': velocity,
                'channel': msg.channel
            })
        elif msg.type == 'control_change' and msg.control == 64:
            events.append({
                'time': current_time,
                'type': 'pedal',
                'value': msg.value
            })
            
    events.sort(key=lambda x: x['time'])
    
    notes = []
    # (channel, note) -> {'start_time': t, 'velocity': v}
    active_notes = {} 
    # (channel, note) -> {'start_time': t, 'velocity': v, 'release_time': t}
    sustaining_notes = {} 
    pedal_down = False
    
    for event in events:
        time = event['time']
        
        if event['type'] == 'note_on':
            key = (event['channel'], event['note'])
            # If note is already playing (shouldn't happen often in clean MIDI but possible), end it
            if key in active_notes:
                start = active_notes[key]['start_time']
                vel = active_notes[key]['velocity']
                duration = time - start
                if duration > 0:
                    freq = 440.0 * (2.0 ** ((key[1] - 69) / 12.0))
                    amp = vel / 127.0
                    notes.append((start, duration, freq, amp))
                del active_notes[key]
            
            # If it was sustaining, end it (re-struck)
            if key in sustaining_notes:
                start = sustaining_notes[key]['start_time']
                vel = sustaining_notes[key]['velocity']
                # Duration is from start to NOW (re-strike)
                duration = time - start
                if duration > 0:
                    freq = 440.0 * (2.0 ** ((key[1] - 69) / 12.0))
                    amp = vel / 127.0
                    notes.append((start, duration, freq, amp))
                del sustaining_notes[key]
                
            active_notes[key] = {'start_time': time, 'velocity': event['velocity']}
            
        elif event['type'] == 'note_off':
            key = (event['channel'], event['note'])
            if key in active_notes:
                note_data = active_notes[key]
                del active_notes[key]
                
                if pedal_down:
                    # Move to sustaining
                    sustaining_notes[key] = note_data
                else:
                    # End note
                    start = note_data['start_time']
                    vel = note_data['velocity']
                    duration = time - start
                    if duration > 0:
                        freq = 440.0 * (2.0 ** ((key[1] - 69) / 12.0))
                        amp = vel / 127.0
                        notes.append((start, duration, freq, amp))
                        
        elif event['type'] == 'pedal':
            if event['value'] >= 64:
                pedal_down = True
            else:
                pedal_down = False
                # End all sustaining notes
                for key, note_data in sustaining_notes.items():
                    start = note_data['start_time']
                    vel = note_data['velocity']
                    duration = time - start
                    if duration > 0:
                        freq = 440.0 * (2.0 ** ((key[1] - 69) / 12.0))
                        amp = vel / 127.0
                        notes.append((start, duration, freq, amp))
                sustaining_notes.clear()
                
    return notes
