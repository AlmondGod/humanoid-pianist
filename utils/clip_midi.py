from mido import MidiFile, MidiTrack, Message, MetaMessage
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_midi_path", type=str, required=True, help="Path to the input MIDI file")
    parser.add_argument("--output_midi_path", type=str, required=True, help="Path to the output MIDI file")
    parser.add_argument("--start_time", type=float, default=0, help="Clip start time in seconds")
    parser.add_argument("--duration", type=float, default=30, help="Clip duration in seconds")
    return parser.parse_args()

def bpm2tempo(bpm):
    """Convert beats per minute (BPM) to microseconds per beat."""
    return int(60 * 1000000 / bpm)

def get_tempo(midi_file):
    """Get the tempo from the MIDI file. Returns default 120 BPM if not found."""
    for track in midi_file.tracks:
        for msg in track:
            if isinstance(msg, MetaMessage) and msg.type == 'tempo':
                return msg.tempo
    return bpm2tempo(120)  # Default 120 BPM

def ticks_to_seconds(ticks, tempo, ticks_per_beat):
    """Convert ticks to seconds based on tempo and ticks per beat."""
    return (ticks * tempo) / (ticks_per_beat * 1000000)

def clip_midi(input_file, output_file, start_seconds, max_seconds):
    """
    Clip a MIDI file to a specified length in seconds, starting from a specific time.
    
    Args:
        input_file (str): Path to input MIDI file
        output_file (str): Path to save the clipped MIDI file
        start_seconds (float): Start time in seconds
        max_seconds (float): Length of clip in seconds after start_seconds
    """
    # Read the input MIDI file
    midi_in = MidiFile(input_file)
    
    # Create a new MIDI file with the same ticks_per_beat
    midi_out = MidiFile(ticks_per_beat=midi_in.ticks_per_beat)
    
    # Get the tempo (microseconds per beat)
    tempo = get_tempo(midi_in)
    
    # Process each track
    for track_in in midi_in.tracks:
        track_out = MidiTrack()
        midi_out.tracks.append(track_out)
        
        absolute_time = 0.0  # Track absolute time in seconds
        accumulated_ticks = 0  # Track accumulated ticks
        active_notes = {}  # Dictionary to track active notes {(note, channel): start_time}
        
        # Copy time signature and tempo messages from the start
        for msg in track_in:
            if isinstance(msg, MetaMessage) and msg.type in ['time_signature', 'tempo', 'key_signature']:
                track_out.append(msg)
                if msg.type == 'tempo':
                    tempo = msg.tempo
        
        # Process messages within our time window
        for msg in track_in:
            if hasattr(msg, 'time'):
                accumulated_ticks += msg.time
                absolute_time = ticks_to_seconds(accumulated_ticks, tempo, midi_in.ticks_per_beat)
            
            if isinstance(msg, Message):
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[(msg.note, msg.channel)] = absolute_time
                elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                    if (msg.note, msg.channel) in active_notes:
                        del active_notes[(msg.note, msg.channel)]
            
            if absolute_time > start_seconds:
                break
        
        # Reset for second pass
        absolute_time = 0.0
        accumulated_ticks = 0
        last_time = 0
        first_msg = True
        
        # Add still-active notes at start point
        for (note, channel), note_start in active_notes.items():
            if note_start < start_seconds:
                msg = Message('note_on', note=note, velocity=64, time=0, channel=channel)
                track_out.append(msg)
        
        # Second pass: process messages within our time window
        for msg in track_in:
            if hasattr(msg, 'time'):
                accumulated_ticks += msg.time
                absolute_time = ticks_to_seconds(accumulated_ticks, tempo, midi_in.ticks_per_beat)
            
            # Update tempo if we encounter a tempo change
            if isinstance(msg, MetaMessage) and msg.type == 'tempo':
                tempo = msg.tempo
            
            # Only process messages within our time window
            if start_seconds <= absolute_time <= (start_seconds + max_seconds):
                msg_copy = copy.deepcopy(msg)
                
                if first_msg and hasattr(msg, 'time'):
                    # Adjust the first message's timing
                    delta_time = absolute_time - start_seconds
                    adjusted_ticks = int(delta_time * 1000000 * midi_in.ticks_per_beat / tempo)
                    msg_copy.time = adjusted_ticks
                    first_msg = False
                
                track_out.append(msg_copy)
                last_time = absolute_time
            
            # If we're past the end time, handle active notes
            elif absolute_time > (start_seconds + max_seconds):
                if isinstance(msg, Message) and msg.type == 'note_off':
                    # Include note-off messages for notes that were active
                    if (msg.note, msg.channel) in active_notes:
                        msg_copy = copy.deepcopy(msg)
                        msg_copy.time = 0
                        track_out.append(msg_copy)
                break
        
        # Add end of track message
        track_out.append(MetaMessage('end_of_track'))
    
    # Save the output file
    midi_out.save(output_file)

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    input_midi, output_midi, start_time, duration = (
        args.input_midi_path, args.output_midi_path, args.start_time, args.duration
    )
    clip_midi(input_midi, output_midi, start_time, duration)