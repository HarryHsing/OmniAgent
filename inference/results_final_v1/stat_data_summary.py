#!/usr/bin/env python3
"""
Compute summary statistics for training data:
- Total trajectory count
- Total turn count (split into valid and invalid)
- Total frames count (valid actions only)
- Total audio time (valid actions only)
- Total clip time (valid actions only)
"""

import os
import json
from collections import defaultdict

def stat_file(filepath):
    """Compute statistics for a single file"""
    traj_ids = set()
    valid_turns = 0
    invalid_turns = 0
    total_frames = 0
    total_audio_time = 0.0
    total_clip_time = 0.0
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            traj_ids.add(data['traj_id'])
            
            is_valid = data['extra_info']['is_action_valid'] == 'True'
            if is_valid:
                valid_turns += 1
                # Parse action
                try:
                    output = json.loads(data['output'])
                    action = output.get('action', {})
                    action_type = action.get('type', '')
                    
                    if action_type == 'get_frames':
                        total_frames += action.get('num', 0)
                    elif action_type == 'get_audio':
                        start = action.get('start', 0)
                        end = action.get('end', 0)
                        total_audio_time += (end - start)
                    elif action_type == 'get_clip':
                        start = action.get('start', 0)
                        end = action.get('end', 0)
                        total_clip_time += (end - start)
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                invalid_turns += 1
    
    return {
        'traj_count': len(traj_ids),
        'valid_turns': valid_turns,
        'invalid_turns': invalid_turns,
        'total_frames': total_frames,
        'total_audio_time': total_audio_time,
        'total_clip_time': total_clip_time,
    }


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    # Get all jsonl files
    jsonl_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jsonl')])
    
    # Aggregate statistics
    total_stats = {
        'traj_ids': set(),
        'valid_turns': 0,
        'invalid_turns': 0,
        'total_frames': 0,
        'total_audio_time': 0.0,
        'total_clip_time': 0.0,
    }
    
    per_file_stats = {}
    
    for filename in jsonl_files:
        filepath = os.path.join(dir_path, filename)
        stats = stat_file(filepath)
        per_file_stats[filename] = stats
        
        # Accumulate totals (traj_ids need separate handling)
        # Note: traj_ids may overlap across files, so collect all and deduplicate later
        print(f"Processing: {filename}")
    
    # Re-traverse to collect all traj_ids (avoid duplicates)
    all_traj_ids = set()
    for filename in jsonl_files:
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_traj_ids.add(data['traj_id'])
    
    # Compute totals
    for filename, stats in per_file_stats.items():
        total_stats['valid_turns'] += stats['valid_turns']
        total_stats['invalid_turns'] += stats['invalid_turns']
        total_stats['total_frames'] += stats['total_frames']
        total_stats['total_audio_time'] += stats['total_audio_time']
        total_stats['total_clip_time'] += stats['total_clip_time']
    
    total_stats['traj_count'] = len(all_traj_ids)
    
    # Print results
    print("\n" + "="*80)
    print("Per-file Statistics:")
    print("="*80)
    print(f"{'File Name':<80} {'traj':>8} {'valid':>8} {'invalid':>8} {'frames':>10} {'audio(s)':>12} {'clip(s)':>12}")
    print("-"*80)
    
    for filename in jsonl_files:
        stats = per_file_stats[filename]
        print(f"{filename:<80} {stats['traj_count']:>8} {stats['valid_turns']:>8} {stats['invalid_turns']:>8} {stats['total_frames']:>10} {stats['total_audio_time']:>12.1f} {stats['total_clip_time']:>12.1f}")
    
    print("="*80)
    print(f"{'Total':<80} {total_stats['traj_count']:>8} {total_stats['valid_turns']:>8} {total_stats['invalid_turns']:>8} {total_stats['total_frames']:>10} {total_stats['total_audio_time']:>12.1f} {total_stats['total_clip_time']:>12.1f}")
    print("="*80)
    
    # Print summary
    print("\nAggregate Statistics:")
    print(f"  - Total trajectories: {total_stats['traj_count']:,}")
    print(f"  - Total turns: {total_stats['valid_turns'] + total_stats['invalid_turns']:,}")
    print(f"    - valid turns: {total_stats['valid_turns']:,}")
    print(f"    - invalid turns: {total_stats['invalid_turns']:,}")
    print(f"  - Total frames: {total_stats['total_frames']:,}")
    print(f"  - Total audio time: {total_stats['total_audio_time']:,.1f} sec ({total_stats['total_audio_time']/3600:.2f} hours)")
    print(f"  - Total clip time: {total_stats['total_clip_time']:,.1f} sec ({total_stats['total_clip_time']/3600:.2f} hours)")


if __name__ == '__main__':
    main()