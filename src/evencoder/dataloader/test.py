#!/usr/bin/env python3
"""
Super fast event copying using system commands.
"""

import subprocess
from pathlib import Path
import argparse

def copy_events_fast(source_dir, target_dir, num_workers=32):
    """Use system commands for maximum speed."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print(f"Fast copying with {num_workers} parallel jobs...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    # Create bash script on the fly
    bash_script = f"""
#!/bin/bash
SOURCE="{source_dir}"
TARGET="{target_dir}"

# Function to copy one sequence
copy_seq() {{
    seq_name=$(basename "$1")
    if [ -d "$1/events" ]; then
        mkdir -p "$TARGET/$seq_name/events"
        cp -f "$1/events/"*.pt "$TARGET/$seq_name/events/" 2>/dev/null || true
        echo "✓ $seq_name"
    fi
}}

export -f copy_seq
export SOURCE TARGET

# Use find + xargs for parallel execution
find "$SOURCE" -maxdepth 1 -type d | tail -n +2 | \\
    xargs -P {num_workers} -I {{}} bash -c 'copy_seq "{{}}"'

echo "✅ Done!"
"""
    
    # Execute
    try:
        result = subprocess.run(
            bash_script,
            shell=True,
            executable='/bin/bash',
            check=True,
            text=True,
            capture_output=False
        )
        print("\n✅ Copying complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Fast event file copying")
    parser.add_argument("--source", type=str, default="/data/fcr/data/rgv_interval")
    parser.add_argument("--target", type=str, default="/data/fcr/data/rgv_interval_preprocessed")
    parser.add_argument("--workers", type=int, default=32)
    
    args = parser. parse_args()
    
    copy_events_fast(args. source, args.target, args. workers)