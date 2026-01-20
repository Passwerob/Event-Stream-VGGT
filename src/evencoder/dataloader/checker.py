#!/usr/bin/env python3
"""
Verify preprocessing completeness - check for missing files.
"""

from pathlib import Path
import argparse


def check_completeness(input_dir, output_dir, verbose=False):
    """
    Check if all input files were successfully processed.
    
    Args:
        input_dir: Input data directory
        output_dir: Output directory
        verbose: Print detailed information
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    total_sequences = 0
    total_issues = 0
    
    print(f"🔍 Checking completeness...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    for seq_dir in sorted(input_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        
        seq_name = seq_dir.name
        total_sequences += 1
        
        # Get input files
        in_imgs = sorted((seq_dir / "images").glob("*.png"))
        in_events = sorted((seq_dir / "events").glob("*.pt"))
        
        # Get output files
        out_imgs_dir = output_dir / seq_name / "images"
        out_events_dir = output_dir / seq_name / "events"
        
        if not out_imgs_dir.exists() or not out_events_dir.exists():
            print(f"❌ {seq_name}: Output directories missing!")
            total_issues += 1
            continue
        
        out_imgs = sorted(out_imgs_dir.glob("*.pt"))
        out_events = sorted(out_events_dir.glob("*.pt"))
        
        # Check file names match
        in_img_names = set(f.stem for f in in_imgs)
        out_img_names = set(f.stem for f in out_imgs)
        
        missing_imgs = in_img_names - out_img_names
        extra_imgs = out_img_names - in_img_names
        
        in_event_names = set(f.name for f in in_events)
        out_event_names = set(f.name for f in out_events)
        
        missing_events = in_event_names - out_event_names
        extra_events = out_event_names - in_event_names
        
        # Report issues
        has_issues = False
        
        if missing_imgs:
            print(f"❌ {seq_name}: Missing {len(missing_imgs)} images")
            if verbose:
                print(f"   Missing: {list(missing_imgs)[:5]}")
            has_issues = True
        
        if extra_imgs:
            print(f"⚠️  {seq_name}: {len(extra_imgs)} extra images in output")
            if verbose: 
                print(f"   Extra:  {list(extra_imgs)[:5]}")
            has_issues = True
        
        if missing_events:
            print(f"❌ {seq_name}: Missing {len(missing_events)} events")
            if verbose: 
                print(f"   Missing: {list(missing_events)[:5]}")
            has_issues = True
        
        if extra_events:
            print(f"⚠️  {seq_name}: {len(extra_events)} extra events in output")
            if verbose:
                print(f"   Extra: {list(extra_events)[:5]}")
            has_issues = True
        
        if has_issues:
            total_issues += 1
        elif verbose:
            print(f"✅ {seq_name}:  {len(out_imgs)} images, {len(out_events)} events OK")
    
    print()
    print(f"{'='*60}")
    print(f"Checked {total_sequences} sequences")
    
    if total_issues == 0:
        print(f"✅ All sequences are complete!")
    else:
        print(f"❌ Found issues in {total_issues} sequences")
        print(f"   Please re-run preprocessing for these sequences")
    
    return total_issues == 0


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Verify preprocessing completeness")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    success = check_completeness(args.input, args.output, args.verbose)
    exit(0 if success else 1)