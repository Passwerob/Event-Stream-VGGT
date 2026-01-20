#!/usr/bin/env python3
"""
Parallel preprocessing - FIXED VERSION
Uses multiprocessing with better progress tracking
"""

import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import argparse
import shutil
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time


def process_single_image(img_path, transform, out_images_dir):
    """Process a single image file with atomic write."""
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert('RGB')
            image_tensor = transform(img_rgb)
        
        out_path = out_images_dir / f"{img_path.stem}. pt"
        temp_path = out_path.with_suffix('.pt.tmp')
        
        torch.save(image_tensor, temp_path)
        temp_path.rename(out_path)
        
        return (True, None)
    except Exception as e: 
        return (False, f"{img_path. name}: {str(e)}")


def process_single_sequence(seq_info, input_dir, output_dir, image_size, progress_dict=None):
    """Process a single sequence."""
    # Create transform inside process
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms. Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    seq_name = seq_info
    seq_dir = input_dir / seq_name
    
    images_dir = seq_dir / "images"
    events_dir = seq_dir / "events"
    
    if not images_dir.exists() or not events_dir.exists():
        if progress_dict is not None:
            progress_dict[seq_name] = 'missing_dirs'
        return (0, 0, [f"Missing directories in {seq_name}"])
    
    # Create output directories
    out_seq_dir = output_dir / seq_name
    out_images_dir = out_seq_dir / "images"
    out_events_dir = out_seq_dir / "events"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_events_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    image_files = sorted(images_dir.glob("*.png"))
    expected_images = len(image_files)
    num_images = 0
    errors = []
    
    for img_path in image_files:
        success, error = process_single_image(img_path, transform, out_images_dir)
        if success:
            num_images += 1
        else:
            errors.append(error)
    
    if num_images < expected_images:
        errors.append(f"⚠️ {seq_name}: Expected {expected_images} images, processed {num_images}")
    
    # Copy events
    event_files = sorted(events_dir.glob("*.pt"))
    expected_events = len(event_files)
    num_events = 0
    
    for event_path in event_files:
        try:
            out_path = out_events_dir / event_path.name
            src_size = event_path.stat().st_size
            shutil.copy2(event_path, out_path)
            dst_size = out_path.stat().st_size
            
            if src_size != dst_size:
                errors.append(f"Size mismatch:  {event_path.name}")
                out_path.unlink()
            else:
                num_events += 1
        except Exception as e:
            errors.append(f"Event copy error {event_path.name}: {e}")
    
    if num_events < expected_events:
        errors.append(f"⚠️ {seq_name}:  Expected {expected_events} events, copied {num_events}")
    
    # Update progress
    if progress_dict is not None:
        progress_dict[seq_name] = 'done'
    
    # Print progress (每个进程独立打印)
    print(f"✓ {seq_name}: {num_images} images, {num_events} events", flush=True)
    
    return (num_images, num_events, errors)


def preprocess_dataset_parallel(input_dir, output_dir, image_size=(392, 518), num_workers=None):
    """Preprocess dataset using multiple processes - STABLE VERSION."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    print(f"🚀 Starting parallel preprocessing")
    print(f"   Workers: {num_workers}")
    print(f"   Input:    {input_dir}")
    print(f"   Output:  {output_dir}")
    print(f"   Image size: {image_size}")
    print()
    
    # Get all sequences
    sequences = [d. name for d in sorted(input_dir.iterdir()) if d.is_dir()]
    
    if not sequences:
        print("❌ No sequences found!")
        return
    
    print(f"Found {len(sequences)} sequences")
    print(f"Starting processing.. .\n")
    
    # Create partial function
    process_func = partial(
        process_single_sequence,
        input_dir=input_dir,
        output_dir=output_dir,
        image_size=image_size,
        progress_dict=None  # 不使用共享字典，避免锁竞争
    )
    
    # Process with map_async (更稳定)
    total_images = 0
    total_events = 0
    all_errors = []
    
    start_time = time.time()
    
    # 使用 map 而不是 imap，更稳定但内存占用稍高
    print("Processing sequences (progress will be printed as completed)...")
    print("-" * 60)
    
    with Pool(processes=num_workers) as pool:
        try:
            # 使用 chunksize 来优化性能
            chunksize = max(1, len(sequences) // (num_workers * 4))
            results = pool.map(process_func, sequences, chunksize=chunksize)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            pool.terminate()
            pool.join()
            return
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            pool.terminate()
            pool.join()
            return
    
    print("-" * 60)
    
    # Aggregate results
    for num_images, num_events, errors in results:
        total_images += num_images
        total_events += num_events
        all_errors.extend(errors)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Preprocessing complete in {elapsed:.1f}s!")
    print(f"   Processed {total_images} images")
    print(f"   Copied {total_events} event files")
    print(f"   Speed: {len(sequences)/elapsed:.1f} seq/s")
    
    if all_errors:
        print(f"\n⚠️ {len(all_errors)} errors occurred:")
        for err in all_errors[: 10]: 
            print(f"   - {err}")
        if len(all_errors) > 10:
            print(f"   ... and {len(all_errors) - 10} more")
    
    print(f"\n📁 Output:  {output_dir}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="Parallel preprocess dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image_height", type=int, default=392)
    parser.add_argument("--image_width", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=None)
    
    args = parser.parse_args()
    
    preprocess_dataset_parallel(
        args. input,
        args.output,
        image_size=(args. image_height, args.image_width),
        num_workers=args.num_workers
    )