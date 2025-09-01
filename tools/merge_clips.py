#!/usr/bin/env python3
"""
Script to merge multiple video clips into one single video.
Usage: python merge_clips.py [clips_dir] [output_video]
"""

import cv2
import os
import argparse
from pathlib import Path
import glob

def get_video_info(video_path):
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return fps, width, height, frame_count

def merge_videos(input_dir, output_path):
    """Merge all MP4 files in the input directory into one video"""
    
    # Get all MP4 files and sort them
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    if not video_files:
        print(f"❌ No MP4 files found in {input_dir}")
        return False
    
    print(f"📁 Found {len(video_files)} video files to merge:")
    for i, video_file in enumerate(video_files, 1):
        filename = os.path.basename(video_file)
        file_size = os.path.getsize(video_file) / (1024 * 1024)  # MB
        print(f"   {i:2d}. {filename} ({file_size:.1f} MB)")
    
    # Get properties from the first video (assuming all have same properties)
    first_video = video_files[0]
    fps, width, height, _ = get_video_info(first_video)
    
    print(f"\n📹 Video properties: {width}x{height} @ {fps:.1f} FPS")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Failed to create output video: {output_path}")
        return False
    
    total_frames = 0
    print(f"\n🎬 Merging videos...")
    
    # Process each video file
    for i, video_file in enumerate(video_files, 1):
        print(f"   Processing clip {i}/{len(video_files)}: {os.path.basename(video_file)}")
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"   ⚠️  Warning: Cannot open {video_file}, skipping...")
            continue
        
        # Copy all frames from current video
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if dimensions don't match (safety check)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
            frame_count += 1
            total_frames += 1
        
        cap.release()
        print(f"      ✅ Added {frame_count} frames")
    
    out.release()
    
    # Get output file size
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    duration = total_frames / fps
    
    print(f"\n🎉 Successfully merged {len(video_files)} clips!")
    print(f"📊 Output: {output_path}")
    print(f"   Size: {output_size:.1f} MB")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Total frames: {total_frames}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Merge video clips into one video')
    parser.add_argument('clips_dir', nargs='?', default='./data/clips', 
                       help='Directory containing video clips (default: ./data/clips)')
    parser.add_argument('output_video', nargs='?', default='./data/merged_clips.mp4', 
                       help='Output merged video file (default: ./data/merged_clips.mp4)')
    
    args = parser.parse_args()
    
    clips_dir = args.clips_dir
    output_video = args.output_video
    
    # Check if clips directory exists
    if not os.path.exists(clips_dir):
        print(f"❌ Error: Clips directory '{clips_dir}' not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        success = merge_videos(clips_dir, output_video)
        if success:
            print(f"\n✨ You can now test the merged video with your CUDA application:")
            print(f"   ./run.sh")
            print(f"   Then choose option 4 (Recorded Video) and use: {os.path.basename(output_video)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
