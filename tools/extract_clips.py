#!/usr/bin/env python3
"""
Script to extract 10 random 10-second clips from a video file.
Usage: python extract_clips.py [input_video] [output_dir]
"""

import cv2
import os
import random
import argparse
from pathlib import Path

def get_video_info(video_path):
    """Get video duration and FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return duration, fps, width, height

def extract_clip(input_path, output_path, start_time, clip_duration, fps, width, height):
    """Extract a clip from the video"""
    cap = cv2.VideoCapture(input_path)
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate start and end frames
    start_frame = int(start_time * fps)
    end_frame = int((start_time + clip_duration) * fps)
    
    # Set the starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    while cap.isOpened() and frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"✅ Created clip: {output_path} (start: {start_time:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description='Extract 10 random 10-second clips from a video')
    parser.add_argument('input_video', nargs='?', default='./data/input.mp4', 
                       help='Input video file (default: ./data/input.mp4)')
    parser.add_argument('output_dir', nargs='?', default='./data/clips', 
                       help='Output directory (default: ./data/clips)')
    
    args = parser.parse_args()
    
    input_video = args.input_video
    output_dir = args.output_dir
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video '{input_video}' not found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Get video information
        duration, fps, width, height = get_video_info(input_video)
        print(f"📹 Video info: {duration:.1f}s, {fps:.1f} FPS, {width}x{height}")
        
        # Check if video is long enough
        clip_duration = 10  # seconds
        if duration < clip_duration:
            print(f"❌ Error: Video is too short ({duration:.1f}s). Need at least {clip_duration}s.")
            return
        
        # Generate 10 random start times
        max_start_time = duration - clip_duration
        start_times = []
        
        # Ensure clips don't overlap by sorting and checking
        attempts = 0
        while len(start_times) < 10 and attempts < 100:
            start_time = random.uniform(0, max_start_time)
            
            # Check for overlap with existing clips
            overlap = False
            for existing_start in start_times:
                if abs(start_time - existing_start) < clip_duration:
                    overlap = True
                    break
            
            if not overlap:
                start_times.append(start_time)
            
            attempts += 1
        
        # Sort start times for better organization
        start_times.sort()
        
        print(f"🎬 Extracting {len(start_times)} clips...")
        
        # Extract clips
        input_name = Path(input_video).stem
        for i, start_time in enumerate(start_times, 1):
            output_filename = f"{input_name}_clip_{i:02d}_t{start_time:.1f}s.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            extract_clip(input_video, output_path, start_time, clip_duration, fps, width, height)
        
        print(f"\n🎉 Successfully created {len(start_times)} clips in '{output_dir}'")
        
        # List created files
        print("\n📁 Created files:")
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.mp4'):
                file_path = os.path.join(output_dir, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   {filename} ({file_size:.1f} MB)")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
