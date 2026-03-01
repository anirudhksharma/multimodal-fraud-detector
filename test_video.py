import sys
import json
from backend.qwen_agent import analyze_video

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path_to_video>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    print(f"🎬 Testing Video Detection on:\n{video_path}\n")
    print("-" * 50)
    
    try:
        # Run the video pipeline
        result = analyze_video(video_path)
        
        print("\n" + "=" * 50)
        print("🔍 ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"VERDICT:      {result.get('classification')}")
        print(f"CONFIDENCE:   {result.get('confidence_score') * 100:.1f}%")
        print(f"CONSENSUS:    {result.get('consensus')}")
        print(f"REASONING:    {result.get('reason')}")
        print("-" * 50)
        
        # Show breakdown
        print("FRAME BREAKDOWN:")
        for frame, info in result.get('vote_breakdown', {}).items():
            print(f"  {frame}: {info.get('classification')} ({info.get('confidence', 0)*100:.0f}%)")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()
