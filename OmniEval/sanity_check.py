# OmniEval/sanity_check.py
import argparse
import importlib.util
import os
import sys
from moviepy import VideoFileClip

def check_video_has_audio(video_path: str) -> bool:
    try:
        clip = VideoFileClip(video_path)
        has_audio = clip.audio is not None
        clip.close()
        return has_audio
    except Exception as e:
        print(f"[Warning] Failed to read audio track info: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sanity check tool for testing Omni-modal model wrappers.")
    parser.add_argument('--wrapper-file', type=str, required=True, help="Path to your model wrapper python file")
    parser.add_argument('--class-name', type=str, required=True, help="Name of the class in your wrapper")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the model weights")
    parser.add_argument('--video-path', type=str, required=True, help="Path to a test video file")
    parser.add_argument('--prompt', type=str, default="What is happening in this video? Please describe it briefly.", help="Test prompt")
    parser.add_argument('--device-ids', type=str, default="0", help="Comma-separated GPU IDs to test on (default: 0)")
    
    args = parser.parse_args()

    # Dynamic path injection for user imports
    wrapper_dir = os.path.dirname(os.path.abspath(args.wrapper_file))
    if wrapper_dir not in sys.path:
        sys.path.insert(0, wrapper_dir)
        
    omni_eval_root = os.path.dirname(os.path.abspath(__file__))
    if omni_eval_root not in sys.path:
        sys.path.insert(0, omni_eval_root)

    print("="*60)
    print("🚀 OmniEval: Starting Sanity Check")
    print("="*60)

    try:
        device_ids = [int(idx.strip()) for idx in args.device_ids.split(",")]
        
        print(f"Loading wrapper module: {args.wrapper_file} -> {args.class_name}...")
        spec = importlib.util.spec_from_file_location("user_module", args.wrapper_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModelClass = getattr(module, args.class_name)

        print(f"Initializing model on devices {device_ids} (This might take a while)...")
        model_instance = ModelClass(model_path=args.model_path, device_ids=device_ids)
        print("Model initialized successfully! ✅")

        has_audio = check_video_has_audio(args.video_path)
        print(f"\nRunning inference on video: {args.video_path} (Has Audio: {has_audio})")
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        
        output = model_instance.generate(
            video_path=args.video_path, 
            prompt=args.prompt, 
            has_audio=has_audio
        )
        
        print("-" * 60)
        print("📝 Model Output (Raw):")
        print(output)
        print("\nSanity Check Passed! 🎉 Your wrapper is ready for full evaluation.")

    except Exception as e:
        print("\n[Error] Sanity Check Failed! ❌")
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()