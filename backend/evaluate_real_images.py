import os
import json
import mimetypes
from qwen_agent import analyze_media

# Directory to scan
DIR_PATH = r"D:\Stevens Hackathon\real\Real"
OUTPUT_FILE = r"D:\Stevens Hackathon\real\Real\evaluation_results.txt"

def evaluate_directory():
    if not os.path.exists(DIR_PATH):
        print(f"Directory not found: {DIR_PATH}")
        return

    print(f"Starting evaluation of images in: {DIR_PATH}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("FRAUD DETECTION SYSTEM - BATCH EVALUATION RESULTS\n")
        f.write(f"Directory: {DIR_PATH}\n")
        f.write("=" * 80 + "\n\n")

    files = [f for f in os.listdir(DIR_PATH) if os.path.isfile(os.path.join(DIR_PATH, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total = len(image_files)
    print(f"Found {total} images to evaluate.\n")

    fake_count = 0

    for i, file_name in enumerate(image_files, 1):
        file_path = os.path.join(DIR_PATH, file_name)
        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            content_type = "image/jpeg"
            
        print(f"[{i}/{total}] Evaluating: {file_name}...")
        
        try:
            result = analyze_media(file_path, content_type)
            
            classification = result.get('classification', 'Unknown')
            if classification.lower() == 'fake':
                fake_count += 1
                
            # Format output
            output = f"--- {file_name} ---\n"
            output += f"Verdict: {classification}\n"
            output += f"Confidence Score: {result.get('confidence_score', 0.0) * 100:.2f}%\n"
            output += f"Reason: {result.get('reason', 'N/A')}\n"
            output += "Agent 1 Findings:\n"
            output += f"{result.get('vision_findings', 'N/A')}\n"
            output += "-" * 40 + "\n\n"
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(output)
                
            print(f"  -> Verdict: {classification}")
            
        except Exception as e:
            error_msg = f"--- {file_name} ---\nERROR: {str(e)}\n{'-' * 40}\n\n"
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(error_msg)
            print(f"  -> ERROR: {str(e)}")

    summary_msg = f"\nEvaluation complete.\nTotal Images: {total}\nTotal Classified as Fake: {fake_count}\nResults saved to {OUTPUT_FILE}\n"
    print(summary_msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(summary_msg)

if __name__ == "__main__":
    evaluate_directory()
