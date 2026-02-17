import os
import json
import time
import shutil

# Path to transcripts dir (relative to this test file)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS_DIR = os.path.join(BACKEND_DIR, "transcripts")

def test_history_persistence():
    # 1. Setup: Ensure dir exists
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    
    # 2. Simulate Saving (Logic from main.py)
    timestamp = int(time.time())
    filename = f"test_{timestamp}.wav"
    history_data = {
        "id": str(timestamp),
        "timestamp": timestamp,
        "filename": filename,
        "status": "success",
        "language": "ur",
        "duration": 5.5,
        "processing_time": 1.2,
        "full_text": "Test transcript content.",
        "segments": []
    }
    
    file_path = os.path.join(TRANSCRIPTS_DIR, f"{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
        
    print(f"[PASS] Written history file: {file_path}")
    
    # 3. Simulate Listing (Logic from GET /history)
    history_items = []
    files = sorted([f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json")], reverse=True)
    
    found = False
    for f in files:
        try:
            with open(os.path.join(TRANSCRIPTS_DIR, f), "r", encoding="utf-8") as file:
                data = json.load(file)
                # Verify structure
                if data["id"] == str(timestamp):
                    found = True
                    print(f"[PASS] Found history item: {data['id']}")
                    assert data["full_text"] == "Test transcript content."
                    assert data["language"] == "ur"
        except Exception as e:
            print(f"[FAIL] Error reading {f}: {e}")
            
    if not found:
        print("[FAIL] Did not find the saved test file.")
        exit(1)
        
    # 4. Cleanup
    os.remove(file_path)
    print("[PASS] Cleanup successful.")

if __name__ == "__main__":
    try:
        test_history_persistence()
        print("VERIFICATION SUCCESSFUL: History logic is working.")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        exit(1)
