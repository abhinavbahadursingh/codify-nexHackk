import requests
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# ✅ Hostinger upload API URL
UPLOAD_URL = "https://goldenrod-bee-415447.hostingersite.com/upload_api.php"

# ✅ Firebase setup
cred = credentials.Certificate(
    r"E:\codify_hackquanta\traffic-monitoring-f490d-firebase-adminsdk-fbsvc-29d3ed2ce7.json"
)  # Path to your Firebase Admin SDK JSON
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/"  # Replace with your Firebase DB URL
})


def upload_image(image_path):
    """Uploads image to Hostinger and returns public URL"""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "multipart/form-data")}
            headers = {"User-Agent": "Mozilla/5.0"}  # Avoid Hostinger 403
            response = requests.post(UPLOAD_URL, files=files, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Uploaded Successfully:", image_path)
                print("Public URL:", data["url"])
                return data["url"]
            else:
                print("❌ Upload failed:", data.get("message"))
        else:
            print("❌ Server error:", response.text)
    except Exception as e:
        print("⚠️ Error uploading image:", e)

    return None


def save_to_firebase(crash_id, urls_dict):
    """Adds crash frame URLs directly inside the crash record"""
    ref = db.reference(f"traffic_data/crash_vehicle/{crash_id}")

    # Update existing crash entry with frame URLs
    update_data = {
        "frame_urls": urls_dict,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    ref.update(update_data)   # ✅ Updates inside crash-1 / crash-2 / crash-3
    print(f"🔥 Crash data updated in Firebase for {crash_id}:", update_data)


if __name__ == "__main__":
    # ✅ Only before & after frames
    images = {
        "before": r"E:\codify_hackarena\crashDetectionSystem\Testing\crash_frames\crash-3_before.jpg",
        "after":  r"E:\codify_hackarena\crashDetectionSystem\Testing\crash_frames\crash-3_after.jpg"
    }

    uploaded_urls = {}

    for key, path in images.items():
        url = upload_image(path)
        if url:
            uploaded_urls[key] = url

    # ✅ Update crash-3 with URLs (change crash_id accordingly)
    if uploaded_urls:
        save_to_firebase("crash-3", uploaded_urls)
