import os
import cv2
import sqlite3
DATABASE = 'attendance.db'

def check_user_photos():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, photo_path FROM users")
    users = cursor.fetchall()
    conn.close()

    for user_id, name, path in users:
        print(f"Checking [{user_id}] {name!r}: {path}", end=" → ")

        # 1) Does the file exist?
        if not os.path.isfile(path):
            print("❌ File not found")
            continue

        # 2) Can OpenCV read it?
        img = cv2.imread(path)
        if img is None:
            print("❌ Unreadable or corrupted image")
            continue

        # 3) If loaded, report shape
        h, w, _ = img.shape
        print(f"✅ OK ({w}×{h})")

if __name__ == "__main__":
    check_user_photos()
