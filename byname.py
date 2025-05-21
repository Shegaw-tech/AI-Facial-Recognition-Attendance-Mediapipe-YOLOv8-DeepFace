import sqlite3
import os

DATABASE = 'attendance.db'
UPLOAD_FOLDER = 'static/faces'

def delete_user_by_name(user_name):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Get the photo path before deleting
    cursor.execute("SELECT id, photo_path FROM users WHERE name = ?", (user_name,))
    row = cursor.fetchone()

    if row:
        user_id, photo_path = row

        # Delete related attendance records (optional)
        cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))

        # Delete the user from the users table
        cursor.execute("DELETE FROM users WHERE name = ?", (user_name,))
        conn.commit()

        # Delete the photo file if it exists
        if os.path.exists(photo_path):
            os.remove(photo_path)
            print(f"Deleted photo: {photo_path}")
        else:
            print("Photo file not found.")

        print(f"User '{user_name}' deleted successfully.")
    else:
        print(f"No user found with name: {user_name}")

    conn.close()
