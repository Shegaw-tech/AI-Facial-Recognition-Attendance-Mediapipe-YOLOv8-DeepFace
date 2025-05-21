import sqlite3

DATABASE = 'attendance.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        occupation TEXT NOT NULL,
                        photo_path TEXT NOT NULL)''')

    # Attendance table
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id))''')

    # Leave Requests table
    cursor.execute('''CREATE TABLE IF NOT EXISTS leave_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_name TEXT NOT NULL,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        reason TEXT NOT NULL)''')

    # Issue Reports table
    cursor.execute('''CREATE TABLE IF NOT EXISTS issue_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_name TEXT NOT NULL,
                        issue_description TEXT NOT NULL)''')

    # Training Requests table
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_name TEXT NOT NULL,
                        training_topic TEXT NOT NULL,
                        training_reason TEXT NOT NULL)''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
