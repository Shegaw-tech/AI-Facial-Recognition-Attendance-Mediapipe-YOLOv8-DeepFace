import sqlite3
from datetime import datetime
DATABASE = 'attendance.db'
def get_todays_attendance():
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Query to select all attendance records for today including full details from both tables
    cursor.execute(
        "SELECT a.id, u.id AS user_id, u.name, u.occupation, a.timestamp FROM attendance a "
        "JOIN users u ON a.user_id = u.id WHERE DATE(a.timestamp) = ?",
        (today_date,)
    )

    # Fetch all results
    attendance_list = cursor.fetchall()

    # Close the connection
    conn.close()

    # Return the attendance list
    return attendance_list


def get_all_requests():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Query to fetch all leave requests (including employee_name, start_date, end_date, reason)
    cursor.execute("SELECT id, employee_name, start_date, end_date, reason FROM leave_requests")
    leave_requests = cursor.fetchall()

    # Query to fetch all issue reports (including employee_name, issue_description)
    cursor.execute("SELECT id, employee_name, issue_description FROM issue_reports")
    issue_reports = cursor.fetchall()

    # Query to fetch all training requests (including employee_name, training_topic, training_reason)
    cursor.execute("SELECT id, employee_name, training_topic, training_reason FROM training_requests")
    training_requests = cursor.fetchall()

    # Close the connection
    conn.close()

    return leave_requests, issue_reports, training_requests


def display_requests():
    # Get all requests from the database
    leave_requests, issue_reports, training_requests = get_all_requests()

    # Print Leave Requests
    print("Leave Requests:")
    if leave_requests:
        for request in leave_requests:
            print(f"ID: {request[0]}, Employee Name: {request[1]}, Start Date: {request[2]}, End Date: {request[3]}, Reason: {request[4]}")
    else:
        print("No leave requests found.")

    print("\nIssue Reports:")
    if issue_reports:
        for report in issue_reports:
            print(f"ID: {report[0]}, Employee Name: {report[1]}, Issue Description: {report[2]}")
    else:
        print("No issue reports found.")

    print("\nTraining Requests:")
    if training_requests:
        for request in training_requests:
            print(f"ID: {request[0]}, Employee Name: {request[1]}, Training Topic: {request[2]}, Training Reason: {request[3]}")
    else:
        print("No training requests found.")


def display_attendance():
    # Get today's attendance
    attendance = get_todays_attendance()

    # Print Today's Attendance
    print("\nToday's Attendance List:")
    if attendance:
        for record in attendance:
            print(f"Attendance ID: {record[0]}, User ID: {record[1]}, Name: {record[2]}, Occupation: {record[3]}, Timestamp: {record[4]}")
    else:
        print("No attendance recorded for today.")


# usage to print all requests and today's attendance
display_requests()
display_attendance()
