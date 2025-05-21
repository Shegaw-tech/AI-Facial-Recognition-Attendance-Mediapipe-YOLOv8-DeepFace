from flask import Flask, jsonify, Response
import os
import sqlite3
from datetime import datetime
import json
from database import init_db
from routes import register_routes, attendance_routes, request_routes
from flask import Blueprint, render_template, request, abort, current_app
tracking_bp = Blueprint('tracking', __name__)
app = Flask(__name__)
app.config['ADMIN_PASSWORD'] = '123'
# Configure file upload folder
app.config['UPLOAD_FOLDER'] = 'static/faces'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the database
init_db()

# Register routes
register_routes(app)
attendance_routes(app)
request_routes(app)
DATABASE = 'attendance.db'
def get_todays_attendance():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # e.g. '2025-04-23'
    today_date = datetime.now().strftime('%Y-%m-%d')

    cursor.execute("""
        SELECT
          a.id,
          u.id   AS user_id,
          u.name,
          u.occupation,
          a.timestamp
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.timestamp LIKE ?
        ORDER BY a.timestamp ASC
    """, (today_date,))

    attendance_list = cursor.fetchall()
    conn.close()
    return attendance_list

def get_all_requests():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # 1) Leave requests
    cursor.execute("SELECT id, employee_name, start_date, end_date, reason FROM leave_requests")
    leave_requests = cursor.fetchall()

    # 2) Issue reports
    cursor.execute("SELECT id, employee_name, issue_description FROM issue_reports")
    issue_reports = cursor.fetchall()

    # 3) Training requests
    cursor.execute("SELECT id, employee_name, training_topic, training_reason FROM training_requests")
    training_requests = cursor.fetchall()

    # 4) Monthly attendance (all records for this month)
    month_prefix = datetime.now().strftime('%Y-%m') + '%'
    cursor.execute("""
        SELECT a.id, u.id AS user_id, u.name, u.occupation, a.timestamp
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.timestamp LIKE ?
        ORDER BY a.timestamp ASC
    """, (month_prefix,))
    monthly_attendance = cursor.fetchall()

    conn.close()
    return leave_requests, issue_reports, training_requests, monthly_attendance

@app.route('/tracking')
def tracking():
    password = request.args.get('password')  # Get password from URL

    # Unpack all four lists from get_all_requests
    leave_requests, issue_reports, training_requests, attendance = get_all_requests()

    # Check if the user is admin
    if password == current_app.config['ADMIN_PASSWORD']:
        data = {
            "attendance": attendance,  # Monthly attendance data
            "leave_requests": leave_requests,
            "issue_reports": issue_reports,
            "training_requests": training_requests
        }
    else:
        data = {
            "message": "Limited Access: Only issue reports are visible.",
            "issue_reports": issue_reports
        }

    # Pretty print the data as JSON
    pretty_json = json.dumps(data, indent=4, sort_keys=True)
    return Response(pretty_json, mimetype='application/json')

# ----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
