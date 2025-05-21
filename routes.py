from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3
import os
from detect_and_compare import capture_and_compare_from_saved_image
from byname import delete_user_by_name
DATABASE = 'attendance.db'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/faces'
def register_routes(app):
    @app.route('/')
    def home():
        return render_template('index.html')  # Now renders the home page

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            name = request.form['name']
            occupation = request.form['occupation']
            file = request.files['photo']
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (name, occupation, photo_path) VALUES (?, ?, ?)",
                               (name, occupation, filepath))
                conn.commit()
                conn.close()
                return redirect(url_for('register'))
        return render_template('register.html')

    @app.route('/delete_user_by_name', methods=['POST'])
    def delete_user_by_name_route():
        user_name = request.form['name']
        delete_user_by_name(user_name)
        return redirect(url_for('register'))

def attendance_routes(app):
    @app.route('/attendance')
    def attendance():
        return render_template('attendance.html')

    @app.route('/mark_attendance', methods=['POST'])
    def mark_attendance():
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image uploaded'})

        image = request.files['image']
        captured_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured.jpg')

        image.save(captured_image_path)
        #return capture_and_compare(captured_image_path)

        return capture_and_compare_from_saved_image(captured_image_path)



# Add routes for handling leave request form, issue report form, and training request form
def request_routes(app):
    @app.route('/submit_leave_request', methods=['POST'])
    def submit_leave_request():
        employee_name = request.form['employee_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        reason = request.form['reason']

        # Insert the leave request into the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO leave_requests (employee_name, start_date, end_date, reason)
                          VALUES (?, ?, ?, ?)''',
                       (employee_name, start_date, end_date, reason))
        conn.commit()
        conn.close()

        return redirect(url_for('home'))

    @app.route('/submit_issue_report', methods=['POST'])
    def submit_issue_report():
        employee_name = request.form['employee_name']
        issue_description = request.form['issue_description']

        # Insert the issue report into the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO issue_reports (employee_name, issue_description)
                          VALUES (?, ?)''',
                       (employee_name, issue_description))
        conn.commit()
        conn.close()

        return redirect(url_for('home'))

    @app.route('/submit_training_request', methods=['POST'])
    def submit_training_request():
        employee_name = request.form['employee_name']
        training_topic = request.form['training_topic']
        training_reason = request.form['training_reason']

        # Insert the training request into the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO training_requests (employee_name, training_topic, training_reason)
                          VALUES (?, ?, ?)''',
                       (employee_name, training_topic, training_reason))
        conn.commit()
        conn.close()

        return redirect(url_for('home'))

