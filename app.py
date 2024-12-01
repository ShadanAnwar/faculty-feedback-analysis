import streamlit as st
import sqlite3
import hashlib
from faculty_page import faculty_page
from hod_page import hod_page

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Database setup for users
def create_user_table():
    conn = sqlite3.connect('feedback_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT, 
                  user_type TEXT)''')
    conn.commit()
    conn.close()

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Add user to database
def add_user(username, password, user_type):
    conn = sqlite3.connect('feedback_database.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)", 
                  (username, hashed_password, user_type))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Validate login
def validate_login(username, password, user_type):
    conn = sqlite3.connect('feedback_database.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=? AND user_type=?", 
              (username, hashed_password, user_type))
    result = c.fetchone()
    conn.close()
    return result is not None

# Login Page
def login_page():
    st.title("Faculty Feedback Management System")
    
    # Ensure user table exists
    create_user_table()
    
    # Add some initial users if not exists (for demonstration)
    conn = sqlite3.connect('feedback_database.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Add some default users
        add_user('faculty1', 'password1', 'faculty')
        add_user('hod1', 'password1', 'hod')
    conn.close()
    
    # Login section
    st.sidebar.title("Login")
    login_type = st.sidebar.selectbox("Login as", ["", "Faculty", "HOD"])
    
    if login_type:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if validate_login(username, password, login_type.lower()):
                st.session_state.logged_in = True
                st.session_state.user_type = login_type.lower()
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials")
    
    # Registration section (optional)
    st.sidebar.title("Register New User")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    new_user_type = st.sidebar.selectbox("User Type", ["", "Faculty", "HOD"])
    
    if st.sidebar.button("Register"):
        if new_username and new_password and new_user_type:
            if add_user(new_username, new_password, new_user_type.lower()):
                st.sidebar.success("User registered successfully!")
            else:
                st.sidebar.error("Username already exists")
        else:
            st.sidebar.error("Please fill all fields")

# Main application logic
def main():
    # Check if logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Navigation based on user type
        if st.session_state.user_type == 'faculty':
            faculty_page()
        elif st.session_state.user_type == 'hod':
            hod_page()

if __name__ == "__main__":
    main()