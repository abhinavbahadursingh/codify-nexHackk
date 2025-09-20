import sys
sys.path.insert(1, r'/demo_working/firebase_auth.py')
import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    """Initialize Firebase and return a database reference"""
    cred = credentials.Certificate(r"E:\codify\traffic-monitoring-f490d-firebase-adminsdk-fbsvc-29d3ed2ce7")  # Update with your correct path
    if not firebase_admin._apps:  # Prevents re-initialization error
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/'  # Replace with your Firebase URL
        })
    return db.reference("traffic_data")  # Returns database reference


# initialize_firebase()