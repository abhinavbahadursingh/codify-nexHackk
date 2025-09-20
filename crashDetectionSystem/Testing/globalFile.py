import json
import os

# File to store global12 values
SAVE_FILE = "../../global12/global_store.json"

# Global dictionary
GLOBAL_STORE = {}

def save_value(key, value):
    GLOBAL_STORE[key] = value
    save_to_file()

def get_value(key, default=0):
    return GLOBAL_STORE.get(key, default)

def remove_value(key):
    if key in GLOBAL_STORE:
        del GLOBAL_STORE[key]
        save_to_file()

def save_to_file():
    """Write GLOBAL_STORE to JSON file"""
    with open(SAVE_FILE, "w") as f:
        json.dump(GLOBAL_STORE, f)

def load_from_file():
    """Load values from JSON file into GLOBAL_STORE"""
    global GLOBAL_STORE
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            GLOBAL_STORE = json.load(f)
    else:
        GLOBAL_STORE = {}

# ------------------------
# Example usage
# ------------------------
# load_from_file()  # Load saved values at program start
#
# print("Loaded store:", GLOBAL_STORE)
#
# save_value("username", "Shreya")
# save_value("score", 100)
#
# print("Username:", get_value("username"))
# print("Score:", get_value("score")-895)
