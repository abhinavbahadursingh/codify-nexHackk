import csv
import os


def create_csv(file_name, data):
    """
    Creates or appends data to a CSV file.

    Parameters:
    file_name (str): Name of the CSV file.
    data (list of dict or dict): List of dictionaries containing the data.
    """
    if not data:
        print("No data provided.")
        return

    # Convert single dictionary to list of dictionaries
    if isinstance(data, dict):
        data = [data]

    # Get fieldnames from the first dictionary
    fieldnames = list(data[0].keys())

    file_exists = os.path.isfile(file_name + ".csv")  # Check if file exists

    # Open in append mode ('a') and handle headers
    with open(file_name + ".csv", mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if file is new (does not exist)
        if not file_exists:
            writer.writeheader()

        # Append data
        writer.writerows(data)

    print(f"Data has been added to '{file_name}.csv' successfully!")


# Example Usage:

# Adding data initially
data = [
    {"Name": "Alice", "Age": 25, "City": "New York"},
    {"Name": "Bob", "Age": 30, "City": "Los Angeles"}
]
create_csv("typeCsv", data)

# Appending new data
new_data = {"Name": "Charlie", "Age": 28, "City": "Chicago"}
create_csv("typeCsv", new_data)
