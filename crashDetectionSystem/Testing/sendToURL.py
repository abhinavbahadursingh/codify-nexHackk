import requests

# The URL where you want to upload the image
url = "https://goldenrod-bee-415447.hostingersite.com/uploads/"

# Path of the image on your PC
file_path = "E:\codify_hackarena\crashDetectionSystem\Testing\crash_frames\crash-3_after.jpg"

# Open the image in binary mode
with open(file_path, "rb") as img:
    files = {"file": img}  # 'file' should match the server's expected field name
    response = requests.post(url, files=files)

# Print server response
print(response.status_code)
print(response.text)
