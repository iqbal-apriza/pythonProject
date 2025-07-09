import requests
import serial
import time
from requests.auth import HTTPBasicAuth

VLC_URL     = "http://localhost:8080/requests/status.json"
USERNAME    = ""
PASSWORD    = "iqbal123"

def get_playback_time():
    try:
        response = requests.get(VLC_URL, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        if response.status_code == 200:
            data = response.json()
            playback_time = data.get("time", 0)  # Get playback time in seconds
            return playback_time
    except requests.RequestException as e:
        print("Error fetching VLC data:", e)
    return None

while 1:
    playback_time = get_playback_time()
    if playback_time is not None:
        print(f"Playback Time: {playback_time} sec")
    time.sleep(0.01)