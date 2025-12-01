import requests

with open('/home/hp/Documents/Daily_Task/Day_2/Assets/shapes_1.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/detect-shapes', 
                             files={'image': f})
    print(response.json())
