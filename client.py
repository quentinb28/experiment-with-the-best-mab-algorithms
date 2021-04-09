import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
body = {
    "petal_length": 4,
    "sepal_length": 7,
    "petal_width": 0.3,
    "sepal_width": 2.5
}

response = requests.post(url, data=body)
print(response.json())
