import requests

resp = requests.post('http://localhost:5000/predict', files={'file': open('nine.png', 'rb')})

print(resp.text)