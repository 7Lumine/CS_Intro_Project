import requests


with open("image.png", "rb") as f: # 送信する画像ファイル
    files = {"file": ("image.png", f)}
    # FastAPIサーバーが localhost:8000 で動いている前提
    response = requests.post("http://127.0.0.1:8000/send_image/", files=files) # data引数は削除
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)