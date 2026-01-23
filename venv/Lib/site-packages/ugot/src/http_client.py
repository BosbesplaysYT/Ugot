import requests
import os
import json

#basic_url = "http://192.168.8.117:7000/"
#basic_url = "http://127.0.0.1:7000/"

def upload_vision_picture(basic_url, filename):
    #上传文件到服务器
    file_path = filename # os.path.join("/Users/jesse/Downloads/",filename)
    file = {'file': open(file_path,'rb')}
    url = basic_url + "vision/picture"
    response = requests.post(url=url, files=file)
    res = json.loads(str(response.content.decode("utf-8")))
    # print(res)
    return res

# def main():
#     upload_vision_picture("aobama.jpg")

# if __name__ == '__main__':
#     main()
