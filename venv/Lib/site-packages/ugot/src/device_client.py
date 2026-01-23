import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

from base_client import GrpcClient

try:
    from ugot.grpc_pb import device_pb2,device_pb2_grpc
except:
    pass
try:
    import device_pb2,device_pb2_grpc
except:
    pass


class DeviceClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = device_pb2_grpc.DeviceServiceGrpcStub(channel=self.channel)

    """ 显示屏 """
    def display_background(self, color):
        input_data = device_pb2.DisplayUIRequest()
        input_data.background_color = color  # "FFFF0000"

        response = self.client.setDisplayUI(input_data)
        return response

    def print_text(self, text, color):
        input_data = device_pb2.DisplayUIRequest()
        # input_data.background_color = "FF000000"
        text_info = input_data.text_list.add()
        text_info.text = str(text)
        text_info.font_color = color
        text_info.font_size = 16

        response = self.client.setDisplayUI(input_data)
        return response

    def print_text_newline(self, text, color):
        input_data = device_pb2.DisplayUIRequest()
        # input_data.background_color = "FF00FF00"
        text_info = input_data.text_list.add()
        text_info.text = str(text)
        text_info.font_color = color
        text_info.font_size = 16
        text_info.newline = True

        response = self.client.setDisplayUI(input_data)
        return response

    def display_image(self, image_name):
        input_data = device_pb2.DisplayUIRequest()
        input_data.img_name = image_name

        response = self.client.setDisplayUI(input_data)
        return response

    def clear_screen(self):
        input_data = device_pb2.DisplayUIRequest()
        input_data.reset = True

        response = self.client.setDisplayUI(input_data)
        return response
    
    def display_emotion(self, name, loop = False):
        input_data = device_pb2.DisplayEmotionRequest()
        input_data.name = str(name)
        input_data.loop = loop 
        input_data.path = '' 
        input_data.extra = '' 

        response = self.client.setDisplayEmotion(input_data)
        return response

    """ 灯效 """
    def showLightEffect(self, color, effect, level=2):
        input_data = device_pb2.LightEffectRequest()
        input_data.effect = effect
        input_data.color = color
        input_data.level = level

        response = self.client.showLightEffect(input_data)
        return response

    def showLightColor(self, top_light_color=-1, left_light_color=-1, right_light_color=-1, down_light_color=-1):

        input_data = device_pb2.LightColorRequest()
        input_data.top_light = top_light_color
        input_data.left_light = left_light_color
        input_data.right_light = right_light_color
        input_data.down_light = down_light_color

        response = self.client.showLightColor(input_data)
        return response

    def turnOffLight(self, lights):
        input_data = device_pb2.TurnOffLightRequest()
        ids = []
        if isinstance(lights, list):
            ids += lights
        elif isinstance(lights, int):
            ids.append(lights)
        for item in ids:
            input_data.lights.append(item)

        response = self.client.showLightEffect(input_data)
        return response

    def turnOffAllLights(self):
        input_data = device_pb2.LightEffectRequest()
        input_data.effect = 1
        input_data.color = 0
        input_data.level = 2

        response = self.client.showLightEffect(input_data)
        return response

    def setVolume(self, volume):
        input_data = device_pb2.SetVolumeRequest()
        input_data.volume = volume

        response = self.client.setVolume(input_data)
        return response

    def getVolume(self):
        input_data = device_pb2.GetVolumeRequest()

        response = self.client.getVolume(input_data)
        return response

    def getMacAddress(self):
        input_data = device_pb2.MacAddressRequest()

        response = self.client.getMacAddress(input_data)
        return response

    def getDeviceList(self):
        input_data = device_pb2.DeviceListRequest()

        response = self.client.getDeviceList(input_data)
        return response
    
    def getDeviceModel(self):
        input_data = device_pb2.DeviceModelRequest()

        response = self.client.getDeviceModel(input_data)
        return response.name
    
    def getLanguage(self):
        input_data = device_pb2.GetLangRequest()

        response = self.client.getLanguage(input_data)
        return response

    def __del__(self):
        # # 销毁时候停止灯光
        try:
            self.turnOffAllLights()
        except Exception as e:
            # logging.debug('device turnOffAllLights error:')
            pass