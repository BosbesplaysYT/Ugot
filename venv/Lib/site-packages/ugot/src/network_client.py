import logging
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

try:
    from ugot.grpc_pb import network_pb2, network_pb2_grpc
except:
    pass
try:
    import network_pb2, network_pb2_grpc
except:
    pass

from base_client import GrpcClient
from ugot.src.enum import E_Joypad
import json

class NetworkClient(GrpcClient):
    last_seq = -1
    last_msg = ''

    def __init__(self, address):
        super().__init__(address)

        self.client = network_pb2_grpc.NetworkServiceGrpcStub(channel=self.channel)

    ### 物联网
    def set_message_param(self, seq, msg):
        self.last_seq = seq
        self.last_msg = msg

    def check_same_message(self, seq, msg):
        if self.last_seq == seq and self.last_msg == msg:
            return True
        return False

    def setBrocastEnable(self, enable):
        input_data = network_pb2.SetBrocastEnableRequest()
        input_data.enable = enable

        response = self.client.setBrocastEnable(input_data)
        return response

    def setBrocastPort(self, nPort):
        input_data = network_pb2.SetBrocastPortRequest()
        input_data.nPort = str(nPort)

        response = self.client.setBrocastPort(input_data)
        return response

    def sendBrocastMsg(self, stMsg):
        input_data = network_pb2.SendBrocastMsgRequest()
        input_data.stMsg = str(stMsg)

        response = self.client.sendBrocastMsg(input_data)
        return response

    def getReceivedBrocastMsg(self):
        input_data = network_pb2.NetworkCommonRequest()

        response = self.client.getReceivedBrocastMsg(input_data)

        if response.code == 0:
            #  {"seq":1,"value":"你好1"}
            if not len(response.received_message):
                # 未收到消息
                self.set_message_param(-1, '')
                return self.last_msg
            try:
                import json
                data = json.loads(response.received_message)
                if data is None:
                    self.set_message_param(-1, '')
                    return self.last_msg

                seq = data['seq']
                value = data['value']

                if self.check_same_message(seq, value):
                    return ''
                else:
                    self.set_message_param(seq, value)
                    return self.last_msg
            except Exception as e:
                # LOG.addPythonLog(1, e)
                # assert ()
                logging.error('getReceivedBrocastMsg error!!')
        return ''

    ### WI-FI

    def getWifiStatus(self):
        input_data = network_pb2.NetworkCommonRequest()
        response = self.client.getWifiStatus(input_data)
        return response


    ### 手柄

    def getBTJoypadStatus(self):
        """获取蓝牙手柄点击状态
        """
        input_data = network_pb2.NetworkCommonRequest()

        response = self.client.getBTJoypadStatus(input_data)
        response = response.status

        # response = '{"stick": {"l": {"x": 100, "y": 100}, "r": {"x": 50, "y": 0}}, "arrow": {"l": 0, "r": 0, "u": 0, "d": 0}, "function": {"start": 0, "power": 0, "bt": 0}, "addition": {"l1": 0, "l2": 0, "l3": 0, "r1": 0, "r2": 0, "r3": 0}, "button": {"a": 1, "b": 0, "x": 0, "y": 0}}'
        return response

    # 获取当前点下的按钮
    def get_joypad_button_state(self):

        result = self.getBTJoypadStatus()

        if result is None or len(result) == 0:
            # 没有获取到事件
            return []
        
        try:
            data = json.loads(result)
            if data is None:
                return []
        except:
            # logging.error('get_joypad_button_state json loads error')
            return []

        pressed_btns = []
        arrow = data['arrow']
        if arrow['l'] == 1:
            pressed_btns.append(E_Joypad.L)
        if arrow['r'] == 1:
            pressed_btns.append(E_Joypad.R)
        if arrow['u'] == 1:
            pressed_btns.append(E_Joypad.U)
        if arrow['d'] == 1:
            pressed_btns.append(E_Joypad.D)

        function = data['function']

        addition = data['addition']
        if addition['l1'] == 1:
            pressed_btns.append(E_Joypad.L1)
        if addition['l2'] == 1:
            pressed_btns.append(E_Joypad.L2)
        if addition['r1'] == 1:
            pressed_btns.append(E_Joypad.R1)
        if addition['r2'] == 1:
            pressed_btns.append(E_Joypad.R2)
        #
        if addition['l3'] == 1:
            pressed_btns.append(E_Joypad.LS)
        if addition['r3'] == 1:
            pressed_btns.append(E_Joypad.RS)

        button = data['button']
        if button['a'] == 1:
            pressed_btns.append(E_Joypad.A)
        if button['b'] == 1:
            pressed_btns.append(E_Joypad.B)
        if button['x'] == 1:
            pressed_btns.append(E_Joypad.X)
        if button['y'] == 1:
            pressed_btns.append(E_Joypad.Y)

        pressed_btns.sort()
        # 用于注册按键按下事件标识
        # code_button = BLUETOOTH.pressed_button_to_str(pressed_btns)

        return pressed_btns

    def get_joypad_coordinate(self):
        result = self.getBTJoypadStatus()

        if result is None or len(result) == 0:
            # 没有获取到事件
            return []

        try:
            data = json.loads(result)
            if data is None:
                # logging.debug("warning-----no data in get_bt_joypad_status")
                return []
        except:
            # logging.error('get_joypad_coordinate json loads error')
            return []

        sticks = []
        stick = data['stick']
        sticks.append(stick['l']['x'])
        sticks.append(stick['l']['y'])
        sticks.append(stick['r']['x'])
        sticks.append(stick['r']['y'])

        return sticks
