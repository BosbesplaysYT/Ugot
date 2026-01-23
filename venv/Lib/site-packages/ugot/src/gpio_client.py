import logging
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

from base_client import GrpcClient

try:
    from ugot.grpc_pb import gpio_pb2, gpio_pb2_grpc
except:
    pass
try:
    import gpio_pb2, gpio_pb2_grpc
except:
    pass



class GpioClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = gpio_pb2_grpc.GpioServiceGrpcStub(channel=self.channel)

    def setGpioExport(self, pin, value):
        input_data = gpio_pb2.SetGpioExportRequest()
        input_data.pin = pin
        input_data.value = value

        response = self.client.setGpioExport(input_data)
        return response

    def readGpio(self, pin):
        input_data = gpio_pb2.ReadGpioRequest()
        input_data.pin = pin

        response = self.client.readGpio(input_data)
        return response

    def setGpioStartExportPwm(self, pin, duty_cycle):
        input_data = gpio_pb2.SetGpioStartExportPwmRequest()
        input_data.pin = pin
        input_data.frequency = 490  # 频率，默认为490Hz
        input_data.duty_cycle = int(duty_cycle)
        input_data.range = 255  # 范围，目前需求是255
        response = self.client.setGpioStartExportPwm(input_data)
        return response
    
    def setGpioStartExportPwmWithDutyCycle(self, pin, duty_cycle, frequency = 490): #频率默认未490Hz
        input_data = gpio_pb2.SetGpioStartExportPwmWithDutyCycleRequest()
        input_data.pin = pin
        input_data.frequency = frequency #频率，默认为490Hz
        input_data.duty_cycle = duty_cycle # 0 ~ 100.0 的值，范围（0.0 ～ 100.0）
        input_data.range = 100
        response = self.client.setGpioStartExportPwmWithDutyCycle(input_data)
        return response

    def setGpioStopExportPwm(self, pin):
        input_data = gpio_pb2.SetGpioStopExportPwmRequest()
        input_data.pin = pin

        response = self.client.setGpioStopExportPwm(input_data)
        return response

    # serial
    def setSerbaud(self, baudrate, port=''):
        input_data = gpio_pb2.SetSerbaudRequest()
        input_data.port = port
        input_data.baudrate = int(baudrate)

        response = self.client.setSerbaud(input_data)
        return response

    def serialExportString(self, value, port=''):
        input_data = gpio_pb2.SerialExportStringRequest()
        input_data.port = port
        input_data.value = str(value)

        response = self.client.serialExportString(input_data)
        return response

    def serialReadByte(self, port=''):
        input_data = gpio_pb2.SerialReadByteRequest()
        input_data.port = port

        response = self.client.serialReadByte(input_data)
        return response

    def serialReadString(self, port=''):
        input_data = gpio_pb2.SerialReadStringRequest()
        input_data.port = port

        response = self.client.serialReadString(input_data)
        return response

    def serialReadUtil(self, char_, port=''):
        input_data = gpio_pb2.SerialReadUtilRequest()
        input_data.port = port
        input_data.char_ = str(char_)

        response = self.client.serialReadUtil(input_data)
        return response

    def clearAllGpioAndSerial(self):
        input_data = gpio_pb2.ClearAllGpioAndSerialRequest()

        response = self.client.clearAllGpioAndSerial(input_data)
        return response

    def __del__(self):
        # 销毁时候释放串口资源
        try:
            self.clearAllGpioAndSerial()
        except Exception as e:
            # logging.debug('gpio clearAllGpioAndSerial error:')
            pass