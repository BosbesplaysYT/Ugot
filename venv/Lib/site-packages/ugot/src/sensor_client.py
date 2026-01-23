import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

try:
    from ugot.grpc_pb import sensor_pb2,sensor_pb2_grpc
except:
    pass
try:
    import sensor_pb2,sensor_pb2_grpc
except:
    pass

from base_client import GrpcClient

class SensorClient(GrpcClient):

    def __init__(self, address):
        super().__init__(address)

        self.client = sensor_pb2_grpc.SensorServiceGrpcStub(channel=self.channel)

    def getDistanceSensorValue(self, sensor_id):
        input_data = sensor_pb2.DistanceSensorRequest()
        input_data.deviceId = str(sensor_id)

        response = self.client.getDistanceSensorValue(input_data)
        if response.code != 0:
            response.data.value = -1
        return response.data

    def getIMUSensorValue(self):
        input_data = sensor_pb2.SensorCommonRequest()

        response = self.client.getIMUSensorValue(input_data)
        return response.data

    def getAttitudeTilt(self):
        # 加速度
        input_data = sensor_pb2.AttitudeTiltRequest()

        response = self.client.getAttitudeTilt(input_data)

        return response