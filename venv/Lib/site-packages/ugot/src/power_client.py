import logging
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

from base_client import GrpcClient

try:
    from ugot.grpc_pb import power_pb2, power_pb2_grpc
except:
    pass
try:
    import power_pb2, power_pb2_grpc
except:
    pass

class PowerClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = power_pb2_grpc.PowerServiceGrpcStub(channel=self.channel)

    def getPowerValue(self):
        input_data = power_pb2.PowerRequest()
        response = self.client.getPowerValue(input_data)
        return response