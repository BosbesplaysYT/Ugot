import logging
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

try:
    from ugot.grpc_pb import bluetooth_pb2, bluetooth_pb2_grpc
except:
    pass
try:
    import bluetooth_pb2, bluetooth_pb2_grpc
except:
    pass

from base_client import GrpcClient

class BlueToothClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = bluetooth_pb2_grpc.BluetoothServiceGrpcStub(channel=self.channel)

    def getBtStatus(self):
        input_data = bluetooth_pb2.BtStatusRequest()
        response = self.client.getBtStatus(input_data)
        return response