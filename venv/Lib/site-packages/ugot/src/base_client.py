import grpc
class GrpcClient:
    def __init__(self, address):
        # 舵机
        self.address = address
        # self._address = '0.0.0.0:50051'
        self._channel = None

    @property
    def channel(self):
        self._channel = grpc.insecure_channel(self.address)

        return self._channel

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, __address):
        if not len(__address):
            return
        self._address = __address

