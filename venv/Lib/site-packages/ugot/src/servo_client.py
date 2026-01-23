import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')

try:
    from ugot.grpc_pb import servo_pb2,servo_pb2_grpc
except:
    pass
try:
    import servo_pb2,servo_pb2_grpc
except:
    pass

from base_client import GrpcClient

JOINT_TYPE_SERVO = 3
JOINT_TYPE_MOTOR = 4

class ServoClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = servo_pb2_grpc.ServoServiceGrpcStub(channel=self.channel)

    def controlSingleClamp(self, status):
        # 控制夹手
        input_data = servo_pb2.ControlSingleClampRequest()
        input_data.status = status

        response = self.client.controlSingleClamp(input_data)
        return response

    def getClampStatus(self):
        # 获取夹手状态
        input_data = servo_pb2.GetClampStatusRequest()

        response = self.client.getClampStatus(input_data)
        return response.status

    def roboticArmRestory(self):
        # 机械臂复位
        input_data = servo_pb2.RoboticArmRestoryRequest()

        response = self.client.roboticArmRestory(input_data)
        return response

    def roboticArmMoveToTargetPostion(self, r, h, theta, time):
        # 机械臂移动到指定的位置
        input_data = servo_pb2.RoboticArmMovePostionRequest()
        input_data.params.r = r
        input_data.params.h = h
        input_data.params.theta = theta
        input_data.params.time = int(time / 20)
        response = self.client.roboticArmMoveToTargetPostion(input_data)
        return response

    def roboticArmSetJointPosition(self, params, type=0):
        # 设置机械臂关节
        # params: [{"joint": 1, "position" :80, "time":1000}, ..]:
        input_data = servo_pb2.RoboticArmSetJointPositionRequest()
        for data in params:
            item = input_data.params.add()
            item.joint = data["joint"]
            item.position = data["position"]
            item.time = int(data["time"] / 20)
            item.type = type  # 0控制舵机角度1控制关节角度

        response = self.client.roboticArmSetJointPosition(input_data)
        return response

    def roboticArmGetJoints(self):
        # 获取机械臂关节
        input_data = servo_pb2.RoboticArmGetJointsRequest()

        response = self.client.roboticArmGetJoints(input_data)
        return response
    
    def setServoRotateBySpeed(self, deviceId, speed, type = JOINT_TYPE_SERVO):
        # 转动舵机
        input_data = servo_pb2.ServoRotateBySpeedRequest()

        ids = []
        if isinstance(deviceId, list):
            ids += deviceId
        elif isinstance(deviceId, str):
            ids.append(deviceId)
        elif  isinstance(deviceId, int):
            ids.append(str(deviceId))

        for item in ids:
            rotate = input_data.servo_rotate.add()
            rotate.deviceId = str(item)
            rotate.speed = speed
            rotate.dev = type

        response = self.client.setServoRotateBySpeed(input_data)
        return response

    def setServoRotateByAngle(self, deviceId, angle, duration, type = JOINT_TYPE_SERVO):
        # 转动舵机
        input_data = servo_pb2.ServoRotateByAngleRequest()

        ids = []
        if isinstance(deviceId, list):
            ids += deviceId
        elif isinstance(deviceId, str):
            ids.append(deviceId)
        elif  isinstance(deviceId, int):
            ids.append(str(deviceId))

        for item in ids:
            rotate = input_data.servo_rotate.add()
            rotate.deviceId = str(item)
            rotate.angle = angle
            rotate.duration = int(duration / 20)
            rotate.dev = type

        response = self.client.setServoRotateByAngle(input_data)
        return response
    
    def setServoRotateByAngleList(self, angles, duration, type = JOINT_TYPE_SERVO):
        # 转动舵机
        input_data = servo_pb2.ServoRotateByAngleRequest()


        for device_id, angle in angles.items():
            rotate = input_data.servo_rotate.add()
            rotate.deviceId = str(device_id)
            rotate.angle = angle
            rotate.duration = int(duration / 20)
            rotate.dev = type

        response = self.client.setServoRotateByAngle(input_data)
        return response

    def getServoAngle(self, deviceIds, type = JOINT_TYPE_SERVO):
        input_data = servo_pb2.ServoGetAngleRequest()
        ids = []
        if isinstance(deviceIds, list):
            ids += deviceIds
        elif isinstance(deviceIds, str):
            ids.append(deviceIds)
        elif isinstance(deviceIds, int):
            ids.append(str(deviceIds))

        for item in ids:
            rotate = input_data.angle_info.add()
            rotate.deviceId = str(item)
            rotate.dev = type

        response = self.client.getServoAngle(input_data)
        return response.angle_list

    def stopServoRotate(self, deviceIds, mode, type = JOINT_TYPE_SERVO):
        # 停止舵机
        input_data = servo_pb2.StopServoRotateRequest()

        ids = []
        if isinstance(deviceIds, list):
            ids += deviceIds
        elif isinstance(deviceIds, str):
            ids.append(deviceIds)
        elif  isinstance(deviceIds, int):
            ids.append(str(deviceIds))

        for item in ids:
            rotate = input_data.servo_list.add()
            rotate.deviceId = str(item)
            rotate.mode = mode # 强锁位
            rotate.dev = type
        input_data.is_all = False

        response = self.client.stopServoRotate(input_data)
        return response
    
    def getMotionInfo(self, deviceIds, type = JOINT_TYPE_MOTOR):
        # 获取运动信息
        input_data = servo_pb2.GetMotionInfoRequest()

        ids = []
        if isinstance(deviceIds, list):
            ids += deviceIds
        elif isinstance(deviceIds, str):
            ids.append(deviceIds)
        elif isinstance(deviceIds, int):
            ids.append(str(deviceIds))

        for item in ids:
            rotate = input_data.motion_info.add()
            rotate.deviceId = str(item)
            rotate.dev = type

        response = self.client.getMotionInfo(input_data)
        return response.motion_list
    
    def stopAllServos(self):
        # 停止所有外设
        input_data = servo_pb2.StopServoRotateRequest()
        input_data.is_all = True

        response = self.client.stopServoRotate(input_data)
        return response
    
    def __del__(self):
        # 销毁时候停止模型
        try:
            self.stopAllServos()
        except Exception as e:
            # logging.debug('servo stopAllServos error:')
            pass