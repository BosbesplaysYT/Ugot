from ugot.src.util import num_normal, Color
from ugot.src.enum import E_Model, E_Device, E_Vision, E_Audio
from ugot.src.model_client import ModelClient, MODEL_TYPE_MECANUM, MODEL_TYPE_BALANCE, MODEL_TYPE_TRANSFORM, MODEL_TYPE_WHEELLEG, MODEL_TYPE_SPIDER, MODEL_TYPE_DOG
from ugot.src.vision_client import VisionClient
from ugot.src.scan_device import DeviceScan
from ugot.src.scan_device import DEV_LIST
from ugot.src.device_client import DeviceClient
from ugot.src.audio_client import AudioClient
from ugot.src.sensor_client import SensorClient
from ugot.src.network_client import NetworkClient
from ugot.src.power_client import PowerClient
from ugot.src.bluetooth_client import BlueToothClient
from ugot.src.servo_client import ServoClient, JOINT_TYPE_MOTOR, JOINT_TYPE_SERVO
from ugot.src.gpio_client import GpioClient
from ugot.src.http_client import upload_vision_picture
from ugot.src.PID import PID

import logging
import json
import threading
import time
import sys
import base64
import os

LANGUAGE_CONFIGURE = None # 语言配置

def _get_translation(key):
    global LANGUAGE_CONFIGURE
    return LANGUAGE_CONFIGURE.get_value_for_key(key)

class UGOT:


    def __init__(self):
        self.http_basic_url = ''
        self.current_mode = ''
        pass

    def __scan(self):
        self.SCAN = DeviceScan()
        self.SCAN.device_discovery()

    

    def scan_device(self):
        """Scan for devices

        Search and print UGOT devices on the same local network.

        Args:
            None

        Returns:
            name_list (dict): Format: {"Device Name 1": "IP Address 1", "Device Name 2": "IP Address 2", ...}
            
        """
        t = threading.Thread(target= self.__scan)
        t.start()
        t.join()
        return DEV_LIST

    def initialize(self, device_ip = None):
        """Initialize a device

        Initialize the relevant device using its IP address.

        Args:
            device_ip (str): IP address of the device as a string

        Returns:
            None

        """
        device_ip = str(device_ip)
        if device_ip is None or len(device_ip) == 0:
            # 如果没传或者传空字符串，默认初始化本机
            device_ip = '0.0.0.0'
        if len(device_ip) == 0:
            print("please input device ip !")
            return
        address = device_ip + ':50051'
        print(address)
        self.__initialize_modules(address)

        self.__initialize_http_client(device_ip)

        self.__configure_language()

    def __initialize_modules(self, address):

        self.MODEL = ModelClient(address)
        self.VISION = VisionClient(address)
        self.DEVICE = DeviceClient(address)
        self.AUDIO = AudioClient(address)
        self.SENSOR = SensorClient(address)
        self.NETWORK = NetworkClient(address)
        self.POWER = PowerClient(address)
        self.BLUETOOTH = BlueToothClient(address)
        self.SERVO = ServoClient(address)
        self.GPIO = GpioClient(address)

    def __initialize_http_client(self, device_ip):
        self.http_basic_url = 'http://' + device_ip + ':7000/'

    def __configure_language(self):
        response = self.DEVICE.getLanguage()
        cur_language = 'en' # 默认为英文
        if response.code == 0:
            lang = response.lang.lower()
            if lang == "china":
                cur_language = "zh_CN"
            elif lang == "korea":
                cur_language = "ko"
            elif lang == "global":
                cur_language = "en"

        global LANGUAGE_CONFIGURE

        from ugot.src.configure_string_single import ConfigureStringSingle
        configure_file_path = os.path.dirname(os.path.realpath(__file__)) + "/src/locale"
        LANGUAGE_CONFIGURE = ConfigureStringSingle(configure_file_path, 'ugot', language=[cur_language])


    """

    >>>>>>>>>>>
    >> Servo & Motor <<
    >>>>>>>>>>>

    """
    def turn_servo_angle(self, id, angle, duration_ms, wait = False):
        """
        Control a servo to rotate by a specified angle over a duration.

        Args:
            id (int/list[int]): Servo ID/ID list
            angle (int): [-180, 180] Angle to rotate, in degrees
            duration_ms (int): Runtime [20, 5000] in milliseconds
            wait (bool, optional): Block until completion, default is False (non-blocking)

        Returns:
            None

        """
        angle = num_normal(angle, 180, -180)
        duration_ms = num_normal(duration_ms, 5000, 20)

        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return

        self.SERVO.setServoRotateByAngle(id, angle, duration_ms)
        if wait:
            time.sleep(duration_ms / 1000.0)

    def turn_servo_speed(self, id, speed):
        """
        Control a servo to rotate at a constant speed of 360 degrees.

        Args:
            id (int/list[int]): Servo ID/ID list
            speed (int): Rotation speed [-100, 100]

        Returns:
            None

        """

        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        speed = num_normal(speed, 100, -100)

        self.SERVO.setServoRotateBySpeed(id, speed)

    def turn_servo_speed_times(self, id, speed, times):
        """
        Control a servo to rotate at a constant speed for xx seconds.

        Args:
            id (int/list[int]): Servo ID/ID list
            speed (int): Rotation speed [-100, 100]
            times (int): Rotation duration [1, 9999] in seconds

        Returns:
            None

        """

        speed = num_normal(speed, 100, -100)
        times = num_normal(times, 9999, 1)

        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return

        self.SERVO.setServoRotateBySpeed(id, speed)
        if times > 0:
            time.sleep(times)
            self.stop_servo(id)

    def read_servo_angle(self, id):
        """
        Read servo angle(s).

        Args:
            id (int/list[int]): Servo ID/ID list

        Returns:
            angle_list (dict): Format: {"Servo ID 1": "Angle of Servo 1", "Servo ID 2": "Angle of Servo 2", ...}

        """

        result = {}

        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return

        angle_list = self.SERVO.getServoAngle(id)
        if angle_list is not None:
            for info in angle_list:
                result[info.deviceId] = info.angle
                # if info.deviceId == str(id):
                #     return info.angle
            return result
        return {}

    def stop_servo(self, id, lock = False):
        """
        Stop a servo.

        Args:
            id (int/list[int]): Servo ID/ID list
            lock (bool): The state of servo after stop, True for lock, False for unlock, default is False
        Returns:
            None

        """
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        if lock:
            self.SERVO.stopServoRotate(id, mode=1) # mode=1表示强锁位
        else:
            self.SERVO.stopServoRotate(id, mode=0) # mode=0表示弱锁位
        

    def turn_motor_speed(self, id, speed):
        """
        Control a motor to rotate at a constant speed of 360 degrees.

        Args:
            id (int/list[int]): Motor ID/Motor ID list
            speed (int): Rotation speed [-140, 140]

        Returns:
            None

        """
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        speed = num_normal(speed, 140, -140)
        self.SERVO.setServoRotateBySpeed(id, speed, JOINT_TYPE_MOTOR)

    def turn_motor_speed_times(self, id, speed, times):
        """
        Control a motor to rotate at a constant speed for a specified duration.

        Args:
            id (int/list[int]): Motor ID/Motor ID list
            speed (int): Rotation speed [-140, 140]
            times (int): Rotation duration [1, 9999] in seconds

        Returns:
            None

        """
        
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        speed = num_normal(speed, 140, -140)
        times = num_normal(times, 9999, 1)
        self.SERVO.setServoRotateBySpeed(id, speed, type = JOINT_TYPE_MOTOR)
        if times > 0:
            time.sleep(times)
            self.stop_motor(id)

    def read_motor_speed(self, id):
        """
        Read motor speed(s).

        Args:
            id (int/list[int]): Motor ID/Motor ID list

        Returns:
            speed_list (dict): Format: {"Motor ID 1": "Speed of Motor 1", "Motor ID 2": "Speed of Motor 2", ...}

        """
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        result = {}
        motion_list = self.SERVO.getMotionInfo(id)
        if motion_list is not None:
            for info in motion_list:
                result[info.deviceId] = info.speed
                # result.append({'deviceId':info.deviceId, 'speed': info.speed})
                # if info.deviceId == str(id):
                #     logging.debug('read servo id:{} `s speed is {}'.format(id, info.speed))
                #     return info.speed
            return result
        return result
    
    def turn_motor_angle(self, id, angle, duration_ms, wait = False):
        """Control a motor to rotate by a specified angle over a duration.

        Args:
            id (int/list[int]): Motor ID/ID list
            angle (int): [-180, 180] Angle to rotate, in degrees
            duration_ms (int): Runtime [20, 5000] in milliseconds
            wait (bool, optional): Block until completion, default is False (non-blocking)

        Returns:
            None

        """
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return

        angle = num_normal(angle, 180, -180)
        duration_ms = num_normal(duration_ms, 5000, 20)

        self.SERVO.setServoRotateByAngle(id, angle, duration_ms, type=JOINT_TYPE_MOTOR)
        if wait:
            time.sleep(duration_ms / 1000.0)

    def read_motor_angle(self, id):
        """
        Read motor angle(s).

        Args:
            id (int/list[int]): Motor ID/ID list

        Returns:
            angle_list (dict): Format: {"Motor ID 1": "Angle of Motor 1", "Motor ID 2": "Angle of Motor 2", ...}

        """
        result = {}

        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return

        angle_list = self.SERVO.getServoAngle(id, type=JOINT_TYPE_MOTOR)
        if angle_list is not None:
            for info in angle_list:
                result[info.deviceId] = info.angle
                # if info.deviceId == str(id):
                #     return info.angle
            return result
        return {}

    def stop_motor(self, id, lock = False):
        """
        Stop a motor.

        Args:
            id (int/list[int]): Motor ID/Motor ID list
            lock (bool): The state of motor after stop, True for lock, False for unlock, default is False
        Returns:
            None

        """
        func_name = sys._getframe().f_code.co_name
        if not self.__validate_servo_id(id,func_name):
            return
        
        if lock:
            self.SERVO.stopServoRotate(id, mode=1, type = JOINT_TYPE_MOTOR) # mode=1表示强锁位
        else:
            self.SERVO.stopServoRotate(id, mode=0, type = JOINT_TYPE_MOTOR) # mode=0表示强锁位

    def stop_all_servos(self):
        """
        Stop all motors and servos.

        Args:
            None

        Returns:
            None

        """
        self.SERVO.stopAllServos() 
    
    def __validate_servo_id(self, id, func_name):
        if not isinstance(id, int) and not isinstance(id, list):
            typestr = 'int or list is required (got type {})'.format(type(id).__name__)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return False
        return True

    def stop_chassis(self):
        """
        Stop the chassis motion.

        Args:
            None

        Returns:
            None

        """
        self.MODEL.stopAllModels()

    def perform_action(self, actionId):
        """
        
        Perform a specific action
        Args:
            actionId (str): Action name, options available (WakeUp, Smile, Doubt, Resist, Love, Anger, Proud, Ticklish, Sleep)

        Returns:
            无
        """
        if not len(self.current_mode):
            self.current_mode = self.DEVICE.getDeviceModel()
            
        if self.current_mode == 'box':
            # 主控为box时，就不发送命令了
            return
        
        # 执行通用动作前，先停止运控，避免动作冲突
        # 平衡车/轮足不需要停止运控
        if not (self.current_mode == 'balance' or self.current_mode == 'wheel_legged'):
            self.MODEL.stop('all')

        if self.current_mode == 'engineer':
            # 底层没有处理工程车按照麦轮车处理
            self.MODEL.performAction('mecanum', actionId)
        else:
            self.MODEL.performAction(self.current_mode, actionId)

    def model_common_move(self, speed, turn_speed):
        """
        Motion universal control interface

        Args:
            speed(int): Linear speed: positive number means moving forward, negative number means going backward
            turn_speed(int): Angular velocity: positive number means turn left, negative number means turn right

        Returns:
            无
        """
        if not len(self.current_mode):
            self.current_mode = self.DEVICE.getDeviceModel()
        angle = 0
        linear_speed = abs(speed)
        if speed >= 0:
            angle = 0
        else:
            angle = 180
        if self.current_mode == 'box':
            # 主控为box时，就不发送命令了
            return
        if self.current_mode == 'engineer':
            # 底层没有处理工程车按照麦轮车处理
            linear_speed, turn_speed = self.__limit_speed_by_model(self.current_mode, linear_speed, turn_speed)
            self.MODEL.model_move_control(model_type='mecanum',linear_speed=linear_speed, direction=angle, rotate_speed=turn_speed)
        else:
            linear_speed, turn_speed = self.__limit_speed_by_model(self.current_mode, linear_speed, turn_speed)
            self.MODEL.model_move_control(model_type=self.current_mode,linear_speed=linear_speed, direction=angle, rotate_speed=turn_speed)

    def __limit_speed_by_model(self, model, speed, turn_speed):
        max_linear_speed = 0
        max_turn_speed = 0
        if model == E_Model.ChassisMode.mecanum or model == E_Model.ChassisMode.engineer:
            max_linear_speed = 80
            max_turn_speed = 280
        elif model == E_Model.ChassisMode.balance:
            max_linear_speed = 80
            max_turn_speed = 360
        elif model == E_Model.ChassisMode.transform:
            max_linear_speed = 80
            max_turn_speed = 280
        elif model == E_Model.ChassisMode.wheel_legged:
            max_linear_speed = 60
            max_turn_speed = 180
        elif model == E_Model.ChassisMode.spider:
            max_linear_speed = 25
            max_turn_speed = 60
        elif model == E_Model.ChassisMode.dog:
            max_linear_speed = 25
            max_turn_speed = 20
        # 前进/后退方向只有正数，转弯方向有正负
        speed = num_normal(speed, max_linear_speed, 0)
        turn_speed = num_normal(turn_speed, max_turn_speed, -max_turn_speed)
        return speed, turn_speed
    
    """

    >>>>>>>>>>>
    >> 变形车 - Transformable Vehicle <<
    >>>>>>>>>>>

    """
    def transform_set_chassis_height(self, height: int):
        """
        Set the chassis height of a transformable vehicle.

        Args:
            height (int): [2-7] in centimeters

        Returns:
            None

        """
        if not isinstance(height, int):
            typestr = 'int is required (got type {})'.format(type(height).__name__)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        height = height * 10  # 换算成mm
        height = int(num_normal(height, 70, 20))
        self.MODEL.set_chassis_height(MODEL_TYPE_TRANSFORM, height)

    def transform_move_speed(self, direction, speed):
        """
        Move the transformable vehicle forward or backward.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Speed in centimeters per second

        Returns:
            None

        """
        speed = num_normal(speed, 80, 0)
        if direction == E_Model.Direction.forward:
            self.MODEL.transform_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.transform_move_control(linear_speed=speed, direction=180)
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)

    def transform_turn_speed(self, turn: int, speed: int):
        """
        Turn the transformable vehicle left or right.

        Args:
            turn (int): Direction (2: Left turn; 3: Right turn)
            speed (int): [5-280] Speed in degrees per second

        Returns:
            None

        """
        speed = num_normal(speed, 280, 5)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.transform_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.transform_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)

    def transform_move_speed_times(self, direction, speed, times, unit):
        """
        Control the transformable vehicle to move forward/backward for x seconds/cm and then stop.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Speed in centimeters per second
            times (int): [0-360] Duration or range
            unit (int): Unit type (0: Move in seconds; 1: Move in centimeters)

        Returns:
            None

        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 80, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.transform_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.transform_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def transform_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the transformable vehicle to rotate left/right for x seconds/degrees and then stop.

        Args:
            turn (int): Direction (2: Left turn; 3: Right turn)
            speed (int): [5-280] Speed in degrees per second
            times (int): [0-360] Duration or range
            unit (int): Unit type (0: Move in seconds; 2: Move in degrees)

        Returns:
            None

        """
        speed = num_normal(speed, 280, 5)
        tar_speed = 0
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.transform_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.transform_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def transform_move_turn(self, direction, speed, turn, turn_speed):
        """
        Control the transformable vehicle to move in a specified direction and simultaneously perform a rotation.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Forward/Backward speed in centimeters per second
            turn (int): Direction (2: Left turn; 3: Right turn)
            turn_speed (int): [5-280] Rotation speed in degrees per second

        Returns:
            None

        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        tar_speed = 0
        turn_speed = num_normal(turn_speed, 280, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 80, 0)

        self.MODEL.transform_move_control(linear_speed=speed, direction=tar_direction, rotate_speed=tar_speed)

    def transform_motor_control(self, lf, rf, lb, rb):
        """
        Control the four motors of the transformable vehicle

        Args:
            lf (int): Left front wheel speed, [-360, 360] units RPM
            rf (int): Right front wheel speed, [-360, 360] units RPM
            lb (int): Left rear wheel speed, [-360, 360] units RPM
            rb (int): Right rear wheel speed, [-360, 360] units RPM

        Returns:
            None

        """
        lf = num_normal(lf, 360, -360)
        rf = num_normal(rf, 360, -360)
        lb = num_normal(lb, 360, -360)
        rb = num_normal(rb, 360, -360)
        self.MODEL.model_motor_control(MODEL_TYPE_TRANSFORM, lf, rf, lb, rb)

    def transform_stop(self):
        """
        Stop the transformable vehicle's movement

        Returns:
            None
        """
        self.MODEL.transform_move_control(linear_speed=0, direction=0)

    def transform_restory(self):
        """
        Restore the transformable vehicle

        Returns:
            None
        """
        self.MODEL.restory(MODEL_TYPE_TRANSFORM)

    def transform_arm_control(self, joint, position, time):
        """
        Set the angles of the four arms of the transformable vehicle

        Args:
            joint (int): Arm (1: left front arm; 2: left rear arm; 3: right rear arm; 4: right front arm)
            position (int): Angle, unit in degrees
            time (int): Duration, unit in milliseconds

        Returns:
            None
        """
        if not 1 <= joint <= 4:
            typestr = 'invalid value of joint id, expected 1/2/3/4, got {}'.format(joint)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        position = num_normal(position, 180, -180)
        time = num_normal(time, 5000, 20)
        self.MODEL.transform_arm_control(joint, position, time)

    def transform_adaption_control(self, option):
        """
        Enable/disable adaptation; the transformable vehicle can adjust its posture based on different terrains

        Args:
            option (bool): Switch status, True for on, False for off

        Returns:
            None
        """
        if option:
            self.MODEL.model_adaption(MODEL_TYPE_TRANSFORM, E_Model.Adaption.ON)
        else:
            self.MODEL.model_adaption(MODEL_TYPE_TRANSFORM, E_Model.Adaption.OFF)

    """

    >>>>>>>>>>>
    >> 麦轮车 - Mecanum Wheel Vehicle <<
    >>>>>>>>>>>

    """

    """麦轮车
        """

    def mecanum_translate_speed(self, angle, speed):
        """
        Translate the mecanum-wheeled vehicle in a specific direction

        Args:
            angle (int): [-180, 180] angle unit: degrees (in the XY plane, with the Y-axis as the 0-degree direction, left [0, -180] right [0, 180])
            speed (int): [5-80] forward/backward speed, unit in centimeters/second

        Returns:
            None
        """
        angle = num_normal(angle, 180, -180)
        speed = num_normal(speed, 80, 0)
        self.MODEL.mecanum_move_control(linear_speed=speed, direction=angle)

    def mecanum_translate_speed_times(self, angle, speed, times, unit):
        """
        Translate the mecanum-wheeled vehicle in a specific direction for x seconds/cm and then stop

        Args:
            angle (int): [-180, 180] angle unit: degrees (in the XY plane, with the Y-axis as the 0-degree direction, left [0, -180] right [0, 180])
            speed (int): [5-80] speed, unit in degrees/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: motion in seconds; 1: motion in centimeters)

        Returns:
            None
        """
        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        angle = num_normal(angle, 180, -180)
        speed = num_normal(speed, 80, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=angle, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=angle, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def mecanum_move_xyz(self, x_speed, y_speed, z_speed):
        """
        Control the mecanum wheel vehicle to move continuously in the specified direction at the specified speed.

        Args:
            x_speed (int): x-axis direction speed [-80, 80]
            y_speed (int): y-axis direction speed [-80, 80]
            z_speed (int): z-axis direction speed [-280, 280]

        Returns:
            None

        """
        x_speed = num_normal(x_speed, 80, -80)
        y_speed = num_normal(y_speed, 80, -80)
        z_speed = num_normal(z_speed, 280, -280)
        self.MODEL.mecanum_xyz_control(x_speed, y_speed, z_speed)

    def mecanum_move_speed(self, direction, speed):
        """
        Move the mecanum-wheeled vehicle forward/backward

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [5-80] speed, unit in centimeters/second

        Returns:
            None
        """
        speed = num_normal(speed, 80, 0)
        if direction == E_Model.Direction.forward:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=180)
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

    def mecanum_turn_speed(self, turn, speed):
        """
        Mecanum wheel vehicle left/right movement.

        Args:
            turn (int): Direction (2: Left turn; 3: Right turn)
            speed (int): [5-280] Speed in degrees per second

        Returns:
            None

        """
        speed = num_normal(speed, 280, 5)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.mecanum_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.mecanum_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)

    def mecanum_move_speed_times(self, direction, speed, times, unit):
        """
        Control the mecanum wheel vehicle to move forward/backward for x seconds/cm and then stop.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Speed in centimeters per second
            times (int): [0-360] Duration or range
            unit (int): Unit type (0: Move in seconds; 1: Move in centimeters)

        Returns:
            None
            
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 80, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.mecanum_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def mecanum_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the mecanum wheel vehicle to rotate left/right for x seconds/degrees and then stop.

        Args:
            turn (int): Direction (2: Left turn; 3: Right turn)
            speed (int): [5-280] Speed in degrees per second
            times (int): [0-360] Duration or range
            unit (int): Unit type (0: Move in seconds; 2: Move in degrees)

        Returns:
            None

        """
        speed = num_normal(speed, 280, 5)
        tar_speed = 0
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.mecanum_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.mecanum_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def mecanum_move_turn(self, angle, speed, turn, turn_speed):
        """
        Control the mecanum wheel vehicle to move in a specified direction while rotating.

        Args:
            angle (int): [-180, 180] Angle in degrees (XY plane, with Y-axis as 0 degrees, left [0, -180] right [0, 180])
            speed (int): [5-80] Forward/backward speed in centimeters per second
            turn (int): Direction (2: Left turn; 3: Right turn)
            turn_speed (int): [5-280] Rotation speed in degrees per second

        Returns:
            None

        """
        angle = num_normal(angle, 180, -180)
        tar_speed = 0
        turn_speed = num_normal(turn_speed, 280, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 80, 0)

        self.MODEL.mecanum_move_control(linear_speed=speed, direction=angle, rotate_speed=tar_speed)

    def mecanum_motor_control(self, lf, rf, lb, rb):
        """
        Control the four motors of the mecanum-wheeled vehicle

        Args:
            lf (int): Left front wheel speed, [-360, 360] units RPM (Revolutions Per Minute)
            rf (int): Right front wheel speed, [-360, 360] units RPM
            lb (int): Left rear wheel speed, [-360, 360] units RPM
            rb (int): Right rear wheel speed, [-360, 360] units RPM

        Returns:
            None
        """
        lf = num_normal(lf, 360, -360)
        rf = num_normal(rf, 360, -360)
        lb = num_normal(lb, 360, -360)
        rb = num_normal(rb, 360, -360)
        self.MODEL.model_motor_control(MODEL_TYPE_MECANUM, lf, rf, lb, rb)

    def mecanum_stop(self):
        """
        Stop the mecanum wheel vehicle.

        Returns:
            None

        """
        self.MODEL.stop(MODEL_TYPE_MECANUM)

    """

    >>>>>>>>>>>
    >> 平衡车 - Self-Balancing Vehicle <<
    >>>>>>>>>>>

    """

    def balance_start_balancing(self):
        """
        Start the vehicle and keep it self-balanced.

        Returns:
            None
        """
        self.MODEL.model_keep_balancing(MODEL_TYPE_BALANCE, True)

    def balance_stop_balancing(self):
        """
        Stop the vehicle while maintaining self-balance.

        Returns:
            None
        """
        self.MODEL.model_keep_balancing(MODEL_TYPE_BALANCE, False)

    def balance_set_acceleration(self, acceleration):
        """
        Set the acceleration of the self-balancing vehicle.

        Args:
            acceleration (float): Acceleration

        Returns:
            None
        """

        if not (isinstance(acceleration, float) or isinstance(acceleration, int)):
            typestr = 'int is required (got type {})'.format(type(acceleration).__name__)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        if acceleration > 0:
            self.MODEL.setBalanceAcceleration(acceleration)

    def balance_reset_acceleration(self):
        """
        Reset the acceleration of the self-balancing vehicle.

        Returns:
            None
        """
        self.MODEL.resetAcceleration('all')

    def balance_move_speed(self, direction, speed):
        """
        Move the self-balancing vehicle forward/backward.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Speed in centimeters per second

        Returns:
            None
        """
        speed = num_normal(speed, 80, 0)
        if direction == E_Model.Direction.forward:
            self.MODEL.balance_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.balance_move_control(linear_speed=speed, direction=180)
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

    def balance_turn_speed(self, turn, speed):
        """
        Turn the self-balancing vehicle left or right.

        Args:
            turn (int): Direction (2: Left; 3: Right)
            speed (int): [5-360] Speed in degrees per second

        Returns:
            None
        """
        # 平衡车转动速度限制5-360
        speed = num_normal(speed, 360, 5)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.balance_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.balance_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)

    def balance_move_speed_times(self, direction, speed, times, unit):
        """
        Control the self-balancing vehicle to move forward/backward for a specified duration or distance.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Speed in centimeters per second
            times (int): [0-360] Duration or distance
            unit (int): Unit type (0: seconds; 1: centimeters)

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 80, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.balance_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.balance_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def balance_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the self-balancing vehicle to turn left or right for a specified duration or angle.

        Args:
            turn (int): Direction (2: Left; 3: Right)
            speed (int): [5-360] Speed in degrees per second
            times (int): [0-360] Duration or angle
            unit (int): Unit type (0: seconds; 2: degrees)

        Returns:
            None
        """
        tar_speed = 0
        speed = num_normal(speed, 360, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.balance_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.balance_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def balance_move_turn(self, direction, speed, turn, turn_speed):
        """
        Control the self-balancing vehicle to move in a specified direction while simultaneously rotating.

        Args:
            direction (int): Direction (0: Forward; 1: Backward)
            speed (int): [5-80] Forward/backward speed in centimeters per second
            turn (int): Direction (2: Left; 3: Right)
            turn_speed (int): [5-360] Rotational speed in degrees per second

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        tar_speed = 0
        # 平衡车转动速度限制0-360
        turn_speed = num_normal(turn_speed, 360, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 80, 0)

        self.MODEL.balance_move_control(linear_speed=speed, direction=tar_direction, rotate_speed=tar_speed)


    """

    >>>>>>>>>>>
    >> 机械臂 - Mechanical Arm <<
    >>>>>>>>>>>

    """

    def mechanical_clamp_release(self):
        """
        Open the clamp.

        Returns:
            None
        """
        self.SERVO.controlSingleClamp(0)

    def mechanical_clamp_close(self):
        """
        Close the clamp.

        Returns:
            None
        """
        self.SERVO.controlSingleClamp(1)

    def mechanical_get_clamp_status(self):
        """
        Get the clamp status.

        Returns:
            status (int): 0 for open, 1 for closed
        """
        result = self.SERVO.getClampStatus()
        return result

    def mechanical_arms_restory(self):
        """
        Reset the mechanical arm.

        Returns:
            None
        """
        self.SERVO.roboticArmRestory()

    def mechanical_joint_control(self, angle1, angle2, angle3, duration):
        """
        Control the joint angles of the mechanical arm.

        Args:
            angle1 (int): Joint 1 angle [-90, 90] degrees
            angle2 (int): Joint 2 angle [-80, 110] degrees
            angle3 (int): Joint 3 angle [-90, 90] degrees
            duration (int): Duration [20, 5000] milliseconds

        Returns:
            None
        """
        params = []
        angle1 = num_normal(angle1, 90, -90)
        angle2 = num_normal(angle2, 110, -80)
        angle3 = num_normal(angle3, 90, -90)
        duration = num_normal(duration, 5000, 20)
        params.append({'joint': 1, 'position': angle1, 'time': duration})
        params.append({'joint': 2, 'position': angle2, 'time': duration})
        params.append({'joint': 3, 'position': angle3, 'time': duration})
        self.SERVO.roboticArmSetJointPosition(params, 1)

    def mechanical_single_joint_control(self, joint, angle, duration):
        """
        Control the angle of a single joint of the mechanical arm.

        Args:
            joint (int): Joint number (1: Joint 1, 2: Joint 2, 3: Joint 3)
            angle (int): Joint angle (Joint 1: [-90, 90], Joint 2: [-80, 110], Joint 3: [-90, 90])
            duration (int): Duration [20, 5000] milliseconds

        Returns:
            None
        """
        valid_joint = True

        if joint == 1:
            angle = num_normal(angle, 90, -90)
        elif joint == 2:
            angle = num_normal(angle, 110, -80)
        elif joint == 3:
            angle = num_normal(angle, 90, -90)
        else:
            valid_joint = False

        if valid_joint:
            duration = num_normal(duration, 5000, 20)
            params = [{'joint': joint, 'position': angle, 'time': duration}]
            self.SERVO.roboticArmSetJointPosition(params, 1)
        else:
            typestr = 'invalid value of joint id, expected 1/2/3, got {}'.format(joint)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)

    def mechanical_move_axis(self, r, h, theta, duration):
        """
        Move the mechanical arm to the specified position in a coordinate system relative to the vehicle.

        Args:
            r (float/int): r [-5.5, 24.9] in cm
            h (float/int): h [-18, 18.2] in cm
            theta (float/int): theta [-1.57, 1.57] in radians
            duration (int): Duration [20, 5000] milliseconds

        Returns:
            None
        """
        error_value = None
        if not (isinstance(r, float) or isinstance(r, int)):
            error_value = r
        elif not (isinstance(h, float) or isinstance(h, int)):
            error_value = h
        elif not (isinstance(theta, float) or isinstance(theta, int)):
            error_value = theta

        if error_value is not None:
            func_name = sys._getframe().f_code.co_name
            typestr = 'int or float is required (got type {})'.format(type(error_value).__name__)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return

        r = int(r * 10)  # 换算成mm
        r = num_normal(r, 249, -55)

        h = int(h * 10)  # 换算成mm
        h = num_normal(h, 182, -180)

        theta = num_normal(theta, 1.570, -1.57)

        duration = num_normal(duration, 5000, 20)

        self.SERVO.roboticArmMoveToTargetPostion(r, h, theta, duration)

    """

        >>>>>>>>>>>
        >> 轮腿 - Wheel-legged Robot <<
        >>>>>>>>>>>

        """

    def wheelleg_start_balancing(self):
        """
        Start the robot and maintain self-balancing.

        Returns:
            None
        """
        self.MODEL.restory(MODEL_TYPE_WHEELLEG)
        # time.sleep(0.5)
        self.MODEL.model_keep_balancing(MODEL_TYPE_WHEELLEG, True)

    def wheelleg_stop_balancing(self):
        """
        Stop the robot and maintain self-balancing.

        Returns:
            None
        """
        self.MODEL.model_keep_balancing(MODEL_TYPE_WHEELLEG, False)

    def wheelleg_move_speed(self, direction, speed):
        """
        Move the wheel-legged robot forward/backward.

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [5-60] speed in cm/second

        Returns:
            None
        """
        speed = num_normal(speed, 60, 5)
        if direction == E_Model.Direction.forward:
            self.MODEL.wheelleg_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.wheelleg_move_control(linear_speed=speed, direction=180)
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

    def wheelleg_turn_speed(self, turn, speed):
        """
        Turn the wheel-legged robot left/right.

        Args:
            turn (int): Direction (2: left; 3: right)
            speed (int): [5-180] speed in degrees/second

        Returns:
            None
        """
        speed = num_normal(speed, 180, 5)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.wheelleg_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.wheelleg_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)

    def wheelleg_move_speed_times(self, direction, speed, times, unit):
        """
        Control the wheel-legged robot to move forward/backward for x seconds/cm and then stop.

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [5-60] speed in cm/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: motion in seconds; 1: motion in centimeters)

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 60, 5)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.wheelleg_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.wheelleg_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def wheelleg_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the wheel-legged robot to move left/right for x seconds/degrees and then stop.

        Args:
            turn (int): Direction (2: left; 3: right)
            speed (int): [5-180] speed in degrees/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: motion in seconds; 2: motion in degrees)

        Returns:
            None
        """
        tar_speed = 0
        speed = num_normal(speed, 180, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.wheelleg_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.wheelleg_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def wheelleg_move_turn(self, direction, speed, turn, turn_speed):
        """
        Control the wheel-legged robot to move in a specified direction and simultaneously rotate.

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [5-60] forward/backward speed in cm/second
            turn (int): Direction (2: left; 3: right)
            turn_speed (int): [5-180] rotation speed in degrees/second

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        tar_speed = 0
        turn_speed = num_normal(turn_speed, 180, 5)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 60, 5)

        self.MODEL.wheelleg_move_control(linear_speed=speed, direction=tar_direction, rotate_speed=tar_speed)

    def wheelleg_set_chassis_height(self, height):
        """
        Set the chassis height of the wheel-legged robot.

        Args:
            height (int): Height (1: high; 2: medium; 3: low)

        Returns:
            None
        """
        if not isinstance(height, int):
            typestr = 'int is required (got type {})'.format(type(height).__name__)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        if not 1 <= height <= 3:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of height, expected [1-3], got {}'.format(height)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        self.MODEL.set_chassis_height(MODEL_TYPE_WHEELLEG, height)

    def wheelleg_restory(self):
        """
        Restore the wheel-legged robot to the initial posture at medium height.

        Returns:
            None
        """
        self.MODEL.restory(MODEL_TYPE_WHEELLEG)

    def wheelleg_set_decline_angle(self, angle):
        """
        Set the left and right inclination angle of the wheel-legged robot.

        Args:
            angle (int): [-10, 10] inclination angle in degrees

        Returns:
            None
        """
        angle = num_normal(angle, 10, -10)
        # pose=0轮腿只有左右倾斜
        # Pose 0 for wheel legs only has left-right inclination.
        self.MODEL.set_decline_angle(MODEL_TYPE_WHEELLEG, 0, angle)

    def wheelleg_adaption_control(self, option):
        """
        Enable/disable adaptation; the wheel-legged robot can adjust its posture based on different terrains

        Args:
            option (bool): Switch status, True for on, False for off

        Returns:
            None
        """
        self.MODEL.model_adaption(MODEL_TYPE_WHEELLEG, option)

    """

        >>>>>>>>>>>
        >> 蜘蛛 - Spider Robot<<
        >>>>>>>>>>>

        """

    def spider_restory(self):
        """
        Restore the spider robot.

        Returns:
            None
        """
        self.MODEL.restory(MODEL_TYPE_SPIDER)

    def spider_move_speed(self, direction, speed):
        """
        Control the spider robot to move linearly (forward/backward/left translation/right translation).

        Args:
            direction (int): Direction (0: forward; 1: backward; 2: left translation; 3: right translation)
            speed (int): [0-25] speed in cm/second

        Returns:
            None
        """
        speed = num_normal(speed, 25, 0)
        if direction == E_Model.Direction.forward:
            self.MODEL.spider_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.spider_move_control(linear_speed=speed, direction=180)
        elif direction == E_Model.SDirection.translate_left:
            self.MODEL.spider_move_control(linear_speed=speed, direction=-90)
        elif direction == E_Model.SDirection.translate_right:
            self.MODEL.spider_move_control(linear_speed=speed, direction=90)
        else:
            self.__print_move_direction1_error_msg(sys._getframe().f_code.co_name, direction)
            return

    def spider_turn_speed(self, turn, speed):
        """
        Control the spider robot to rotate in place.

        Args:
            turn (int): Direction (2: turn left; 3: turn right)
            speed (int): [0-60] speed in degrees/second

        Returns:
            None
        """
        speed = num_normal(speed, 60, 0)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.spider_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.spider_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return
    def spider_move_speed_times(self, direction, speed, times, unit):
        """
        Control the spider robot to move linearly for x seconds/cm and then stop.

        Args:
            direction (int): Direction (0: forward; 1: backward; 2: left translation; 3: right translation)
            speed (int): [0-25] speed in cm/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: motion in seconds; 1: motion in centimeters)

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        elif direction == E_Model.SDirection.translate_left:
            tar_direction = -90
        elif direction == E_Model.SDirection.translate_right:
            tar_direction = 90
        else:
            self.__print_move_direction1_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 25, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.spider_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.spider_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def spider_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the spider robot to rotate for x seconds/degrees and then stop.

        Args:
            turn (int): Direction (2: turn left; 3: turn right)
            speed (int): [0-60] speed in degrees/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: motion in seconds; 2: motion in degrees)

        Returns:
            None
        """
        tar_speed = 0
        speed = num_normal(speed, 60, 0)
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.spider_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.spider_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def spider_move_turn(self, direction, speed, turn, turn_speed):
        """
        Control the spider robot to move linearly while simultaneously performing a rotation.

        Args:
            direction (int): Direction (0: forward; 1: backward; 2: left translation; 3: right translation)
            speed (int): [0-25] speed in cm/second
            turn (int): Direction (2: turn left; 3: turn right)
            turn_speed (int): [0-60] speed in degrees/second

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        elif direction == E_Model.SDirection.translate_left:
            tar_direction = -90
        elif direction == E_Model.SDirection.translate_right:
            tar_direction = 90
        else:
            self.__print_move_direction1_error_msg(sys._getframe().f_code.co_name, direction)
            return

        tar_speed = 0
        turn_speed = num_normal(turn_speed, 60, 0)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 25, 0)

        self.MODEL.spider_move_control(linear_speed=speed, direction=tar_direction, rotate_speed=tar_speed)

    def spider_stop(self):
        """
        Pause the spider robot's motion.

        Returns:
            None
        """
        self.MODEL.stop(MODEL_TYPE_SPIDER)

    """

        >>>>>>>>>>>
        >> 四足狗 - Four-legged Dog Robot <<
        >>>>>>>>>>>

        """

    def dog_restory(self):
        """
        Restore the four-legged dog robot.

        Returns:
            None
        """
        self.MODEL.restory(MODEL_TYPE_DOG)

    def dog_set_decline_angle(self, pose, angle):
        """
        Set the inclination angle of the four-legged dog robot.

        Args:
            pose (int): Inclination direction (0: left-right inclination; 1: front-back inclination)
            angle (int): [-5, 5] angle in degrees

        Returns:
            None
        """
        if not 0 <= pose <= 1:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of pose, expected 0/1, got {}'.format(pose)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        angle = num_normal(angle, 5, -5)
        self.MODEL.set_decline_angle(MODEL_TYPE_DOG, pose, angle)

    def dog_move_speed(self, direction, speed):
        """
        Control the four-legged dog robot to move forward/backward.

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [0-25] speed in cm/second

        Returns:
            None
        """
        speed = num_normal(speed, 25, 0)
        if direction == E_Model.Direction.forward:
            self.MODEL.dog_move_control(linear_speed=speed, direction=0)
        elif direction == E_Model.Direction.backward:
            self.MODEL.dog_move_control(linear_speed=speed, direction=180)
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
    def dog_turn_speed(self, turn, speed):
        """
        Control the four-legged dog robot to rotate in place.

        Args:
            turn (int): Direction (2: turn left; 3: turn right)
            speed (int): [0-20] speed in degrees/second

        Returns:
            None
        """
        speed = num_normal(speed, 20, 0)
        if turn == E_Model.Direction.turn_left:
            self.MODEL.dog_move_control(rotate_speed=speed)
        elif turn == E_Model.Direction.turn_right:
            self.MODEL.dog_move_control(rotate_speed=-speed)
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)

    def dog_move_speed_times(self, direction, speed, times, unit):
        """
        Control the four-legged dog robot to move linearly for x seconds/cm and then stop.

        Args:
            direction (int): Direction (0: forward; 1: backward)
            speed (int): [0-25] speed in cm/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: representing motion in seconds; 1: representing motion in centimeters)

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        speed = num_normal(speed, 25, 0)
        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.dog_move_control(linear_speed=speed, direction=tar_direction, time=times)
        elif unit == E_Model.Unit.mileage:
            self.MODEL.dog_move_control(linear_speed=speed, direction=tar_direction, mileage=times)
        else:
            self.__print_move_unit_m_error_msg(sys._getframe().f_code.co_name, unit)

    def dog_turn_speed_times(self, turn, speed, times, unit):
        """
        Control the four-legged dog to rotate for x seconds/degrees and then stop.

        Args:
            turn (int): Direction (2: turn left; 3: turn right)
            speed (int): [0-20] speed in degrees/second
            times (int): [0-360] duration range
            unit (int): Unit type (0: representing motion in seconds; 2: representing motion in degrees)

        Returns:
            None
        """
        tar_speed = 0
        speed = num_normal(speed, 20, 0)
        if turn == E_Model.Direction.turn_left:
            tar_speed = speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        if times == 0:
            # 时长/里程等于0，此积木块不做控制
            logging.error('Duration/range is 0; this block does not perform any control')
            return

        times = num_normal(times, 360, 0)

        if unit == E_Model.Unit.second:
            self.MODEL.dog_move_control(rotate_speed=tar_speed, time=times)
        elif unit == E_Model.Unit.angle:
            self.MODEL.dog_move_control(rotate_speed=tar_speed, target_angle=times)
        else:
            self.__print_move_unit_a_error_msg(sys._getframe().f_code.co_name, unit)

    def dog_move_turn(self, direction, speed, turn, turn_speed):
        """
        Control the four-legged dog robot to move linearly while simultaneously performing a rotation.

        Args:
            direction (int): Direction (0: forward; 1: backward; 2: left translation; 3: right translation)
            speed (int): [0-25] speed in cm/second
            turn (int): Direction (2: turn left; 3: turn right)
            turn_speed (int): [0-20] speed in degrees/second

        Returns:
            None
        """
        tar_direction = 0
        if direction == E_Model.Direction.forward:
            tar_direction = 0
        elif direction == E_Model.Direction.backward:
            tar_direction = 180
        else:
            self.__print_move_direction_error_msg(sys._getframe().f_code.co_name, direction)
            return

        tar_speed = 0
        turn_speed = num_normal(turn_speed, 20, 0)
        if turn == E_Model.Direction.turn_left:
            tar_speed = turn_speed
        elif turn == E_Model.Direction.turn_right:
            tar_speed = -turn_speed
        else:
            self.__print_move_turn_error_msg(sys._getframe().f_code.co_name, turn)
            return

        speed = num_normal(speed, 25, 0)

        self.MODEL.dog_move_control(linear_speed=speed, direction=tar_direction, rotate_speed=tar_speed)

    def dog_perform_action(self, actionId):
        """
        Control the quadrupedal dog to perform a specific action

        Args:
            actionId (str): Action name, options available (crawling, squatting, standing, handshake, urination, stretch)

        Returns:
            None
        """
        self.MODEL.performAction(MODEL_TYPE_DOG, actionId)

    def dog_stop(self):
        """
        Pause the quadrupedal dog's movement

        Returns:
            None
        """
        self.MODEL.stop(MODEL_TYPE_DOG)

    def dog_adaption_control(self, option):
        """
        Enable/disable adaptation; the quadrupedal dog can adjust its posture based on different terrains

        Args:
            option (bool): Switch status, True for on, False for off

        Returns:
            None
        """
        self.MODEL.model_adaption(MODEL_TYPE_DOG, option)

    def __print_move_direction_error_msg(self, func_name, input_value):
        typestr = 'invalid value of direction, expected 0 or 1, got {}'.format(input_value)
        error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
        print(error_msg)

    def __print_move_direction1_error_msg(self, func_name, input_value):
        typestr = 'invalid value of direction, expected 0/1/2/3, got {}'.format(input_value)
        error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
        print(error_msg)

    def __print_move_turn_error_msg(self, func_name, input_value):
        typestr = 'invalid value of turn, expected 2 or 3, got {}'.format(input_value)
        error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
        print(error_msg)

    def __print_move_unit_m_error_msg(self, func_name, input_value):
        typestr = 'invalid value of unit, expected 0 or 1, got {}'.format(input_value)
        error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
        print(error_msg)

    def __print_move_unit_a_error_msg(self, func_name, input_value):
        typestr = 'invalid value of unit, expected 0 or 2, got {}'.format(input_value)
        error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
        print(error_msg)

    """

       >>>>>>>>>>>
       >> AI视觉 - AI Vision <<
       >>>>>>>>>>>

       """
    def load_models(self, models):
        """
        Load models, multiple models can be selected.

        Args:
            models (list): List of models to load. Correspondences: Human Pose: 'human_pose', Text Recognition: 'word_recognition',
                            Color Recognition: 'color_recognition', AprilTag/QR Code: 'apriltag_qrcode',
                            Emotion Recognition/Face Attributes: 'face_attribute', License Plate Detection: 'lpd_recognition', Gesture Recognition: 'gesture',
                            Traffic Sign Recognition: 'traffic_sign', Face Recognition: 'face_recognition', Single Rail/Double Rail Recognition: 'line_recognition',
                            Toy Recognition: 'toy_recognition'

        Returns:
            True if loading is successful, else False
        """
        if not len(models):
            typestr = 'models is empty!'
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return False
        elif not isinstance(models, list):
            typestr = 'list is required (got type {})'.format(type(models).__name__)
            func_name = sys._getframe().f_code.co_name
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return False
        return self.VISION.load_models(models)

    def release_models(self, models = None):
        """
        Unload models.

        Args:
            models (list): List of models to unload, parameters are the same as those in the 'load_models' method. Default is None; if not provided, all models will be unloaded.

        Returns:
            None
        """
        self.VISION.release_models(models)

    def get_qrcode_apriltag_total_info(self):
        """
        Get QR Code and AprilTag information.

        Args:
            None

        Returns:
            QR Code and AprilTag recognition result (list) [qrcode, id, center_x, center_y, height, width, area, distance5, distance7, distance10, x, y, z]:
            - qrcode (str): QR Code content
            - id (int): AprilTag ID
            - center_x (float): AprilTag center x-coordinate
            - center_y (float): AprilTag center y-coordinate
            - height (float): AprilTag height
            - width (float): AprilTag width
            - area (float): AprilTag area
            - distance5 (float): AprilTag distance (5x5 cm)
            - distance7 (float): AprilTag distance (7x7 cm)
            - distance10 (float): AprilTag distance (10x10 cm)
            - x (float): AprilTag card pose angle x
            - y (float): AprilTag card pose angle y
            - z (float): AprilTag card pose angle z
        """
        result = self.VISION.qrcode_inference()

        qrcode = ''
        id = center_x = center_y = height = width = area = dis5 = dis7 = dis10 = x = y = z = -1

        if result is not None:
            data = json.loads(result)
            if data is not None:
                if 'qrcode' in data:
                    qrcode_list = data["qrcode"]

                    max_item = None
                    max_area = 0
                    for item in qrcode_list:
                        position = item['position']
                        t_area = position[2] * position[3]
                        if t_area > max_area:
                            max_area = t_area
                            max_item = item
                    if max_item is not None:
                        qrcode = max_item['data']
                if 'apriltag' in data:
                    apriltag_list = data["apriltag"]
                    max_item = None
                    max_area = 0
                    for item in apriltag_list:
                        position = item['position']
                        t_area = position[2] * position[3]
                        if t_area > max_area:
                            max_area = t_area
                            max_item = item
                    if max_item is not None:
                        position = max_item['position']
                        id = max_item['id']
                        center_x = round(max_item['center'][0], 2)
                        center_y = round(max_item['center'][1], 2)
                        height = round(position[3], 2)
                        width = round(position[2], 2)
                        area = round(position[3] * position[2], 2)
                        dis5 = round(max_item['instance'][0], 2)
                        dis7 = round(max_item['instance'][1], 2)
                        dis10 = round(max_item['instance'][2], 2)
                        x = round(max_item['pose_ypr'][0], 2)
                        y = round(max_item['pose_ypr'][1], 2)
                        z = round(max_item['pose_ypr'][2], 2)

        result = [qrcode, id, center_x, center_y, height, width, area, dis5, dis7, dis10, x, y, z]
        return result

    def get_qrcode_total_info(self):
        """
        Get QR Code information

        Args:
            None

        Returns:
            QR Code recognition result (list) [ [qrcode, center_x, center_y, height, width, area], ... ]:
            - qrcode (str): QR Code content
            - center_x (float): QR Code center x-coordinate
            - center_y (float): QR Code center y-coordinate
            - height (float): QR Code height
            - width (float): QR Code width
            - area (float): QR Code area
        """
        response = self.VISION.qrcode_inference()

        result = []
        if response is not None:
            data = json.loads(response)
            if data is not None:
                if 'qrcode' in data:
                    qrcode_list = data["qrcode"]
                    for item in qrcode_list:
                        qrcode = item['data']
                        position = item['position']
                        center_x = position[0] + position[2] / 2
                        center_y = position[1] + position[3] / 2
                        width = position[2]
                        height = position[3]
                        area = position[2] * position[3]

                        info = [qrcode, center_x, center_y, height, width, area]
                        result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(5), reverse=True)
        except Exception as e:
            logging.debug('get_qrcode_total_info exception:', e)
        logging.debug('-------get_qrcode_total_info `s result is {}'.format(result))
        return result

    def get_apriltag_total_info(self):
        """
        Get AprilTag information

        Args:
            None

        Returns:
            AprilTag recognition result (list) [ [id, center_x, center_y, height, width, area, distance5, distance7, distance10, x, y, z, bearingAngle_h, bearingAngle_v], ... ]:
            - id (int): AprilTag ID
            - center_x (float): AprilTag center x-coordinate
            - center_y (float): AprilTag center y-coordinate
            - height (float): AprilTag height
            - width (float): AprilTag width
            - area (float): AprilTag area
            - distance5 (float): AprilTag distance (5x5 cm)
            - distance7 (float): AprilTag distance (7x7 cm)
            - distance10 (float): AprilTag distance (10x10 cm)
            - x (float): AprilTag card pose angle x
            - y (float): AprilTag card pose angle y
            - z (float): AprilTag card pose angle z
            - bearingAngle_h (float): AprilTag card horizontal bearing angle
            - bearingAngle_v (float): AprilTag card vertical bearing angle
        """
        response = self.VISION.qrcode_inference()

        result = []
        if response is not None:
            data = json.loads(response)
            if data is not None:
                if 'apriltag' in data:
                    apriltag_list = data["apriltag"]
                    for item in apriltag_list:
                        position = item['position']
                        id = item['id']
                        center_x = round(item['center'][0], 2)
                        center_y = round(item['center'][1], 2)
                        height = round(position[3], 2)
                        width = round(position[2], 2)
                        area = round(position[3] * position[2], 2)
                        dis5 = round(item['instance'][0], 2)
                        dis7 = round(item['instance'][1], 2)
                        dis10 = round(item['instance'][2], 2)
                        x = round(item['pose_ypr'][0], 2)
                        y = round(item['pose_ypr'][1], 2)
                        z = round(item['pose_ypr'][2], 2)
                        bearingAngle_h = round(item['bearingAngle'][0], 3)
                        bearingAngle_v = round(item['bearingAngle'][1], 3)
                        info = [id, center_x, center_y, height, width, area, dis5, dis7, dis10, x, y, z, bearingAngle_h, bearingAngle_v]
                        result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(5), reverse=True)
        except Exception as e:
            logging.debug('get_apriltag_total_info exception:', e)

        logging.debug('-------get_apriltag_total_info `s result is {}'.format(result))
        return result

    def get_license_plate_total_info(self):
        """
        Get license plate information

        Args:
            None

        Returns:
            License plate recognition result (list) [ [number, type, center_x, center_y, height, width, area], ... ]:
            - number (str): License plate number
            - type (str): License plate type (blue plate/green plate)
            - center_x (float): Center x-coordinate
            - center_y (float): Center y-coordinate
            - height (float): Height
            - width (float): Width
            - area (float): Area
        """

        response = self.VISION.license_plate_inference()

        result = []

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                for item in data:
                    number = item['lp']
                    type = item['type']
                    if type == E_Vision.LPD.blue:
                        typestr = _get_translation('licensePlate.blue')
                    elif type == E_Vision.LPD.green:
                        typestr = _get_translation('licensePlate.green')
                    position = item['position']
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2
                    width = position[2]
                    height = position[3]
                    area = position[2] * position[3]
                    info = [number, typestr, center_x, center_y, height, width, area]
                    result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(6), reverse=True)
        except Exception as e:
            logging.debug('get_license_plate_total_info exception:', e)
        logging.debug("get_license_plate_total_info `s result  is :{}".format(result))
        return result

    def get_pose_total_info(self):
        """
        Get the coordinates of recognized human keypoints

        Args:
            None

        Returns:
            Pose recognition result (list) [ [Right Ear x, y, Right Eye x, y, Nose x, y, Left Eye x, y, Left Ear x, y, Right Hand x, y, Right Elbow x, y, Right Shoulder x, y, Left Shoulder x, y, Left Elbow x, y, Left Hand x, y, Right Hip x, y, Left Hip x, y, Right Knee x, y, Left Knee x, y, Right Foot x, y, Left Foot x, y, ], ... ]
        """
        response = self.VISION.pose_identify()

        result = []

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                if len(data) > 0:
                    for item in data:
                        pose = item['keypoint']
                        info = []
                        for idx in E_Vision.Pose.TotalPoseIndexes:
                            x = y = 0
                            if idx < len(pose):
                                points = pose[idx]
                                x = int(round(points['x'], 0))
                                y = int(round(points['y'], 0))
                                x = max(0, x)  # 小于0按照0处理
                                y = max(0, y)  # 小于0按照0处理

                            info.extend([x, y])
                        result.append(info)
        logging.debug("get_pose_total_info `s result  is :{}".format(result))
        return result

    """交通识别
        """

    def get_traffic_total_info(self):
        """
        Get traffic sign recognition results

        Args:
            None

        Returns:
        Traffic sign recognition result (list) [ [sign, center_x, center_y, height, width, area], ... ]:
        - sign (str): Traffic sign (green light, horn, left turn, right turn, zebra crossing, red light, children, no long-time parking, enter tunnel, yellow light)
        - center_x (float): Center x-coordinate
        - center_y (float): Center y-coordinate
        - height (float): Height
        - width (float): Width
        - area (float): Area
        """

        response = self.VISION.traffic_inference()

        result = []

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                for item in data:
                    label = item["label"]
                    if label == E_Vision.Traffic.green_light:
                        sign = _get_translation('traffic.greenLight')
                    elif label == E_Vision.Traffic.horn:
                        sign = _get_translation('traffic.whistle')
                    elif label == E_Vision.Traffic.left:
                        sign = _get_translation('traffic.turnLeft')
                    elif label == E_Vision.Traffic.right:
                        sign = _get_translation('traffic.turnRight')
                    elif label == E_Vision.Traffic.zebra_crossing:
                        sign = _get_translation('traffic.zebraCrossing')
                    elif label == E_Vision.Traffic.red_light:
                        sign = _get_translation('traffic.redLight')
                    elif label == E_Vision.Traffic.children:
                        sign = _get_translation('traffic.children')
                    elif label == E_Vision.Traffic.stop:
                        sign = _get_translation('traffic.noPark')
                    elif label == E_Vision.Traffic.tunnel:
                        sign = _get_translation('traffic.tunnel')
                    elif label == E_Vision.Traffic.yellow_light:
                        sign = _get_translation('traffic.amberLight')

                    position = item['position']
                    center_x = round(position[0] + position[2] / 2, 2)
                    center_y = round(position[1] + position[3] / 2, 2)
                    width = round(position[2], 2)
                    height = round(position[3], 2)
                    area = round(position[2] * position[3], 2)
                    info = [sign, center_x, center_y, height, width, area]
                    result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(5), reverse=True)
        except Exception as e:
            logging.debug('get_traffic_total_info exception:', e)
        logging.debug("get_traffic_total_info `s result is :{}".format(result))
        return result

    def get_face_recognition_total_info(self):
        """
        Get face recognition results

        Args:
            None

        Returns:
        Face recognition result (list) [ [name, center_x, center_y, height, width, area], ... ]:
        - name (str): Name (or "Unknown" for unrecognized faces)
        - center_x (float): Center x-coordinate
        - center_y (float): Center y-coordinate
        - height (float): Height
        - width (float): Width
        - area (float): Area
        """
        response = self.VISION.face_recognition_inference()
        # {'emotion': 2, 'gender': 0, 'mask': 1, 'position': [0, 14, 103, 199]}
        # mask:0——带口罩，1——未戴口罩，2——口罩没有正确佩戴
        result = []

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                for item in data:
                    name = item['name']
                    if name == '':  # 有返回人脸，但是是没录入的，认为是陌生人
                        name = _get_translation('face.stranger')
                    position = item['position']
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2
                    width = position[2]
                    height = position[3]
                    area = position[2] * position[3]
                    info = [name, center_x, center_y, height, width, area]
                    result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(5), reverse=True)
        except Exception as e:
            logging.debug('get_face_recognition_total_info exception:', e)
        logging.debug("get_face_recognition_total_info `s result  is :{}".format(result))
        return result
    
    # def open_camera(self):
    #     self.VISION.openCamera()

    def face_recognition_get_all_names(self):
        """
        Get a list of all registered face names

        Args:
            None

        Returns:
            List of face names (list): [name1, name2, ...]
        """
        names = self.VISION.face_recognition_get_all_names()
        return names

    def face_recognition_delete_name(self, name):
        """
        Delete a registered face

        Args:
            name (str): Face name to delete

        Returns:
            None
        """
        response = self.VISION.face_recognition_delete_name(name)

    def face_recognition_add_name(self, name):
        """
        Register a face

        Args:
            name (str): Face name to register

        Returns:
            None
        """

        names = self.VISION.face_recognition_get_all_names()
        for i in names:
            if i == name:
                print('Face [{}] already exists'.format(name))
                return None
        
        self.load_models(['face_recognition'])

        self.open_camera()
        time.sleep(0.2)

        for i in range(3, 0, -1):
            print('{} seconds until taking a photo'.format(i))
            time.sleep(1)

        print('Taking a photo...')

        response = self.VISION.readCameraData()
        if not (response.code == 0 and response.nDataLen > 0):
            print('No camera data received')
            return None
        

        pdata = response.pdata

        import base64, os
        img = base64.b64decode(pdata)

        # tmp_img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'tmp_image')
        tmp_img_dir = os.path.dirname(os.path.realpath(__file__))

        if not os.path.exists(tmp_img_dir):
            os.system('mkdir -p ' + tmp_img_dir)

        image_name = '{}.jpg'.format(name)

        file_path = os.path.join(tmp_img_dir,image_name)

        with open(file_path,"wb") as fh:
            fh.write(img)

        upload_response = upload_vision_picture(self.http_basic_url, file_path)
        if not upload_response["code"] == 0:
            print('Failed to upload the image')
            return None

        response = self.VISION.face_recognition_insert_data(image_name, name)
        if response is not None:
            
            if response.code == 0:
                data = response.data
                data = json.loads(data)
                print('Face [{}] registered successfully\n'.format(name))
            else:
                msg = response.msg
                print('Face not found\n')

        # from IPython.display import display
        # from PIL import Image
        # show_image = Image.open(file_path)
        # display(show_image)

    def get_face_characteristic_total_info(self):
        """
        Get face characteristic recognition results

        Args:
            None

        Returns:
        Face characteristic recognition result (list) [ [gender, mask_info, emotion, center_x, center_y, height, width, area], ... ]:
        - gender (str): Gender
        - mask_info (str): Mask information
        - emotion (str): Emotion
        - center_x (float): Center x-coordinate
        - center_y (float): Center y-coordinate
        - height (float): Height
        - width (float): Width
        - area (float): Area
        """
        response = self.VISION.face_characteristic_inference()

        result = []
        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                for item in data:
                    # 统计性别&口罩数量
                    gender = item["gender"]
                    if gender == 0:
                        gender_str = _get_translation('face.man')
                    elif gender == 1:
                        gender_str = _get_translation('face.woman')

                    mask = item["mask"]
                    if mask == 0:
                        mask_str = _get_translation('face.wearMask')
                    elif mask == 1:
                        mask_str = _get_translation('face.notProperlyMask')
                    elif mask == 2:
                        mask_str = _get_translation('face.noMask')
                    emotion = item["emotion"]
                    if emotion == 0:  # '生气'
                        emotion_str = _get_translation('emotion.angry')
                    elif emotion == 1:  # '开心'
                        emotion_str = _get_translation('emotion.happy')
                    elif emotion == 2:  # '平静'
                        emotion_str = _get_translation('emotion.silence')
                    elif emotion == 3:  # '惊讶'
                        emotion_str = _get_translation('emotion.amaze')

                    position = item['position']
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2
                    width = position[2]
                    height = position[3]
                    area = position[2] * position[3]

                    face = [gender_str, mask_str, emotion_str, center_x, center_y, height, width, area]
                    result.append(face)

        try:
            import operator
            result.sort(key=operator.itemgetter(7), reverse=True)
        except Exception as e:
            logging.debug('get_face_characteristic_total_info exception:', e)
        logging.debug("get_face_characteristic_total_info `s result  is :{}".format(result))
        return result

    def __judge_track_line_type(self, data):
        cross_type = E_Vision.Intersection.noline  # 线类型
        line_type = E_Vision.LineType.single  # 单双轨类型
        max_item = None

        # 当没有检测到线、也没有检测到标志的时候，表示——无线
        # 当没有检测到标志，但是有线的时候，表示——一条线

        # 判断单轨还是双轨
        if "single_start" in data and "single_end" in data:
            line_type = E_Vision.LineType.single
        elif "double_start" in data and "double_end" in data:
            line_type = E_Vision.LineType.double

        # 先判断是不是有标志牌
        has_sign = False
        if "traffic_sign" in data:
            traffic_sign = data["traffic_sign"]
            if not isinstance(traffic_sign, list) or not len(traffic_sign):
                # 标志数组为空，按没有标识处理
                has_sign = False

            max_area = 0
            for item in traffic_sign:
                position = item["position"]
                area = position[2] * position[3]
                if area > max_area:
                    max_area = area
                    max_item = item
            if max_item is not None:
                has_sign = True

        if has_sign:
            # 有标志牌，按照标志牌里的路口
            if "label" in max_item:
                label = max_item["label"]
                # 有标志，按标志路口走
                if label == 0:
                    cross_type = E_Vision.Intersection.crossroad
                elif label == 1:
                    cross_type = E_Vision.Intersection.ycross
        else:
            # 没有标志，判断线类型
            if "single_start" in data and "single_end" in data:
                single_start = data["single_start"]
                single_end = data["single_end"]
                if single_start[0] == -1 and single_start[1] == -1 and single_end[0] == -1 and single_end[1] == -1:
                    cross_type = E_Vision.Intersection.noline
                else:
                    cross_type = E_Vision.Intersection.straight
            elif "double_start" in data and "double_end" in data:
                double_start = data["double_start"]
                double_end = data["double_end"]
                if double_start[0] == -1 and double_start[1] == -1 and double_end[0] == -1 and double_end[1] == -1:
                    cross_type = E_Vision.Intersection.noline
                else:
                    cross_type = E_Vision.Intersection.straight

        return cross_type, line_type, max_item

    def set_track_recognition_line(self, line_type):
        """
        Set the current recognized lane line type

        Args:
            line_type (int): 0: Single track, 1: Double track

        Returns:
            None
        """
        if not (line_type == 0 or line_type == 1):
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of line_type, expected 0 or 1, got {}'.format(line_type)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        self.VISION.set_track_color_line(0, line_type)

    def get_single_track_total_info(self):
        """
        Get single track recognition results

        Args:
            None

        Returns:
            Single track recognition result (list) [offset, type, x, y]:
            - offset (int): Single track offset
            - type (int): Single track line type (1: Straight, 2: Y-intersection, 3: Crossroad, 0: No Line)
            - x (float): Intersection coordinate x
            - y (float): Intersection coordinate y
        """
        logging.debug("get_single_track_total_info ")

        offset = type = center_x = center_y = 0

        response = self.VISION.track_recognition_inference()

        if response is not None:
            data = json.loads(response)
            if data is not None:
                cross_type, line_type, max_item = self.__judge_track_line_type(data)

                # 路口类型
                if line_type == E_Vision.LineType.single:
                    if cross_type == E_Vision.Intersection.straight:  # 1 直线
                        type = 1
                    elif cross_type == E_Vision.Intersection.ycross:  # 2 y字路口
                        type = 2
                    elif cross_type == E_Vision.Intersection.crossroad:  # 3 十字路口
                        type = 3
                    else:
                        type = 0  # 其余情况默认为无线
                else:
                    type = 0  # 轨道类型不一致，返回无线

                # 坐标
                if max_item is not None:
                    position = max_item["position"]
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2

                # 偏移量
                if 'offset' in data:
                    offset = data['offset']

        result = [offset, type, center_x, center_y]

        logging.debug(' get_single_track_total_info `s result is :{}'.format(result))

        return result

    def get_double_track_total_info(self):
        """
        Get double track recognition results

        Args:
            None

        Returns:
            Double track recognition result (list) [offset, type, x, y]:
            - offset (int): Double track offset
            - type (int): Double track line type (1: Straight, 2: Intersection, 0: No Line)
            - x (float): Intersection coordinate x
            - y (float): Intersection coordinate y
        """

        offset = type = center_x = center_y = 0

        response = self.VISION.track_recognition_inference()
        if response is not None:
            data = json.loads(response)
            if data is not None:
                cross_type, line_type, max_item = self.__judge_track_line_type(data)

                # 路口类型
                if line_type == E_Vision.LineType.double:
                    if cross_type == E_Vision.Intersection.straight:  # 1 直线
                        type = 1
                    # y字路口和十字路口都返回2
                    elif cross_type == E_Vision.Intersection.ycross:  # 2 y字路口
                        type = 2
                    elif cross_type == E_Vision.Intersection.crossroad:  # 3 十字路口
                        type = 2
                    else:
                        type = 0  # 其余情况默认为无线
                else:
                    type = 0  # 轨道类型不一致，返回无线

                # 坐标
                if max_item is not None:
                    position = max_item["position"]
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2

                # 偏移量
                if 'offset' in data:
                    offset = data['offset']

        result = [offset, type, center_x, center_y]

        logging.debug(' get_double_track_total_info `s result is :{}'.format(result))

        return result

    def get_color_total_info(self):
        """
        Get color recognition results

        Args:
            None

        Returns:
            Color recognition result (list) [color, shape, center_x, center_y, height, width, area]:
            - color (str): Color
            - shape (str): Shape (Ball/Square)
            - center_x (float): Center point x
            - center_y (float): Center point y
            - height (float): Height
            - width (float): Width
            - area (float): Area
        """
        response = self.VISION.color_identify()

        colorstr = typestr = ''
        center_x = center_y = height = width = area = -1

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                max_item = None
                max_area = 0
                for item in data:
                    position = item['position']
                    t_area = position[2] * position[3]
                    if t_area > max_area:
                        max_area = t_area
                        max_item = item

                # [{'area': 34581, 'color': 'red', 'shape': 'ball', "position" : [ 119, 272, 164, 143 ]}]
                if max_item is not None:
                    color = max_item['color']
                    colorstr = _get_translation(color)
                    shape = max_item['shape']
                    typestr = _get_translation(shape)
                    
                    position = max_item['position']
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2
                    width = position[2]
                    height = position[3]
                    area = position[2] * position[3]

        result = [colorstr, typestr, center_x, center_y, height, width, area]
        logging.debug("get_color_total_info `s result  is :{}".format(result))
        return result

    def get_words_result(self):
        """
        Get text recognition results

        Args:
            None

        Returns:
            Text recognition result (str)
        """
        result = self.VISION.word_identify()

        empty_ret = ''

        if result is not None:
            data = json.loads(result)
            if not 'words' in data:
                return empty_ret
            words = data["words"]
            if not isinstance(words, list):
                return empty_ret
            if len(words) > 0:
                ret = ','.join(words)
                logging.debug("get_words_result result is :{}".format(ret))
                return ret
        return empty_ret

    def get_gesture_result(self):
        """
        Get gesture recognition results

        Args:
            None

        Returns:
            Gesture recognition result (str): (Rock/Scissors/Paper/OK/Thumbs Up)
        """
        result = self.VISION.gesture_inference()
        ret = ''

        if result is not None:
            data = json.loads(result)
            if data is None:
                return ret
            gesture = data["gesture"]
            if gesture == 'stone':
                ret = _get_translation('gesture.rock')
            elif gesture == 'scissor':
                ret = _get_translation('gesture.scissors')
            elif gesture == 'palm':
                ret = _get_translation('gesture.cloth')
            elif gesture == 'ok':
                ret = _get_translation('gesture.ok')
            elif gesture == 'good':
                ret = _get_translation('gesture.likes')
            else:
                ret = ''
            logging.debug('-------get_gesture_result is {}'.format(ret))
            return ret
            # {'confidence': 0.92, 'gesture': 'palm', 'position': [121, 352, 365, 244]]}
        return ret
    
    def get_knn_result(self, model_name):
        """
        Get local training recognition results

        Args:
            model_name (str): The name of the trained model

        Returns:
            Local training recognition result (dict): ["Category1": Confidence of Category1, "Category2": Confidence of Category2, ...]
        """
        result = {}
        response = self.VISION.knn_identify(model_name)
        
        if response is not None:
            retList = json.loads(response)
            for data in retList:
                # 类别不能重复，直接作为key
                result[data["label_name"]] = int(round(data["confidence"], 2) * 100)
        return result
    
    def knn_train(self, model_name, label_list):
        """
        Train KNN model

        Args:
            model_name (str): KNN model name
            label_list (list): label list to train

        Returns:
            Result(bool): True or False

        """

        func_name = sys._getframe().f_code.co_name
        if not isinstance(label_list, list):
            typestr = 'list is required (got type {})'.format(type(label_list).__name__)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return False
        else:
            response = self.VISION.knn_train(model_name, label_list)
            for item in response:
                code = item.code
                if code == 0:
                    data = item.data
                    data = json.loads(data)
                    status = data["status"]
                    if status == 0:
                        print('knn training in progress')
                    elif status == 2:
                        print('knn training completed')
                        return True
                else:
                    print('knn_train: {}'.format(item.msg))
                    return False

    def knn_rename(self, src_name, dst_name):
        """
        Rename the trained KNN model

        Args:
            src_name (str): Name before modification
            dst_name (str): Name after modification

        Returns:
            Result(bool): True or False
        """
        response = self.VISION.knn_rename(src_name, dst_name)
        if response.code == 0:
            return True
        return False

    def knn_delete(self, model_name):
        """
        Delete the trained KNN model

        Args:
            model_name (str): model name

        Returns:
            Result(bool): True or False
        """
        response = self.VISION.knn_delete(model_name)
        if response.code == 0:
            return True
        return False

    def knn_query(self):
        """
        Get the list of trained KNN models

        Args:
            None

        Returns:
            model_list (dict): Format: { "model_name1": ["label11", "label12", ..], "model_name2": ["label21", "label22", ..] ...}, Returns None for no trained models!
        """
        response = self.VISION.knn_query()
        if response.code == 0:
            data = response.data
            data = json.loads(data)
            if data is None:
                return None
            result = {}
            for info in data:
                model_name = info["model_name"]
                result[model_name] = info["label_list"]
            return result

        return None
    
    def get_toy_total_info(self):
        """
        Get toy information

        Args:
            None

        Returns:
            Toy recognition result (list) [ [label, center_x, center_y, height, width, area], ... ]:
            - label (str): Toy label
            - center_x (float): Center x-coordinate
            - center_y (float): Center y-coordinate
            - height (float): Height
            - width (float): Width
            - area (float): Area
        """

        response = self.VISION.toy_recognition_inference()

        result = []

        if response is not None:
            data = json.loads(response)
            if data is not None and isinstance(data, list):
                for item in data:
                    label = item['label']
                    if label == E_Vision.Toy.YOUYOU:
                        labelstr = _get_translation('toy.youyou')
                    elif label == E_Vision.Toy.WALKERX:
                        labelstr = _get_translation('toy.walkerx')
                    elif label == E_Vision.Toy.WALKER:
                        labelstr = _get_translation('toy.walker')
                    position = item['position']
                    center_x = position[0] + position[2] / 2
                    center_y = position[1] + position[3] / 2
                    width = position[2]
                    height = position[3]
                    area = position[2] * position[3]
                    info = [labelstr, center_x, center_y, height, width, area]
                    result.append(info)
        try:
            import operator
            result.sort(key=operator.itemgetter(5), reverse=True)
        except Exception as e:
            logging.debug('get_toy_total_info exception:', e)
        logging.debug("get_toy_total_info `s result  is :{}".format(result))
        return result

    # def start_auto_inference(self, models):
    #     response = self.VISION.startAutoInference(models)
    #     for item in response:
    #         logging.debug('111111',item)
    #         yield item

    """

    >>>>>>>>>>>
    >> 语音 <<
    >>>>>>>>>>>

    """

    def play_sound(self, data, wait=False):
        """
        Play built-in sound effects

        Args:
            data (str): Content to be played
                    Animal category: bear, bird, chicken, cow, dog, elephant, giraffe, horse, lion, monkey, pig, rhinoceros, sealions, tiger, walrus
                    Command category: complete, cover, move, received, support, transfiguration, yes
                    Emotion category: happy, yawn, snoring, surprise, acting cute, angry, fail, lose, doubt, nonsense, cheerful, come and play, flexin, london bridge, yankee doodle
                    Machine category: ambulance, busy tone, car horn, car horn1, doorbell, engine, laser, meebot, police car 1, police car 2, ringtones, robot, telephone call, touch tone, wave
            wait (bool, optional): Whether to block, default False, non-blocking

        Returns:
            None
        """
        if not wait:
            thread = threading.Thread(target=self._play_audio_file, args=(data, E_Audio.Type.DEFAULT,))
            thread.setDaemon(True)
            thread.start()
            time.sleep(0.1)
        else:
            self.AUDIO.playAudioFile(data, E_Audio.Type.DEFAULT)

    def play_sound_upload(self, data, wait=False):
        """
        Play uploaded audio

        Args:
            data (str): Content to play, needs to include the file type extension
            wait (bool, optional): Whether to block, default is False, not blocking

        Returns:
            None
        """
        if not wait:
            thread = threading.Thread(target=self._play_audio_file, args=(data, E_Audio.Type.UPLOAD,))
            thread.setDaemon(True)
            thread.start()
            time.sleep(0.1)
        else:
            self.AUDIO.playAudioFile(data, E_Audio.Type.UPLOAD)

    def play_record(self, data, wait=False):
        """
        Play a recorded audio

        Args:
            data (string): Recorded audio to play
            wait (bool, optional): Whether to block, default is False, not blocking

        Returns:
            None
        """

        if not wait:
            thread = threading.Thread(target=self._play_audio_file, args=(data, E_Audio.Type.RECORD,))
            thread.setDaemon(True)
            thread.start()
            time.sleep(0.1)
        else:
            self.AUDIO.playAudioFile(data, E_Audio.Type.RECORD)

    def play_tone(self, tone, beat, wait=False):
        """
        Play a tone

        Args:
            tone (str): Tone (C5, D5, E5, F5, G5, A5, B5, C6)
            beat (int): Beat (0: 1/8 beat, 1: 1/4 beat, 2: 1/2 beat, 3: 1 beat, 4: 2 beats)

        Returns:
            None
        """
        if beat in E_Audio.Beat.BEATS:
            data = tone.upper() + '-' + E_Audio.Beat.BEATS[beat]
        else:
            return
        self.play_sound(data, wait)

    def _play_audio_file(self, data, type):
        self.AUDIO.playAudioFile(data, type)

    def start_audio_asr(self):
        """
        Start listening

        Returns:
            Recognized speech content (str)
        """

        result = ''
        response = self.AUDIO.setAudioAsr()
        data = response.data
        if response.code == 0:
            if data:
                result = data.strip()
        return result
    
    def start_audio_asr_doa(self, duration = 60):
        """
        Start listening asr and direction

        Args:
            duration (int): maximum listening time (2-60)

        Returns:
            Recognized result (list): [direction, content]
                    :direction (str): Audio source direction (Left/Right/Front/Back)
                    :content (str): Recognized speech content 
        """

        duration = num_normal(duration, 2, 60)
        begin_vad = 3000
        end_vad = 1500
        if duration <= 3:
            begin_vad = 1000
            end_vad = 1000
        elif 3 < duration <= 4:
            begin_vad = 1500
            end_vad = 1500
        else:
            begin_vad = 3000
            end_vad = 1500
        result = self.AUDIO.getAsrAndDoa(begin_vad = begin_vad, end_vad = end_vad, duration = duration * 1000)
        asr_result = ''
        direction = ''

        if result is not None:
            data = json.loads(result)
            if data is None:
                # logging.debug("warning-----no data in apriltag_inference")
                pass
            
            if 'asr' in data:
                asr = data['asr']
                asr_result = asr['msg'].strip()
            
            if 'doa' in data:
                doa = data['doa']
                angle = doa['angle']
                if 45 <= angle < 135:
                    direction = _get_translation('aisound.front')
                elif 135 <= angle < 225:
                    direction = _get_translation('aisound.left')
                elif 225 <= angle < 315:
                    direction = _get_translation('aisound.behind')
                elif angle >= 315 or angle < 45:
                    direction = _get_translation('aisound.right')

        ret = [direction, asr_result]
        return ret

    def start_audio_nlp(self, data, wait=False):
        """
        Listen to speech and respond with NLP

        Args:
            data (string): Question
            wait (bool, optional): True to block and wait, default is False, not blocking

        Returns:
            None
        """

        if not wait:
            thread = threading.Thread(target=self._play_nlp, args=(data,))
            thread.setDaemon(True)
            thread.start()
            time.sleep(1)
        else:
            self._play_nlp(data)

    def _play_nlp(self, data):
        result = self.AUDIO.setAudioNlp(data)
        data = result.data
        if data:
            self.play_audio_tts(data, 0, True)

    def play_audio_tts(self, data, voice_type = 0, wait=False):
        """
        Play TTS audio

        Args:
            data (string): Content to play
            voice_type (int): Voice type (0: female, 1: male), default is 0 (female)
            wait (bool, optional): Whether to block, default is False, not blocking

        Returns:
            None
        """

        if not (voice_type == 0 or voice_type == 1):
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of voice_type, expected 0 or 1, got {}'.format(voice_type)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return

        if not wait:
            thread = threading.Thread(target=self._play_tts, args=(data, voice_type,))
            thread.setDaemon(True)
            thread.start()
        else:
            self.AUDIO.setAudioTts(data, voice_type)

    def _play_tts(self, data, voice_type):
        self.AUDIO.setAudioTts(data, voice_type)

    def set_volume(self, volume):
        """
        Set volume

        Args:
            volume (int): Volume (0-100)

        Returns:
            None
        """
        tar_volume = num_normal(volume, 100, 0)
        self.DEVICE.setVolume(tar_volume)

    def get_volume(self):
        """
        Get volume

        Returns:
            Volume information (list): [volume, isMute]
                    :volume(int) Volume value 0-100
                    :isMute(bool) Whether it is muted
        """

        response = self.DEVICE.getVolume()
        return [response.volume, response.isMute]


    def stop_audio(self):
        """
        Stop audio playback

        Returns:
            None
        """
        self.AUDIO.stopPlayAudio()

    def enable_audio_direction(self):
        """
        Enable audio direction localization

        Returns:
            None
        """
        self.AUDIO.enableAudioDirection()

    def disable_audio_direction(self):
        """
        Disable audio direction localization

        Returns:
            None
        """
        self.AUDIO.disableAudioDirection()

    def get_audio_direction(self):
        """
        Get audio direction localization

        Returns:
            Audio source direction (str): (Left/Right/Front/Back)
        """

        result = self.AUDIO.getDirectionOfAudio()
        if result.code == 0:
            angle = result.angle
            if angle == -1:  # -1表示没有声音或者无效
                return ''
            ret = ''
            if 45 <= angle < 135:
                ret = _get_translation('aisound.front')
            elif 135 <= angle < 225:
                ret = _get_translation('aisound.left')
            elif 225 <= angle < 315:
                ret = _get_translation('aisound.behind')
            elif angle >= 315 or angle < 45:
                ret = _get_translation('aisound.right')
            return ret
        return ''

    """
    
    >>>>>>>>>>>
    >> 显示屏 - Screen <<
    >>>>>>>>>>>

    """

    def screen_display_background(self, color):
        """
        Main control display screen shows the background color

        Args:
            color (int): [0-8] 0 Black; 1 White; 2 Purple; 3 Red; 4 Orange; 5 Yellow; 6 Green; 7 Cyan; 8 Blue

        Returns:
            None
        """
        if color in E_Device.Color.ColorList:
            dest_color = E_Device.Color.ColorList[color]
            self.DEVICE.display_background(dest_color)
        else:
            logging.error(' unsupported color of {}'.format(color))
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of color, expected [0-8], got {}'.format(color)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return

    def screen_print_text(self, text, color):
        """
        Main control display screen prints text

        Args:
            text (string): Text to be printed
            color (int): [0-8] 0 Black; 1 White; 2 Purple; 3 Red; 4 Orange; 5 Yellow; 6 Green; 7 Cyan; 8 Blue

        Returns:
            None
        """
        if color in E_Device.Color.ColorList:
            dest_color = E_Device.Color.ColorList[color]
            self.DEVICE.print_text(text, dest_color)
        else:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of color, expected [0-8], got {}'.format(color)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return

    def screen_print_text_newline(self, text, color):
        """
        Main control display screen prints text and moves to the next line

        Args:
            text (string): Text to be printed
            color (int): [0-8] 0 Black; 1 White; 2 Purple; 3 Red; 4 Orange; 5 Yellow; 6 Green; 7 Cyan; 8 Blue

        Returns:
            None
        """
        if color in E_Device.Color.ColorList:
            dest_color = E_Device.Color.ColorList[color]
            self.DEVICE.print_text_newline(text, dest_color)
        else:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of color, expected [0-8], got {}'.format(color)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return

    # def screen_display_image(self, image_name):
    #     self.DEVICE.display_image(image_name)

    def screen_display_emotion(self, name, loop=False):
        """
        Main control display emotion

        Args:
            name (string): emotion name, options available (WakeUp, Smile, Doubt, Search, Breathe, Blink, Resist, Love, Anger, Proud, Ticklish, Weakness, Sleep, SleepCirculate, Switch)
            loop (bool): Whether to loop, True for loop, False for not loop (by default)
        Returns:
            None
        """
        self.DEVICE.display_emotion(name, loop)

    def screen_clear(self):
        """
        Clear the main control display screen

        Returns:
            None
        """
        self.DEVICE.clear_screen()

    """
    
    >>>>>>>>>>>
    >> 灯光 <<
    >>>>>>>>>>>

    """

    def show_light_rgb(self, lights, red, green, blue):
        """
        Main control light strips display RGB colors

        Args:
            lights (list): List of light strips [0 - 3] 0: Top light strip 1: Left light strip 2: Right light strip 3: Bottom light strip
            red (int): [0-255] R value
            green (int): [0-255] G value
            blue (int): [0-255] B value

        Returns:
            None
        """

        for id in lights:
            if not 0 <= id <= 3:
                func_name = sys._getframe().f_code.co_name
                typestr = 'invalid value of lights id , expected [0-3], got {}'.format(id)
                error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
                print(error_msg)
                return

        color = Color.create_color_rgb(red, green, blue)

        top_light_color = -1
        left_light_color = -1
        right_light_color = -1
        down_light_color = -1
        if E_Device.Light.ID.TOP in lights:
            top_light_color = color.color
        if E_Device.Light.ID.LEFT in lights:
            left_light_color = color.color
        if E_Device.Light.ID.RIGHT in lights:
            right_light_color = color.color
        if E_Device.Light.ID.DOWN in lights:
            down_light_color = color.color

        self.DEVICE.showLightColor(top_light_color, left_light_color, right_light_color, down_light_color)

    def show_light_hsv(self, lights, hue_percent, saturation, value):
        """
        Main control light strips display colors based on hue, saturation, and brightness

        Args:
            lights (list): List of light strips [0 - 3] 0: Top light strip 1: Left light strip 2: Right light strip 3: Bottom light strip
            hue_percent (int): [0-100] Hue
            saturation (int): [0-100] Saturation
            value (int): [0-100] Brightness

        Returns:
            None
        """
        for id in lights:
            if not 0 <= id <= 3:
                func_name = sys._getframe().f_code.co_name
                typestr = 'invalid value of lights id , expected [0-3], got {}'.format(id)
                error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
                print(error_msg)
                return

        color = Color.create_color_hsv(hue_percent, saturation, value)

        top_light_color = -1
        left_light_color = -1
        right_light_color = -1
        down_light_color = -1
        if E_Device.Light.ID.TOP in lights:
            top_light_color = color.color
        if E_Device.Light.ID.LEFT in lights:
            left_light_color = color.color
        if E_Device.Light.ID.RIGHT in lights:
            right_light_color = color.color
        if E_Device.Light.ID.DOWN in lights:
            down_light_color = color.color

        self.DEVICE.showLightColor(top_light_color, left_light_color, right_light_color, down_light_color)

    def show_light_rgb_effect(self, red, green, blue, effect):
        """
        Display the main control LED strip effect (in RGB)

        Args:
            red(int): [0-255] R value
            green(int): [0-255] G value
            blue(int): [0-255] B value
            effect(int): [0-3] LED effect type 0: steady 1: off 2: breathing 3: flashing

        Returns:
            None
        """
        color = Color.create_color_rgb(red, green, blue)
        if not 0 <= effect <= 3:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of effect, expected [0-3], got {}'.format(effect)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        self.DEVICE.showLightEffect(color.color, effect)

    def show_light_hsv_effect(self, hue_percent, saturation, value, effect):
        """
        Display light bars on the main control light bars with effects (based on color, saturation, and brightness)

        Args:
            hue_percent (int): [0-100] Color
            saturation (int): [0-100] Saturation
            value (int): [0-100] Brightness
            effect (int): [0-3] Light effect type 0: Always on 1: Off 2: Breathing 3: Flashing

        Returns:
            None

        """
        color = Color.create_color_hsv(hue_percent, saturation, value)
        if not 0 <= effect <= 3:
            func_name = sys._getframe().f_code.co_name
            typestr = 'invalid value of effect, expected [0-3], got {}'.format(effect)
            error_msg = 'TypeError: {}(): {}'.format(func_name, typestr)
            print(error_msg)
            return
        self.DEVICE.showLightEffect(color.color, effect)

    def turn_off_lights(self):
        """
        Turn off all lights

        Returns:
            None
        """
        self.DEVICE.turnOffAllLights()

    """

    >>>>>>>>>>>
    >> 传感器 - Sensor <<        
    >>>>>>>>>>>

    """

    def read_distance_data(self, id):
        """
        Read sensor data

        Args:
            id (int): Sensor ID

        Returns:
            Distance (int) in centimeters, or -1 if no data is obtained
        """
        logging.debug('read_distance_sensor id:{}'.format(id))

        data = self.SENSOR.getDistanceSensorValue(id)
        if data is not None:
            if data.deviceId == str(id):
                logging.debug('-----获取传感器 {} 的距离为:{}'.format(id, data.value))
                if data.value == -1: # -1表示未获取到
                    return -1
                # 返回值单位是毫米，转换成厘米
                return data.value / 10
        return 0

    def read_gyro_data(self):
        """
        Read gyroscope data

        Returns:
            Gyroscope data (list): [pitch, roll, yaw, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
                    :pitch (float): Pitch angle
                    :roll (float): Roll angle
                    :yaw (float): Yaw angle
                    :gyro_x (float): Angular velocity x
                    :gyro_y (float): Angular velocity y
                    :gyro_z (float): Angular velocity z
                    :accel_x (float): Acceleration x
                    :accel_y (float): Acceleration y
                    :accel_z (float): Acceleration z
        """

        data = self.SENSOR.getIMUSensorValue()

        pitch = roll = yaw = gyro_x = gyro_y = gyro_z = accel_x = accel_y = accel_z = 0
        if data is not None:
            pitch = round(data.pitch, 2)
            roll = round(data.roll, 2)
            yaw = round(data.yaw, 2)
            gyro_x = round(data.gyro_x, 2)
            gyro_y = round(data.gyro_y, 2)
            gyro_z = round(data.gyro_z, 2)
            accel_x = round(data.accel_x, 2)
            accel_y = round(data.accel_y, 2)
            accel_z = round(data.accel_z, 2)

        result = [pitch, roll, yaw, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
        return result

    def get_attitude(self):
        """
        Get the main control attitude

        Returns:
            Attitude data (list): [is_lean_forward, is_lean_backward, is_lean_left, is_lean_right, is_screen_up, is_screen_down, is_shake]
                    :is_lean_forward (bool): Whether leaning forward
                    :is_lean_backward (bool): Whether leaning backward
                    :is_lean_left (bool): Whether leaning left
                    :is_lean_right (bool): Whether leaning right
                    :is_screen_up (bool): Whether the screen is up
                    :is_screen_down (bool): Whether the screen is down
                    :is_shake (bool): Whether shaking
        """
        data = self.SENSOR.getAttitudeTilt()
        lean_forward = lean_backward = lean_left = lean_right = screen_up = screen_down = shake = False
        if data is not None:
            lean_forward = data.tilt_forward
            lean_backward = data.tilt_back
            lean_left = data.tilt_left
            lean_right = data.tilt_right
            screen_up = data.face_up
            screen_down = data.face_down
            shake = data.shaking
        result = [lean_forward, lean_backward, lean_left, lean_right, screen_up, screen_down, shake]

        return result

    """

    >>>>>>>>>>>
    >> 物联网 - Broadcast <<        
    >>>>>>>>>>>

    """

    def enable_broadcast(self):
        """
        Enable local network broadcasting

        Returns:
            None
        """
        self.NETWORK.setBrocastEnable('1')

    def disable_broadcast(self):
        """
        Disable local network broadcasting

        Returns:
            None
        """
        self.NETWORK.setBrocastEnable('0')

    def send_broadcast_message(self, message):
        """
        Send a local network broadcast message

        Args:
            message (str): Message to send

        Returns:
            None
        """
        s = str(message)
        if s is None:
            s = ''
        self.NETWORK.sendBrocastMsg(s)

    def set_broadcast_channel(self, channel):
        """
        Set the local network broadcast channel

        Args:
            channel (int): Channel, in the range [0-99]

        Returns:
            None
        """
        self.NETWORK.setBrocastPort(num_normal(channel, 99, 0))

    def get_broadcast_message(self):
        """
        Get the received local network broadcast message content

        Returns:
            Message content (str)
        """
        result = self.NETWORK.getReceivedBrocastMsg()
        return result

    def get_wifi_status(self):
        """
        Get the current Wi-Fi connection information

        Returns:
            Wi-Fi information (list): [ssid, ip_address, mac, rssi, key_mgmt]
                :ssid (str): SSID
                :ip_address (str): IP address
                :mac (str): MAC address
                :rssi (str): Signal strength
                :key_mgmt (str): Encryption method
        """
        response = self.NETWORK.getWifiStatus()
        if response.code == 0:
            data = response.data
            return [data.ssid, data.key_mgmt, data.mac, data.rssi, data.key_mgmt]
        return ['', '', '', '', '']

    """

    >>>>>>>>>>>
    >> 电池 <<        
    >>>>>>>>>>>

    """
    def get_power_status(self):
        """
        Get battery status

        Returns:
            Status (list): [scale, power_plug, status]
                :scale (int): Battery percentage
                :power_plug (bool): Whether the power plug is inserted
                :status (int): Status (0 Normal, 1 Low battery, 2 Charging, 3 Full, 4 Abnormal)
        """
        response = self.POWER.getPowerValue()
        if response.code == 0:
            data = response.data
            return [data.scale, data.power_plug, data.status]
        return None

    """

    >>>>>>>>>>>
    >> 蓝牙 <<        
    >>>>>>>>>>>

    """

    def get_bluetooth_status(self):
        """
        Get current Bluetooth information

        Returns:
            Bluetooth status information (list): [poweron, connected, remote_device_name, remote_device_address]
                :poweron (bool): Whether Bluetooth is turned on
                :connected (bool): Whether a Bluetooth peripheral is connected
                :remote_device_name (str): Peripheral device name
                :remote_device_address (str): Peripheral device MAC address
        """
        response = self.BLUETOOTH.getBtStatus()
        if response.code == 0:
            remoteDevice = response.remoteDevice
            return [response.poweron, response.connected, remoteDevice.name, remoteDevice.address]
        return None

    def get_joypad_pressing_buttons(self):
        """
        Get the currently pressed buttons

        Returns:
            List (list): List of currently pressed buttons[], with corresponding values:
                        L1: Button L1, L2: Button L2, LS: Left roller press button
                        R1: Button R1, R2: Button R2, RS: Right roller press button
                        U: Direction key up, D: Direction key down, L: Direction key left, R: Direction key right
                        X: Button X, Y: Button Y, A: Button A, B: Button B
        """
        response = self.NETWORK.get_joypad_button_state()
        return response

    def get_joypad_coordinate(self):
        """
        Get joystick roller coordinates

        Returns:
            Roller coordinates (list): [LX, LY, RX, RY]
                    :LX(int): Left roller X-coordinate within the range [-255, 255]
                    :LY(int): Left roller Y-coordinate within the range [-255, 255]
                    :RX(int): Right roller X-coordinate within the range [-255, 255]
                    :RY(int): Right roller Y-coordinate within the range [-255, 255]

        """
        response = self.NETWORK.get_joypad_coordinate()
        return response

    def get_mac_address(self):
        """
        Get the main control MAC address

        Returns:
            MAC address (str)
        """
        response = self.DEVICE.getMacAddress()
        if response.code == 0:
            return response.mac
        return ''

    def get_device_name(self):
        """
        Get the main control name

        Returns:
            Main control name (str): UGOT_XXXX
        """
        response = self.DEVICE.getMacAddress()
        if response.code == 0:
            mac = response.mac
            mac_array = mac.split(':')
            if len(mac_array) > 2:
                last = mac_array.pop()
                last2 = mac_array.pop()
                name = 'ugot_' + last2 + last
                name = name.upper()
                return name
        return ''

    """

    >>>>>>>>>>>
    >> 外设列表 <<        
    >>>>>>>>>>>

    """

    def get_peripheral_devices_list(self):
        """
        Get a list of all connected peripheral devices

        Returns:
            Device list (list): [{'type': '', 'deviceId': '', 'firmware': '', 'serial':', ...]
                :type (str): Device type (motor, servo, power, Clamp, Infrared, etc.)
                :deviceId (str): Device ID
                :firmware (str): Device version
                :serial (str): Device serial number
        """
        response = self.DEVICE.getDeviceList()
        if response.code == 0:

            result = []
            for key in list(response.data):
                item = response.data[key]
                for device in item.device_list:
                    # if device.type == 'motor' or device.type == 'servo':
                    result.append({'type':device.type, 'deviceId':device.deviceId, 'firmware':device.firmware, 'serial':device.serial})

            return result
        return []

    """

    >>>>>>>>>>>
    >> 引脚 - Pins <<        
    >>>>>>>>>>>

    """

    def set_pin_level(self, pin, level):
        """
        Set pin level

        Args:
            pin (int): Pin number, supported pins: 4, 5, 6
            level (bool): True for high level, False for low level

        Returns:
            None
        """
        self.GPIO.setGpioExport(str(pin), level)

    def read_pin_level(self, pin):
        """
        Read pin level

        Args:
            pin (int): Pin number, supported pins: 4, 5, 6

        Returns:
            Level (int): 1 for high level, 0 for low level
        """
        data = self.GPIO.readGpio(str(pin))
        if data.code == 0:
            result = data.result
            return int(result)
        return 0

    def set_pin_pwm(self, pin, duty_cycle):
        """
        Write an analog value to the pin

        Args:
            pin (int): Pin number, supported pins: 1, 2
            duty_cycle (int): Value in the range [0, 255]

        Returns:
            None
        """
        duty_cycle = num_normal(duty_cycle, 255, 0)
        self.GPIO.setGpioStartExportPwm(str(pin), duty_cycle)

    def set_pin_pwm_duty_cycle(self, pin, frequency, duty_cycle):
        """
        Write an analog value to the pin by frequency and duty cycle

        Args:
            pin (int): Pin number, supported pins: 1, 2
            frequency(int): frequency [10, 980]
            duty_cycle(float): duty cycle [0.0, 100.0]

        Returns:
            None
        """
        logging.debug('set_pin_pwm_duty_cycle:{},{},{}'.format(pin, duty_cycle,frequency))
        frequency = num_normal(frequency, 980, 10)
        duty_cycle = num_normal(duty_cycle, 100.0, 0.0)
        self.GPIO.setGpioStartExportPwmWithDutyCycle(str(pin), duty_cycle, frequency)

    def set_serial_serbaud(self, baudrate):
        """
        Set serial baud rate

        Args:
            baudrate (int): Baud rate, available values: 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200

        Returns:
            None
        """
        self.GPIO.setSerbaud(baudrate=baudrate)

    def write_serial_string(self, newline, text):
        """
        Write a string to the serial port

        Args:
            newline (bool): True to add a newline, False to not add a newline
            text (str): String to write

        Returns:
            None
        """
        text = str(text)
        if newline:
            text += '\r\n'
        self.GPIO.serialExportString(value=text)

    def write_serial_number(self, number):
        """
        Write a number to the serial port

        Args:
            number (int): Number to write

        Returns:
            None
        """

        text = str(number)
        self.GPIO.serialExportString(value=text)

    def read_serial_byte(self):
        """
        Read a byte from the serial port

        Returns:
            Byte (str)
        """

        data = self.GPIO.serialReadByte()
        if data.code == 0:
            result = data.result
            return result
        return ''

    def read_serial_string(self):
        """
        Read a string from the serial port

        Returns:
            String (str)
        """
        data = self.GPIO.serialReadString()
        if data.code == 0:
            result = data.result
            return result
        return ''

    def read_serial_string_until(self, char):
        """
        Read a string from the serial port until a specific character

        Args:
            char (str): Character to stop at

        Returns:
            String (str)
        """
        data = self.GPIO.serialReadUtil(char_=char)
        if data.code == 0:
            result = data.result
            return result
        return ''

    def clear_gpio_serial(self):
        """
        Release serial port resources

        Returns:
            None
        """
        self.GPIO.clearAllGpioAndSerial()

    def create_pid_controller(self):
        """
        Create a PID controller

        Returns:
            PID(class): PID controller object
        """
        pid = PID()
        return pid


    """

    >>>>>>>>>>>
    >> 摄像头 - Camera <<        
    >>>>>>>>>>>

    """

    def open_camera(self):
        """
        Open camera

        Returns:
            None
        """
        if self.VISION.camera_client_id > 0:
            logging.debug('当前摄像头已经打开,无需重复打开')
            pass
        else:
            self.VISION.camera_client_id = self.VISION.openCamera()
            logging.debug('open_camera:',self.VISION.camera_client_id)
        time.sleep(1)

    # def close_camera(self):
    #     """
    #     Close camera

    #     Returns:
    #         None
    #     """
    #     self.VISION.closeCamera(self.VISION.camera_client_id)
    
    def read_camera_data(self):
        """
        Read the data of the current frame of the camera

        Returns:
            data(str): the decoded base64 image string (Returns None when no data is obtained)
        """
        response = self.VISION.readCameraData()
        if not (response.code == 0 and response.nDataLen > 0):
            print('No camera data received')
            return None
        
        pdata = response.pdata
        img_data = base64.b64decode(pdata)
        return img_data