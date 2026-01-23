class E_Model:
    class Direction:
        forward = 0  # 前进
        backward = 1  # 后退
        turn_left = 2  # 左转
        turn_right = 3  # 右转
    class SDirection:
        forward = 0  # 前进
        backward = 1  # 后退
        translate_left = 2 # 左平移
        translate_right = 3 # 右平移
    class Unit:
        second = 0  # 秒
        mileage = 1  # 厘米
        angle = 2  # 度

    class ArmJoint:
        left_forward = 1
        left_backward = 2
        right_backward = 3
        right_forward = 4

    class ArmControl:
        RELEASE = 0
        CLOSE = 1

    class Adaption:
        OFF = 0
        ON = 1

    class Mechanical:
        joint1 = 1
        joint2 = 2
        joint3 = 3

    class WheelLegHeight:
        HIGH = 1
        MEDIUM = 2
        LOW = 3

    class ChassisMode:
        box = 'box'
        mecanum = 'mecanum'
        balance = 'balance'
        transform = 'transform'
        wheel_legged = 'wheel_legged'
        spider = 'spider'
        dog = 'dog'
        engineer = 'engineer'

class E_Device:
    class Color:
        black = 0  # 黑色
        white = 1  # 白色
        purple = 2  # 紫色
        red = 3  # 红色
        orange = 4  # 橙色
        yellow = 5  # 黄色
        green = 6  # 绿色
        cyan = 7  # 青色
        blue = 8  # 蓝色
        ColorList = {
            black: "FF000000",
            white: "FFFFFFFF",
            purple: "FFFF00FF",
            red: "FFFF0000",
            orange: "FFFFA500",
            yellow: "FFFFFF00",
            green: "FF00FF00",
            cyan: "FF00FFFF",
            blue: "FF0000FF",
        }

    class Light:
        class ID:
            TOP = 0
            LEFT = 1
            RIGHT = 2
            DOWN = 3

        class Effect:
            TURN_ON = 0  # 常亮
            TURN_OFF = 1  # 关闭
            BREATHING = 2  # 呼吸
            FLASHING = 3  # 闪烁

class E_Vision:
    class Traffic:
        green_light = 0 # 绿灯
        horn = 1 # 鸣笛
        left = 2 # 左转
        right = 3 # 右转
        zebra_crossing = 4 # 斑马线
        red_light = 5 # 红灯
        children = 6 # 注意儿童
        stop = 7 # 禁止长时间停车
        tunnel = 8 # 进入隧道
        yellow_light = 9 # 黄灯

        TrafficSigns = {
            green_light : '绿灯',
            horn: '鸣笛',
            left: '左转',
            right: '右转',
            zebra_crossing: '斑马线',
            red_light: '红灯',
            children: '注意儿童',
            stop: '禁止长时间停车',
            tunnel: '进入隧道',
            yellow_light: '黄灯',
        }
    class LPD:
        # "蓝牌", "绿牌", "黄牌", "黑牌", "黄绿牌", "不确定"
        blue = 0 # 蓝牌
        green = 1 # 绿牌
        yellow = 2 # 黄牌
        black = 3 # 黑牌
        yellow_green = 4 # 黄绿牌
        unknown = 5 # 不确定

    class Pose:
        nose = 0  # 鼻子
        neck = 1  # 脖子
        rightShoulder = 2  # 右肩
        rightElbow = 3  # 右肘
        rightWrist = 4  # 右手
        leftShoulder = 5  # 左肩
        leftElbow = 6  # 左肘
        leftWrist = 7  # 左手
        rightHip = 8  # 右胯
        rightKnee = 9  # 右膝
        rightAnkle = 10  # 右脚
        leftHip = 11  # 左胯
        leftKnee = 12  # 左膝
        leftAnkle = 13  # 左脚
        rightEye = 14  # 右眼
        leftEye = 15  # 左眼
        rightEar = 16  # 右耳
        leftEar = 17  # 左耳

        TotalPoseIndexes = [rightEar, rightEye, nose, leftEye, leftEar,
                            rightWrist, rightElbow, rightShoulder, leftShoulder, leftElbow, leftWrist,
                            rightHip, leftHip, rightKnee, leftKnee, rightAnkle, leftAnkle]

    class Intersection:
        crossroad = 0  # 十字路口
        ycross = 1  # y字路口
        straight = 2  # 一条直线
        junction = 3  # 路口 - 双轨判断
        noline = 10  # 无线

    class LineColor:
        # 0.1.2.3.4 对应黑白蓝绿红
        black = 0  # 黑色
        white = 1  # 白色
        blue = 2  # 蓝色
        green = 3  # 绿色
        red = 4  # 红色

    class LineType:
        single = 0  # 单轨
        double = 1  # 双轨

    class Toy:
        YOUYOU = 0 # 优悠
        WALKERX = 1 # walkerx
        WALKER = 2 # walker

class E_Audio:
    class Volume:
        MINIMUM = 1  # 最小 20%
        LOW = 2  # 小 40%
        MEDIUM = 3  # 中 60%
        HIGH = 4  # 大 80%
        MAXIMUM = 5  # 最大 100%

    class Type:
        UPLOAD = 0 # 0 上传的音频
        RECORD = 1  # 1 表示录音
        DEFAULT = 2  # 2 表示内部音效

    class Direction:
        FRONT = 1
        BACK = 2
        LEFT = 3
        RIGHT = 4

    class Timbre:
        Female = 0  # 女声
        Male = 1  # 男声

    class DirectionControl:
        ON = 1  # 开启声源定位
        OFF = 0  # 关闭声源定位

    class Beat:
        EIGHTH = 0
        QUARTER = 1
        HALF = 2
        ONE = 3
        DOUBLE = 4
        BEATS = {
            EIGHTH : '1-8',
            QUARTER : '1-4',
            HALF : '1-2',
            ONE : '1',
            DOUBLE : '2',
        }

class E_Joypad:

    X = 'X' # X
    Y = 'Y' # Y
    A = 'A' # A
    B = 'B' # B
    L1 = 'L1' # L1
    L2 = 'L2' # L2
    R1 = 'R1' # R1
    R2 = 'R2' # R2
    LS = 'LS' # LS 左滚轮
    RS = 'RS' # RS 右滚轮
    U = 'U' # 方向 - 上
    D = 'D' # 方向 - 下
    L = 'L' # 方向 - 左
    R = 'R' # 方向 - 右

