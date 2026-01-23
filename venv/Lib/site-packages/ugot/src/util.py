
#
class RangMode(object):
    """定义数据范围控制模式

    NORMAL: 正常
    ABSOLUTE: 绝对值
    OPPOSITE: 相反数
    """
    NORMAL = 0
    ABSOLUTE = 1
    OPPOSITE = 2
def num_normal(value, upper_limit, lower_limit, mode=RangMode.NORMAL):
    """规范化数值

            Args:
                value: 原始值
                upper_limit: 数值上限
                lower_limit: 数值下限
                mode (RangMode, optional): 规范模式. Defaults to RangMode.NORMAL.

            Returns:
                返回规范后的值. 如果 value 超出了 [lower_limit, upper_limit], 会被置为 lower_limit
                或 upper_limit, 然后根据 mode 进行改变, 例如:

                k = num_normal(5, 3, 4) # k = 5
                p = num_normal(5, 8, 10, RangMode.OPPOSITE) # p = -8
                n = num_normal(-5, -10, -6, RangMode.ABSOLUTE) # n = 6
            """
    # 确保 上限 不低于 下限
    if upper_limit < lower_limit:
        (upper_limit, lower_limit) = (lower_limit, upper_limit)

    if value > upper_limit:
        value = upper_limit
    if value < lower_limit:
        value = lower_limit
    if mode == RangMode.ABSOLUTE:
        value = abs(value)
    elif mode == RangMode.OPPOSITE:
        value = value * (-1)

    return value

import math
class _Color(object):
    color = 0

    def __init__(self, color=0):
        self.color = color

    def red(self):
        return (self.color & 0x00FF0000) >> 16

    def green(self):
        return (self.color & 0x0000FF00) >> 8

    def blue(self):
        return self.color & 0x000000FF


class Color(_Color):
    RED = _Color(0x00FF0000)
    ORANGE = _Color(0x00FF8000)
    YELLOW = _Color(0x00FFF000)
    GREEN = _Color(0x0000FF00)
    CYAN = _Color(0x0000FFFF)
    BLUE = _Color(0x000000FF)
    PURPLE = _Color(0x00FF00FF)
    BLACK = _Color(0x00000000)
    WHITE = _Color(0x00FFFFFF)
    GREY = _Color(0x00BEBEBE)

    Lights = {
        1 : RED,
        2 : ORANGE,
        3 : YELLOW,
        4 : GREEN,
        5 : CYAN,
        6 : BLUE,
        7 : PURPLE,
        8 : WHITE,
        9 : GREY,
    }

    """ 定义颜色类型

    Args:
        object ([type]): [description]
    """

    def __init__(self, color=0):
        super().__init__(color)

    @staticmethod
    def create_color_rgb(red: int, green: int, blue: int):
        """根据红, 绿, 蓝通道值创建颜色

        Args:
            red (int): 红通道值
            green (int): 绿通道值
            blue (int): 蓝通道值

        Returns:
            Color: 返回 Color 对象
        """
        v_red = num_normal(red, 255, 0)
        v_green = num_normal(green, 255, 0)
        v_blue = num_normal(blue, 255, 0)
        v = v_red << 16 | v_green << 8 | v_blue
        return Color(v)

    @staticmethod
    def __create_color_hsv(hue: int, saturation: int, value: int) -> _Color:
        """[summary]

        Args:
            hue (int): 色调值, 范围: [0, 360]
            saturation (int): 饱和度, 范围: [0, 100]
            value (int): 明度, 范围: [0, 100]
        """
        h = int(hue) % 360
        s = num_normal(saturation, 100, 0) * 0.01
        v = num_normal(value, 100, 0) * 0.01

        i = math.floor(h / 60)
        f = (h / 60) - i
        p = v * (1 - s)
        q = v * (1 - (s * f))
        t = v * (1 - (s * (1 - f)))

        if i == 0:
            rr = (v, t, p)
        elif i == 1:
            rr = (q, v, p)
        elif i == 2:
            rr = (p, v, t)
        elif i == 3:
            rr = (p, q, v)
        elif i == 4:
            rr = (t, p, v)
        else:
            rr = (v, p, q)

        result = Color.create_color_rgb(math.floor(rr[0] * 255),
                                        math.floor(rr[1] * 255),
                                        math.floor(rr[2] * 255))
        return result

    @staticmethod
    def create_color_hsv(hue_percent: int, saturation: int, value: int) -> _Color:
        p = num_normal(hue_percent, 100, 0)
        return Color.__create_color_hsv(int(p * 360 * 0.01), saturation, value)