#!/usr/bin/python
#

import time

class PID:
    """PID Controller
    """

    def __init__(self,current_time=None):

        self.set_pid(0.2,0.0,0.0)

        self.__current_time = current_time if current_time is not None else time.time()
        self.__last_time = self.__current_time

        self.__clear()

        self.__setSampleTime(0.01)

    def __clear(self):
        """Clears PID computations and coefficients"""
        self.__SetPoint = 0.0

        self.__PTerm = 0.0
        self.__ITerm = 0.0
        self.__DTerm = 0.0
        self.__last_error = 0.0

        # Windup Guard
        self.__int_error = 0.0
        self.__windup_guard = 20.0

        self.__output = 0.0

    """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        .. figure:: images/pid_1.png
           :align:   center

           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

        """
    def update(self, feedback_value, current_time=None):
        """
        Set the error of pid controller

        Args:
            error (float): error

        Returns:
            output (float):
        """
        
        error = self.__SetPoint - feedback_value

        self.__current_time = current_time if current_time is not None else time.time()
        delta_time = self.__current_time - self.__last_time
        delta_error = error - self.__last_error

        if (delta_time >= self.__sample_time):
            self.__PTerm = self.__Kp * error
            self.__ITerm += error * delta_time

            if (self.__ITerm < -self.__windup_guard):
                self.__ITerm = -self.__windup_guard
            elif (self.__ITerm > self.__windup_guard):
                self.__ITerm = self.__windup_guard

            self.__DTerm = 0.0
            if delta_time > 0:
                self.__DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.__last_time = self.__current_time
            self.__last_error = error

            self.__output = self.__PTerm + (self.__Ki * self.__ITerm) + (self.__Kd * self.__DTerm)

        return self.__output

    def set_pid(self, kp, ki, kd):
        """
        Set the paramters of pid controller

        Args:
            kp (float): kp
            ki (float): ki
            kd (float): kd

        Returns:
            None
        """
        self.__setKp(kp)
        self.__setKi(ki)
        self.__setKd(kd)

    def __setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.__Kp = proportional_gain

    def __setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.__Ki = integral_gain

    def __setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.__Kd = derivative_gain

    def __setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.__windup_guard = windup

    def __setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.__sample_time = sample_time
