import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_acc, max_steer_angle, deceleration_limit, vehicle_mass, wheel_radius):
        
        ## Controller parameters
        # PID controller
        kp = 0.3
        ki = 0.1
        kd = 0.
        min_throttle = 0.
        max_throttle = 0.2

        # Low Pass Filter
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        sample_time = .02  # sample time

        ## Controller
        self.yaw_contr = YawController(wheel_base, steer_ratio, 0.1, max_lat_acc, max_steer_angle)
        self.throttle_contr = PID(kp, ki, kd, min_throttle, max_throttle)
        self.velocity_filter = LowPassFilter(tau, sample_time)

        ## Other needed parameters
        self.last_time = rospy.get_time()
        self.last_vel = 0
        self.decel_limit = deceleration_limit
        self.vehicle_m = vehicle_mass
        self.wheel_r = wheel_radius


    def control(self, current_vel, linear_vel, angular_vel, enabled):
        '''
        Function to achieve throttle, brake and steering values by using PID and yaw controller
        '''
    

        ## Reset PID controller, if the DBW is disabled
        # This prevents an accumelation of the error, when e.g. standing at a traffic light
        if not enabled:
            self.throttle_contr.reset()
            return 0., 0., 0.

        ## Perform lowpass filter on current velocity
        current_vel = self.velocity_filter.filt(current_vel)

        ## Get steering from yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        ## Calculate throttle with PID controller
        # Calculate the velocity error delta v
        delta_v = linear_vel - current_vel
        self.last_vel = current_vel

        # Calculate delta t
        current_time = rospy.get_time()
        delta_t = current_time - self.last_time
        self.last_time = current_time

        # Perform next step in PID controller
        throttle = self.throttle_contr.step(delta_v, delta_t)
        brake = 0

        ## Brake value calculation
        # Since the car has an automatic transmission, a brake value is needed to keep the car in position
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 #N*m - Torque
        # If we want to decrease velocity, brake is used
        elif throttle < .1 and delta_v < 0:
            throttle = 0
            decel = max(delta_v, self.decel_limit)
            brake = abs(decel)*self.vehicle_m*self.wheel_r #N*m - Torque

        return throttle, brake, steering
