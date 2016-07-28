import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw
import os, sys

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.

CT_TARGET = 3
CT_STATIC = 1
CT_CAR = 0


class Shape:
    def __init__(self, space, r=30, x=50, y=height - 100, angle=0.5, color='orange',
                 static=True, collision_type=CT_STATIC):
        self.r = r
        if static:
            self.body = pymunk.Body(pymunk.inf, pymunk.inf)
        else:
            self.body = pymunk.Body(1, 1./2.*r**2)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r)
        self.shape.color = THECOLORS[color]
        self.shape.elasticity = 1.0
        self.shape.angle = angle
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)



class World:
    def __init__(self, visualize=False):
        global show_sensors, draw_screen
        if visualize:
            show_sensors = True
            draw_screen = True
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            show_sensors = False
            draw_screen = False

        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.space.add_collision_handler(CT_CAR, CT_STATIC, self._crash_handler)

        # Create the car.
        self.car = Shape(self.space, r=30, x=100, y=100, color='green', static=False,
                         collision_type=CT_CAR)
        self.dynamic = [Shape(self.space, r=30, x=200, y=200, color='orange', static=False)]

        # Record steps.
        self.num_steps = 0

        self.last_step = self.car.body.position

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(Shape(self.space, r=55, x=25, y=350, color='purple'))
        self.obstacles.append(Shape(self.space, r=95, x=250, y=550, color='purple'))
        self.obstacles.append(Shape(self.space, r=155, x=500, y=150, color='purple'))

        self.target = Shape(self.space, r=10, x=600, y=60, color='red', collision_type=CT_TARGET)

    def _crash_handler(self, space, arbiter):
        self.crashed = True
        return False

    def step(self, action):
        go = action[0]
        turn = action[1]

        self.car.body.angle += .2 * turn

        # Move cat.
        if self.num_steps % 5 == 0:
            self._move_dynamic()

        driving_direction = Vec2d(1, 0).rotated(self.car.body.angle)
        self.car.body.velocity = int(100 * go) * driving_direction

        # Update the screen and stuff.
        self.space.step(1./10)
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car.body.position
        readings = self._get_sonar_readings(x, y, self.car.body.angle)
        state = np.array([readings])
        if self.crashed:
            if self.num_steps == 0:
                self.reset()
                return self.step(action)
        self.num_steps += 1

        r = self.get_reward(action)

        self.last_step = self.car.body.position

        return r, state, self.crashed

    def get_reward(self, action):
        max_dist = np.linalg.norm([width, height])
        dist = np.linalg.norm(self.target.body.position - self.car.body.position)
        if self.crashed:
            # if dist <= self.car.r + self.target.r:
            #     r = +3.
            # else:
            r = -1.
        else:
            # dist_last = np.linalg.norm(self.target.body.position - self.last_step)
            # # r = ((max_dist - dist)/max_dist)**2
            # r = dist_last - dist
            # print dist_last, dist, r
            r = action[0]/100.

        return r

    def _move_dynamic(self):
        for obj in self.dynamic:
            speed = random.randint(20, 200)
            obj.body.angle -= random.randint(-1, 1)
            direction = Vec2d(1, 0).rotated(obj.body.angle)
            obj.body.velocity = speed * direction

    def reset(self):
        self.crashed = False
        self.num_steps = 0
        for shape in self.obstacles + [self.car, self.target] + self.dynamic:
            shape.body.position = random.randint(0, width), random.randint(0, height)
            shape.body.velocity = Vec2d(0, 0)
            shape.body.angle = random.random()*2*np.pi

    def _get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self._make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self._get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self._get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self._get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()

        return readings

    def _get_arm_distance(self, arm, x, y, angle, offset):
        # TODO: is there a function for this?
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self._get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self._is_empty(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def _make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def _get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def _is_empty(self, reading):  # TODO: fix this stupid color-check
        if reading == THECOLORS['black'] or reading == THECOLORS['green']\
                or reading == THECOLORS['blue']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # show_sensors = False
    # draw_screen = False

    show_sensors = True
    draw_screen = True

    game_state = World()
    import time
    start = time.time()

    for i in range(int(1e4)):
        reward, state, reset = game_state.step(1-2*np.random.random((2,)))
        if reset:
            print reward
            game_state.reset()
        if i % 1000 == 0:
            print i
        time.sleep(0.1)

    end = time.time()
    print 'time=', end-start
