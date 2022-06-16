import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import ast

matplotlib.use("TkAgg")



class Square:
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.x_size = x2 - x1
        self.y_size = y2 - y1

    def intersect(self, other):
        intersecting = not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)
        inside = (other.x1 >= self.x1 and other.x2 <= self.x2 and other.y1 >= self.y1 and other.y2 <= self.y2)
        return intersecting or inside

    def update_pos(self, x, y):
        self.x1, self.x2, self.y1, self.y2 = x, x + self.x_size, y, y + self.y_size

class SimGrid:
    """A different grid to make copies of the existing grid,
    mainly to calculate rewards and simulate it more discretely"""
    def __init__(self, grid):
        self.width = grid.width
        self.height = grid.height
        self.obstacles = grid.obstacles
        self.goals = grid.goals
        self.filthy = grid.filthy
        self.robot = grid.robots[0]
        self.goal_lines = grid.goal_lines
        self.filthy_lines = grid.filthy_lines
        self.robot_lines = grid.robot_lines[0]

    def move_robot(self, directionvector):
        """"this function will move the robot, not by actually moving,
        but by teleporting it forward by the directionvector
        right now, we do not yet remove reward tiles"""
        self.robot.pos = (self.robot.pos[0] + directionvector[0], self.robot.pos[1] + directionvector[1])

    def reward(self, directionvector):
        """"this function will return the reward of a move
        In order for it to stay accurate, no directionvectors of higher than 1 in a specific dimension should be used"""
        self.move_robot(directionvector)
        reward = 0
        for goal in self.goals:
            if goal != None:
                if self.robot.intersect(goal):
                    reward += 3
        for filthy in self.filthy:
            if filthy != None:
                if self.robot.intersect(filthy):
                    reward += 1
        for obstacle in self.obstacles:
            if obstacle != None:
                if self.robot.intersect(obstacle):
                    reward -= 1
        #put the robot back in its original position
        self.move_robot((-directionvector[0], -directionvector[1]))
        return reward

    def something(self):
        pass



class Grid:
    def __init__(self, width, height, ):
        self.width = width
        self.height = height
        self.obstacles = []
        self.goals = []
        self.filthy = []
        self.robots = []

        self.fig = plt.figure()
        axes = self.fig.add_subplot(111)
        self.border_line, = plt.plot(*self.get_border_coords(), color='black')
        self.obstacle_lines = []
        self.goal_lines = []
        self.filthy_lines = []
        self.robot_lines = []

    def spawn_robots(self, robots, starting_positions):
        self.robots = robots
        for i, robot in enumerate(robots):
            robot.spawn(self, *starting_positions[i])
            robot_box = robot.history[-1]
            self.robot_lines.append(plt.plot([robot_box.x1, robot_box.x2, robot_box.x2, robot_box.x1, robot_box.x1],
                                             [robot_box.y1, robot_box.y1, robot_box.y2, robot_box.y2, robot_box.y1],
                                             color='blue')[0])
        for robot in robots:
            if self.is_blocked(robot):
                raise ValueError('Invalid starting pos, position is blocked!')

    def is_in_bounds(self, x, y, size_x, size_y):
        return x >= 0 and x + size_x <= self.width and y >= 0 and y + size_y <= self.height

    def put_obstacle(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        ob = Square(x, x + size_x, y, y + size_y)
        self.obstacles.append(ob)
        self.obstacle_lines.append(
            plt.plot([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='black')[0])

    def put_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        goal = Square(x, x + size_x, y, y + size_y)
        self.goals.append(goal)
        self.goal_lines.append(
            plt.plot([goal.x1, goal.x2, goal.x2, goal.x1, goal.x1], [goal.y1, goal.y1, goal.y2, goal.y2, goal.y1],
                     color='orange')[0])

    def check_goals(self, robot):
        for i, goal in enumerate(self.goals):
            if goal != None:
                if goal.intersect(robot.bounding_box):
                    self.goals[i] = None
                    self.goal_lines[i].set_data(None, None)

    def put_filth(self, x, y, size_x=1, size_y=1):
        assert self.is_in_bounds(x, y, size_x, size_y)
        filth = Square(x + 0.05, x + size_x - 0.05, y + 0.05, y + size_y - 0.05)
        # if the filth intersects with a goal or an obstacle, don't create it
        for goal in self.goals:
            if filth.intersect(goal):
                return
        for obstacle in self.obstacles:
            if filth.intersect(obstacle):
                return
        self.filthy.append(filth)
        self.filthy_lines.append(
            plt.plot([filth.x1, filth.x2, filth.x2, filth.x1, filth.x1],
                     [filth.y1, filth.y1, filth.y2, filth.y2, filth.y1], color='yellow')[0])

    def check_filth(self, robot):
        for i, filth in enumerate(self.filthy):
            if filth != None:
                if filth.intersect(robot.bounding_box):
                    self.filthy[i] = None
                    self.filthy_lines[i].set_data(None, None)

    def is_blocked(self, robot):
        blocked_by_obstacle = any([ob.intersect(robot.bounding_box) for ob in self.obstacles])
        blocked_by_robot = any(
            [robot.bounding_box.intersect(other_robot.bounding_box) for other_robot in self.robots if
             other_robot.id != robot.id])
        return blocked_by_obstacle or blocked_by_robot

    def get_border_coords(self):
        return [0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0]

    def check_intersections(self):
        bounding_box = self.robots[0]
        obstacles = self.obstacles
        blocked = any([ob.intersect(bounding_box) for ob in obstacles]) or \
                  not (bounding_box.x1 >= 0 and bounding_box.x2 <= self.width and bounding_box.y1 >= 0 and
                       bounding_box.y2 <= self.height)

        if blocked:
            return None, None, blocked

        new_filthy = deepcopy(self.filthy)
        for i, filth in enumerate(self.filthy):
            if filth is not None:
                if filth.intersect(bounding_box):
                    new_filthy.remove(new_filthy[i])

        new_goals = deepcopy(self.goals)
        for i, goal in enumerate(self.goals):
            if goal is not None:
                if goal.intersect(bounding_box):
                    new_goals.remove(new_goals[i])

        return new_filthy, new_goals, blocked

    def plot_grid(self):
        for i, robot in enumerate(self.robots):
            robot_box = robot.history[-1]
            self.robot_lines[i].set_xdata([robot_box.x1, robot_box.x2, robot_box.x2, robot_box.x1, robot_box.x1])
            self.robot_lines[i].set_ydata([robot_box.y1, robot_box.y1, robot_box.y2, robot_box.y2, robot_box.y1])
        plt.title('Battery levels: ' + '|'.join([str(round(robot.battery_lvl, 2)) for robot in self.robots]))
        plt.draw()
        plt.pause(0.0001)

    def evaluate(self):
        maxscore = len(self.goals)*3 + len(self.filthy)
        goals = [i for i in self.goals if i is not None]
        filthy = [i for i in self.filthy if i is not None]
        score = len(goals)*3 + len(filthy)
        cleanpercent = 100 - (score/maxscore*100)
        batteryleft = sum([i.battery_lvl for i in self.robots])/len(self.robots)
        return cleanpercent, batteryleft

class Robot:
    def __init__(self, id, size=1, battery_drain_p=0, battery_drain_lam=0):
        self.size = size
        self.id = id
        self.direction_vector = (0, 0)
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True

    def spawn(self, grid, start_x=0, start_y=0):
        self.pos = (start_x, start_y)
        self.bounding_box = Square(start_x, start_x + self.size, start_y, start_y + self.size)
        self.history = [self.bounding_box]
        self.grid = grid
        assert self.grid.is_in_bounds(start_x, start_y, self.size, self.size)

    def move(self, p_random=0):
        """"returns False if the robot is not moving
        Moves the robot and returns True otherwise"""
        # If we have a random move happen:
        if np.random.binomial(n=1, p=p_random) == 1:
            self.direction_vector = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        # If there are no goals or filthy tiles left, die:
        if len([ i for i in self.grid.goals if i is not None]) == 0 and \
                len([ i for i in self.grid.filthy if i is not None]) == 0:
            self.alive = False
        # Cant move if we died:
        if self.alive:
            if self.direction_vector == (0, 0):  # Literally 0 speed so no movement.
                return False
            # print('vector '+str(self.direction_vector[0]))

            # print('init pos: '+str(self.pos))
            # print('dir vec: '+str(self.direction_vector))
            new_pos = tuple(np.array(self.pos) + self.direction_vector)
            # Temporarily set the new bounding box:
            new_box = deepcopy(self.bounding_box)
            new_box.update_pos(*new_pos)
            self.bounding_box = new_box
            if self.grid.is_blocked(self):
                return False
            elif not self.grid.is_in_bounds(new_pos[0], new_pos[1], self.size, self.size):
                return False
            else:
                do_battery_drain = np.random.binomial(1, self.battery_drain_p)
                if do_battery_drain == 1 and self.battery_lvl > 0:
                    # print('4: '+str(self.direction_vector))
                    self.battery_lvl -= (
                            np.random.exponential(self.battery_drain_lam) * abs(sum(self.direction_vector)))
                    if self.battery_lvl <= 0:
                        self.alive = False
                        self.battery_lvl = 0
                        return False
                del new_box
                self.pos = new_pos
                # print('new pos: '+str(self.pos))
                self.bounding_box.update_pos(*self.pos)
                self.history.append(self.bounding_box)
                # Check if in this position we have reached a goal:
                self.grid.check_goals(self)
                self.grid.check_filth(self)
                return True
        else:
            return False

    def intersect(self, other):
        x1, y1 = self.pos
        x2, y2 = self.pos[0]+self.size, self.pos[1]+self.size
        intersecting = not (x2 <= other.x1 or x1 >= other.x2 or y2 <= other.y1 or y1 >= other.y2)
        inside = (other.x1 >= x1 and other.x2 <= x2 and other.y1 >= y1 and other.y2 <= y2)
        return intersecting or inside

def parse_config(file, divideby = 1):
    with open(file, 'r') as f:
        data = f.read().split('\n')
        if len(data) == 0:
            raise ValueError('Config file does not contain any lines!')
        else:
            grid = None
            for line in data:
                if '=' not in line:
                    raise ValueError("Invalid formatting, use size/obstacle/goal = ()")
                else:
                    typ, coords = (i.strip() for i in line.split('='))
                    if typ == 'size':
                        grid = Grid(*ast.literal_eval(coords))
                    else:
                        if not grid:
                            raise ValueError('Wrong order in config file! Start with size!')
                        else:
                            if typ == 'obstacle':
                                grid.put_obstacle(*ast.literal_eval(coords))
                            elif typ == 'goal':
                                grid.put_goal(*ast.literal_eval(coords))
                            else:
                                raise ValueError(f"Unkown type '{typ}'.")
            # the baseline grid is fully defined here
            #TODO: use the divideby variable to put filth smaller
            for i in range(grid.width):
                for j in range(grid.height):
                    grid.put_filth(i,j)

            return grid


