# Road traffic simulator
import pygame
from pygame.locals import *
import time
import random
import matplotlib.pyplot as plt

class Car:
    def __init__(self):
        self.starting_position = [0, 0]
        self.destination_position = [100, 100]
        self.image_object = None

        self.time_to_take = 5000
        self.steps = [0, 0]

        self.driving_boundaries = None # [[line_1_start, line_1_end], [line_2_start, line_2_end]] in normal plane coordinate

        self.temporary_destination_coordinate = None
        self.temporary_starting_coordinate = None
        self.current_lane_count = 0
        self.temporary_steps = []
        self.time_counter = 0
        self.drunk_driver_path = None

    def set_start_end_coordinate(self, coordinates_array):
        self.starting_position = coordinates_array[0]
        self.temporary_starting_coordinate = self.starting_position
        self.destination_position = coordinates_array[1]

        self.steps = [(self.destination_position[0] - self.starting_position[0])/self.time_to_take, (self.destination_position[1] - self.starting_position[1])/self.time_to_take]

    def drunk_driver_next_coordinate(self):
        next_coordinate = self.drunk_driver_path[0][self.current_lane_count]
        if(self.current_lane_count == len(self.drunk_driver_path[0])-1):
            return next_coordinate
        self.current_lane_count += 1
        # print("Moving to coordinate: ", next_coordinate, " and path is ", self.drunk_driver_path[0])
        
        return next_coordinate

    def create_drunk_and_drive_path(self):
        x_boundary_breach = 200 # After it hits it walls, how much more, in x direction can it go

        lane_change_count = random.randint(3,10)
        print("Driving boundaries: ", self.driving_boundaries)
        # random_y_coordinates = [random.uniform(self.driving_boundaries[0][0][1], self.driving_boundaries[0][1][1]) for i in range(lane_change_count)]
        random_y_coordinates = [random.uniform(self.starting_position[1], self.driving_boundaries[0][1][1]) for i in range(lane_change_count)]
        random_y_coordinates.sort()
        time_in_lane = [random.randint(5000,10000) for i in range(lane_change_count + 1)]
        random_x_coordinates = [random.uniform(self.driving_boundaries[0][0][0] - x_boundary_breach, self.driving_boundaries[1][0][0] + x_boundary_breach) for i in range(lane_change_count)]
        random_coordinates = [[random_x_coordinates[i], random_y_coordinates[i]] for i in range(lane_change_count)]
        random_coordinates.append(self.destination_position)
        print("Drunk drive path: ", random_coordinates)
        print("Time to be taken in each lane: ", time_in_lane)
        print("Lane change count: ", lane_change_count)

        # Plotting code
        # [[line_1_start, line_1_end], [line_2_start, line_2_end]]
        # plt.plot([self.driving_boundaries[0][i][0] for i in range(2)], [self.driving_boundaries[0][i][1] for i in range(2)])
        # plt.plot([self.driving_boundaries[1][i][0] for i in range(2)], [self.driving_boundaries[1][i][1] for i in range(2)])
        # plt.scatter(random_x_coordinates, random_y_coordinates)
        # plt.show()

        self.drunk_driver_path = [[self.starting_position]+random_coordinates, time_in_lane]

    def simple_driving_next_coordinate(self):
        self.starting_position[0] += self.steps[0]
        self.starting_position[1] += self.steps[1]

        return self.starting_position


class Simulator:
    def __init__(self):
        # Initializing the simulator
        pygame.init()
        screen_info = pygame.display.Info()
        self.screen_width = screen_info.current_w
        self.screen_height = screen_info.current_h
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Collision Simulator")
        self.running = True
        self.day_mode = "day"
        self.line_color = (0, 0, 0)
        self.line_width = 3
        self.cars = []
        self.path_coordinates = []

    def simulate_day_mode(self, mode):
        if(mode == "day"):
            self.screen.fill((255, 255, 255))
        elif(mode == "night"):
            self.screen.fill((0, 0, 0))
            self.line_color = ((255, 255, 255))
        self.day_mode = mode

    def calculate_percentage_value(self, number, percentage_value):
        return number*percentage_value/100
    
    def to_pygame_coordinates(self, coordinates):
        """Converts coordinates from conventional 2D plane to pygame's coordinate system"""
        return (coordinates[0], self.screen_height - coordinates[1])
    
    def get_mid_y_reflection(self, coordinates):
        """Gets reflection of a point along mid screen, vertically in regular 2D coordinate."""
        return (self.screen_width - coordinates[0], coordinates[1])
    
    def get_mid_x_reflection(self, coordinates):
        """Gets reflection of a point along mid screen, horizontally in regular 2D coordinate."""
        return (coordinates[0], self.screen_height - coordinates[1])

    def create_path(self, path_type):
        if(path_type == "t-shape"):
            
            line1_start_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 40), 0))
            line1_end_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 40), self.calculate_percentage_value(self.screen_height, 50)))

            line2_start_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 60), 0))
            line2_end_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 60), self.calculate_percentage_value(self.screen_height, 50)))

            line3_start_coordinate = self.to_pygame_coordinates((0, self.calculate_percentage_value(self.screen_height, 50)))
            line3_end_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 40), self.calculate_percentage_value(self.screen_height, 50)))

            line4_start_coordinate = self.to_pygame_coordinates((self.calculate_percentage_value(self.screen_width, 60), self.calculate_percentage_value(self.screen_height, 50)))
            line4_end_coordinate = self.to_pygame_coordinates((self.screen_width, self.calculate_percentage_value(self.screen_height, 50)))

            line5_start_coordinate = self.to_pygame_coordinates((0, self.calculate_percentage_value(self.screen_height, 80)))
            line5_end_coordinate = self.to_pygame_coordinates((self.screen_width, self.calculate_percentage_value(self.screen_height, 80)))


            pygame.draw.line(self.screen, self.line_color, line1_start_coordinate, line1_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line2_start_coordinate, line2_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line3_start_coordinate, line3_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line4_start_coordinate, line4_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line5_start_coordinate, line5_end_coordinate, self.line_width)

        elif(path_type == "crossroad"):

            line1_normal_start_coordinate = (self.calculate_percentage_value(self.screen_width, 40), 0)
            line1_normal_end_coordinate = (self.calculate_percentage_value(self.screen_width, 40), self.calculate_percentage_value(self.screen_height, 30))
            
            line1_start_coordinate = self.to_pygame_coordinates(line1_normal_start_coordinate)
            line1_end_coordinate = self.to_pygame_coordinates(line1_normal_end_coordinate)

            line2_normal_start_coordinate = self.get_mid_y_reflection(line1_normal_start_coordinate)
            line2_normal_end_coordinate = self.get_mid_y_reflection(line1_normal_end_coordinate)

            line2_start_coordinate = self.to_pygame_coordinates(line2_normal_start_coordinate)
            line2_end_coordinate = self.to_pygame_coordinates(line2_normal_end_coordinate)

            line3_normal_start_coordinate = (0, line1_normal_end_coordinate[1])
            line3_normal_end_coordinate = (line1_normal_start_coordinate[0], line3_normal_start_coordinate[1])

            line3_start_coordinate = self.to_pygame_coordinates(line3_normal_start_coordinate)
            line3_end_coordinate = self.to_pygame_coordinates(line3_normal_end_coordinate)

            line4_normal_start_coordinate = self.get_mid_x_reflection(line3_normal_start_coordinate)
            line4_normal_end_coordinate = self.get_mid_x_reflection(line3_normal_end_coordinate)

            line4_start_coordinate = self.to_pygame_coordinates(line4_normal_start_coordinate)
            line4_end_coordinate = self.to_pygame_coordinates(line4_normal_end_coordinate)

            line5_normal_start_coordinate = (line2_normal_start_coordinate[0], line2_normal_end_coordinate[1])
            line5_normal_end_coordinate = (self.screen_width, line5_normal_start_coordinate[1])

            line5_start_coordinate = self.to_pygame_coordinates(line5_normal_start_coordinate)
            line5_end_coordinate = self.to_pygame_coordinates(line5_normal_end_coordinate)

            line6_normal_start_coordinate = self.get_mid_x_reflection(line5_normal_start_coordinate)
            line6_normal_end_coordinate = self.get_mid_x_reflection(line5_normal_end_coordinate)

            line6_start_coordinate = self.to_pygame_coordinates(line6_normal_start_coordinate)
            line6_end_coordinate = self.to_pygame_coordinates(line6_normal_end_coordinate)

            line7_normal_start_coordinate = self.get_mid_x_reflection(line1_normal_start_coordinate)
            line7_normal_end_coordinate = self.get_mid_x_reflection(line1_normal_end_coordinate)

            line7_start_coordinate = self.to_pygame_coordinates(line7_normal_start_coordinate)
            line7_end_coordinate = self.to_pygame_coordinates(line7_normal_end_coordinate)

            line8_normal_start_coordinate = self.get_mid_x_reflection(line2_normal_start_coordinate)
            line8_normal_end_coordinate = self.get_mid_x_reflection(line2_normal_end_coordinate)

            line8_start_coordinate = self.to_pygame_coordinates(line8_normal_start_coordinate)
            line8_end_coordinate = self.to_pygame_coordinates(line8_normal_end_coordinate)


            pygame.draw.line(self.screen, self.line_color, line1_start_coordinate, line1_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line2_start_coordinate, line2_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line3_start_coordinate, line3_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line4_start_coordinate, line4_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line5_start_coordinate, line5_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line6_start_coordinate, line6_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line7_start_coordinate, line7_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line8_start_coordinate, line8_end_coordinate, self.line_width)

        elif (path_type == "simple"):

            line1_normal_start_coordinate = (self.calculate_percentage_value(self.screen_width, 20), 0)
            line1_normal_end_coordinate = (line1_normal_start_coordinate[0], self.screen_height)

            line1_start_coordinate = self.to_pygame_coordinates(line1_normal_start_coordinate)
            line1_end_coordinate = self.to_pygame_coordinates(line1_normal_end_coordinate)

            line2_normal_start_coordinate= self.get_mid_y_reflection(line1_normal_start_coordinate)
            line2_normal_end_coordinate= self.get_mid_y_reflection(line1_normal_end_coordinate)

            line2_start_coordinate = self.to_pygame_coordinates(line2_normal_start_coordinate)
            line2_end_coordinate = self.to_pygame_coordinates(line2_normal_end_coordinate)

            self.path_coordinates = [[line1_normal_start_coordinate, line1_normal_end_coordinate], [line2_normal_start_coordinate, line2_normal_end_coordinate]]
            # print(self.path_coordinates)

            pygame.draw.line(self.screen, self.line_color, line1_start_coordinate, line1_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line2_start_coordinate, line2_end_coordinate, self.line_width)

        elif (path_type == "roundabout"):

            circle_radius = 100

            pygame.draw.circle(self.screen, self.line_color, (self.screen_width/2, self.screen_height/2), circle_radius)

            line1_normal_start_coordinate = (self.calculate_percentage_value(self.screen_width, 40), 0)
            line1_normal_end_coordinate = (self.calculate_percentage_value(self.screen_width, 40), self.calculate_percentage_value(self.screen_height, 30))
            
            line1_start_coordinate = self.to_pygame_coordinates(line1_normal_start_coordinate)
            line1_end_coordinate = self.to_pygame_coordinates(line1_normal_end_coordinate)

            line2_normal_start_coordinate = self.get_mid_y_reflection(line1_normal_start_coordinate)
            line2_normal_end_coordinate = self.get_mid_y_reflection(line1_normal_end_coordinate)

            line2_start_coordinate = self.to_pygame_coordinates(line2_normal_start_coordinate)
            line2_end_coordinate = self.to_pygame_coordinates(line2_normal_end_coordinate)

            line3_normal_start_coordinate = (0, line1_normal_end_coordinate[1])
            line3_normal_end_coordinate = (line1_normal_start_coordinate[0], line3_normal_start_coordinate[1])

            line3_start_coordinate = self.to_pygame_coordinates(line3_normal_start_coordinate)
            line3_end_coordinate = self.to_pygame_coordinates(line3_normal_end_coordinate)

            line4_normal_start_coordinate = self.get_mid_x_reflection(line3_normal_start_coordinate)
            line4_normal_end_coordinate = self.get_mid_x_reflection(line3_normal_end_coordinate)

            line4_start_coordinate = self.to_pygame_coordinates(line4_normal_start_coordinate)
            line4_end_coordinate = self.to_pygame_coordinates(line4_normal_end_coordinate)

            line5_normal_start_coordinate = (line2_normal_start_coordinate[0], line2_normal_end_coordinate[1])
            line5_normal_end_coordinate = (self.screen_width, line5_normal_start_coordinate[1])

            line5_start_coordinate = self.to_pygame_coordinates(line5_normal_start_coordinate)
            line5_end_coordinate = self.to_pygame_coordinates(line5_normal_end_coordinate)

            line6_normal_start_coordinate = self.get_mid_x_reflection(line5_normal_start_coordinate)
            line6_normal_end_coordinate = self.get_mid_x_reflection(line5_normal_end_coordinate)

            line6_start_coordinate = self.to_pygame_coordinates(line6_normal_start_coordinate)
            line6_end_coordinate = self.to_pygame_coordinates(line6_normal_end_coordinate)

            line7_normal_start_coordinate = self.get_mid_x_reflection(line1_normal_start_coordinate)
            line7_normal_end_coordinate = self.get_mid_x_reflection(line1_normal_end_coordinate)

            line7_start_coordinate = self.to_pygame_coordinates(line7_normal_start_coordinate)
            line7_end_coordinate = self.to_pygame_coordinates(line7_normal_end_coordinate)

            line8_normal_start_coordinate = self.get_mid_x_reflection(line2_normal_start_coordinate)
            line8_normal_end_coordinate = self.get_mid_x_reflection(line2_normal_end_coordinate)

            line8_start_coordinate = self.to_pygame_coordinates(line8_normal_start_coordinate)
            line8_end_coordinate = self.to_pygame_coordinates(line8_normal_end_coordinate)


            pygame.draw.line(self.screen, self.line_color, line1_start_coordinate, line1_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line2_start_coordinate, line2_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line3_start_coordinate, line3_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line4_start_coordinate, line4_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line5_start_coordinate, line5_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line6_start_coordinate, line6_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line7_start_coordinate, line7_end_coordinate, self.line_width)
            pygame.draw.line(self.screen, self.line_color, line8_start_coordinate, line8_end_coordinate, self.line_width)

    def make_car(self, count):
        for i in range(count):
            car = Car()
            car_img = pygame.image.load("rough/car1.png")
            car.image_object = car_img

            car_width = self.calculate_percentage_value(car_img.get_rect().width, 15)
            car_height = self.calculate_percentage_value(car_img.get_rect().height, 15)

            car.image_object = pygame.transform.scale(car.image_object, (car_width, car_height))
            car.image_object = pygame.transform.rotate(car.image_object, 0)
            # print(self.path_coordinates)
            car.driving_boundaries = self.path_coordinates

            car.set_start_end_coordinate([[self.screen_width/2, 100], [self.screen_width/2, self.screen_height]])
            self.screen.blit(car.image_object, self.to_pygame_coordinates((self.screen_width/2, 0)))


            self.cars.append(car)

    def car_driver(self):
        for car in self.cars:
            self.screen.blit(car.image_object, self.to_pygame_coordinates(car.simple_driving_next_coordinate()))

    def drunk_driver(self):
        car = self.cars[0]
        self.screen.blit(car.image_object, self.to_pygame_coordinates(car.drunk_driver_next_coordinate()))

    def start_simulator(self):
        
        self.simulate_day_mode(self.day_mode)
        self.create_path("simple")

        self.make_car(1)
        self.cars[0].create_drunk_and_drive_path()
        pygame.display.flip()


        # self.screen.blit(self.cars[0].image_object, self.to_pygame_coordinates(self.cars[0].starting_position))
        while self.running:
            for event in pygame.event.get():
                if ((event.type == pygame.QUIT) or ((event.type == KEYDOWN) and (event.key == K_ESCAPE))):
                    self.running = False
            # new_pos = [self.cars[0].starting_position[0], self.cars[0].starting_position[1]+1]
            # self.cars[0].starting_position = new_pos
            time.sleep(2)
            self.simulate_day_mode(self.day_mode)
            self.create_path("simple")
            # self.car_driver()
            self.drunk_driver()
            # Simulating day mode
            pygame.display.flip()
        pygame.quit()

simulator = Simulator()
simulator.simulate_day_mode("night")
simulator.start_simulator()