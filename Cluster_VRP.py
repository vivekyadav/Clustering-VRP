# -*- coding: utf-8 -*-
"""
Uses distance matrix provided by GoogleDistMatrix.py script
"""

from math import radians, sin, cos, sqrt, asin
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math

# This determines how many clusters to create
depot = 6
folder = "C:\\Users\\vivek\\Milk Production\\59 Points\\"
dist_matrix_filename = "Dist_Matrix in meters.csv"
time_matrix_filename = "Time_Matrix in seconds.csv"
demand_filename = "demand_array.csv"
xcord = "Latitude.csv"
ycord = "Longitude.csv"

factory_index = 0
number_of_vehicles_for_main = {"BIG": depot - 1}
number_of_vehicles_for_clusters = {"SMALL": 7}
label_for_main_cluster = depot
vehicle_max_distance = 100000
existing_cost = 13378.3
EMI = 170
SERVICE_TIME_PER_VEHICLE = 600 #seconds

class Vehicle():
    def __init__(self, capacity, rate, labour, max_distance):
        self.capacity = capacity
        self.rate = rate
        self.labour = labour
        self.max_distance = max_distance
    def __repr__(self):
        return "(capacity = %s, rate = %s, labour = %s, max_distance = %s)" % (self.capacity, self.rate, self.labour, self.max_distance)
    def cost_provider(self, distance_matrix):
        return lambda x, y: int(distance_matrix[x][y]*self.rate)

# Modify these to control labour charge, capacity, rate and max travel distance for each vehicle type
vehicle_params = {
	"BIG" : Vehicle(2700, 0.025, 300, 60000),
   "BIG_LOCAL": Vehicle(2700, 0.025, 300, 40000),
	"SMALL" : Vehicle(500, 0.020, 100, 100000)
}

class Solution():
    def __init__(self):
        self.xs = []
        self.ys = []
        self.sps = []
        self.dist_matrix = []
        self.demand = []
        self.depots = []
        self.sps_for_depot = {}
        self.pair_dist = None
        self.cluster = None
        self.routes_for_cluster = {}
        self.read_data_from_files()
        self.init_vehicles()
        return

    def read_data_from_files(self):
        """ Reads demands, distance, time, geo coordinates """
        self.xs = list(map(float, open(folder + xcord).read().strip().split('\n')))
        self.ys = list(map(float, open(folder + ycord).read().strip().split('\n')))
        # X(latitude) and Y(longitude) reversed intentionally 
        self.sps = np.array(list(map(list, zip(self.ys, self.xs))))
        self.dist_matrix = np.loadtxt(open(folder + dist_matrix_filename), delimiter = ',')
        self.time_matrix = np.loadtxt(open(folder + time_matrix_filename), delimiter = ',')
        self.demand = np.loadtxt(open(folder + demand_filename))
    
    def init_vehicles(self):
        """ Sets up vehicle types to be used for each cluster.
            Main cluster (Factory to Stockists) usually uses BIG trucks,
            while all other clusters use SMALL vehicles """
        self.vehicles_for_main = []
        for vtype, count in number_of_vehicles_for_main.items():
            for i in range(count):
                self.vehicles_for_main.append(vehicle_params[vtype])
        self.vehicles_for_clusters = []
        for vtype, count in number_of_vehicles_for_clusters.items():
            for i in range(count):
                self.vehicles_for_clusters.append(vehicle_params[vtype])

    def print_data(self):
        """ This isn't called by default, intented for debugging """
        print(self.dist_matrix)
        print(self.demand)

    def find_cluster(self):
        """ Uses K-Means clustering to produce cluster.
            K-Means will assign a label to each service point.
            Points with same label belong to the same cluster"""
        self.cluster = KMeans(n_clusters = depot).fit(self.sps)
        self.prepare_for_vrp()
        # If any of the clusters has too large combined demand, we will re-create clusters
        while max(self.get_demand_for_main_cluster()) > 2700:
            self.cluster = KMeans(n_clusters = depot).fit(self.sps)
            self.prepare_for_vrp()
    
    def prepare_for_vrp(self):
        """ Creates some data structure for easier processing and
            assigns a Stockist/Depot to each cluster.
            Service point with the highest demand in a cluster
            is picked as the depot for that cluster."""
        self.sps_for_cluster = {}
        for index, label in enumerate(self.cluster.labels_):
            if label not in self.sps_for_cluster:
                self.sps_for_cluster[label] = []
            self.sps_for_cluster[label].append(index)
        
        self.depot_for_cluster = {}
        for label in set(self.cluster.labels_):
            if factory_index in self.sps_for_cluster[label]:
                self.depot_for_cluster[label] = factory_index
            else:
                # Uncomment the line below to use the point closets to Factory as depot
                # self.depot_for_cluster[label] = min(self.sps_for_cluster[label], key = lambda x: self.dist_matrix[0][x])
                self.depot_for_cluster[label] = max(self.sps_for_cluster[label], key = lambda x: self.demand[x])
        print("Depots are : %s" % self.depot_for_cluster)
        self.main_cluster = list(self.depot_for_cluster.values())
        # Dairy factory should be first
        self.main_cluster.sort()
        print("Main cluster = %s" % self.main_cluster)
        
    
    def get_distance_matrix_for_cluster(self, cluster_label):
        """Create a distance matrix for all points which belong to the cluster with label = cluster_label"""
        cluster = self.sps_for_cluster[cluster_label]
        cluster_size =  len(cluster)
        distance_matrix_for_cluster = np.arange(cluster_size*cluster_size, dtype=np.float64).reshape((cluster_size, cluster_size))
        for i in range(cluster_size):
            for j in range(cluster_size):
                distance_matrix_for_cluster[i][j] = self.dist_matrix[cluster[i]][cluster[j]]
        
        return distance_matrix_for_cluster
    
    def get_time_matrix_for_cluster(self, cluster):
        """Create a travel time matrix for all points which belong to the cluster with label = cluster_label"""
        cluster_size =  len(cluster)
        time_matrix_for_cluster = np.arange(cluster_size*cluster_size, dtype=np.float64).reshape((cluster_size, cluster_size))
        for i in range(cluster_size):
            for j in range(cluster_size):
                time_matrix_for_cluster[i][j] = self.time_matrix[cluster[i]][cluster[j]]
        
        return time_matrix_for_cluster
    
    def get_demands_for_cluster(self, cluster_label):
        """Create a demand array for all points which belong to the cluster with label = cluster_label"""
        cluster = self.sps_for_cluster[cluster_label]
        cluster_size =  len(cluster)
        demands_for_cluster = np.zeros(cluster_size, dtype=np.float64)
        depots = self.depot_for_cluster.values()
        for i in range(cluster_size):
            if cluster[i] in depots:
                demands_for_cluster[i] = 0
            else:
                demands_for_cluster[i] = self.demand[cluster[i]]
        
        print("Demand for cluster %s is %s" % (cluster_label, demands_for_cluster))
        return demands_for_cluster
    
    def get_demand_for_main_cluster(self):
        """ Create a deamnd array for points in the main cluster (Factory to Depots)"""
        main_cluster = self.main_cluster
        demands = []
        # depot(stockist) -> demand of the cluster depot is in.
        demand_for_depot = {}
        for cluster_label in set(self.cluster.labels_):
            demand_for_depot[self.depot_for_cluster[cluster_label]] = sum(self.get_demands_for_cluster(cluster_label))

        print("Demand for depots = %s" % demand_for_depot)
        for depot in main_cluster:
            if depot == 0:
                # No need to deliver to factory
                demands.append(0)
            else:
                demands.append(demand_for_depot[depot])
        print("Main cluster demand = %s" % demands)
        return demands
    
    def run_vrp_all(self):
        """ This is the function that creates VRP models and solves them"""
        #Distance for Dairy to cluster center and then from each center to service points in that cluster.
        total_distance = 0.0
        main_cluster = self.main_cluster
        main_cluster_size =  len(main_cluster)
        # create a distance matrix for just the Dairy Factory + selected depots for each cluster
        self.distance_matrix_depots = np.arange(main_cluster_size*main_cluster_size, dtype=np.float64).reshape((main_cluster_size, main_cluster_size))
        for i in range(main_cluster_size):
            for j in range(main_cluster_size):
                self.distance_matrix_depots[i][j] = self.dist_matrix[main_cluster[i]][main_cluster[j]]
        
        main_cluster_demand = self.get_demand_for_main_cluster()
        
        #Find route from Dairy Factory to depots in each cluster.
        self.run_vrp(self.distance_matrix_depots, main_cluster_demand, self.vehicles_for_main, main_cluster_size, factory_index)
        total_distance, main_time, total_cost = self.calculate_total_and_print(self.distance_matrix_depots, main_cluster, self.vehicles_for_main, label_for_main_cluster)
        #total_distance, total_time = self.calulate_and_print_all(self.distance_matrix_depots, main_cluster, number_of_vehicles_for_main)
        max_cluster_time = 0
        #Now find route for each cluster
        for cluster_label in set(self.cluster.labels_):
            cluster = self.sps_for_cluster[cluster_label]
            start_point = cluster.index(self.depot_for_cluster[cluster_label])
            print("Calculating route for cluster %s : %s" % (cluster_label, cluster))
            print("Starting point is %s" % self.depot_for_cluster[cluster_label])
            distance_matrix_for_cluster = self.get_distance_matrix_for_cluster(cluster_label)
            demands_for_cluster = self.get_demands_for_cluster(cluster_label)
            self.run_vrp(distance_matrix_for_cluster, demands_for_cluster, self.vehicles_for_clusters, len(cluster), start_point)
            distance, time, cost = self.calculate_total_and_print(distance_matrix_for_cluster, cluster, self.vehicles_for_clusters, cluster_label)
            #distance, time = self.calulate_and_print_all(distance_matrix_for_cluster, cluster, number_of_vehicles_for_clusters)
            total_distance += distance
            total_cost += cost
            max_cluster_time = max(max_cluster_time, time)

        self.cost = total_cost + EMI
        # meters to kilometers
        total_distance = total_distance/1000
        self.total_solution_distance = total_distance
        
        self.total_solution_time = (main_time + max_cluster_time + 2*600)/3600
        print("Total time = (%s + %s + 2*600)/3600 = %s" % (main_time, max_cluster_time, self.total_solution_time))
        print("Total Distance of All clusters is %s KMs" % total_distance)

        #self.cost = total_distance*vehicle.rate + (len(self.vehicles_for_main) - 1)*vehicle.labour + depot*len(self.vehicles_for_clusters)*vehicle.labour + EMI + cost_for_cant
        print("Total cost = %s Rupees" % self.cost)
        self.savings = (existing_cost - self.cost)*100/existing_cost
        print("Savings = (%s - %s)*100/%s = %s" % (existing_cost, self.cost, existing_cost, self.savings) + ' %')
        print("Routes = %s" % self.routes_for_cluster)
        #print("Objective value = %s" % self.assignment.ComputeObjectiveValue())
    
    def replace_big_with_small(self, demands_for_cluster, vehicles):
        big_demands = [d for d in demands_for_cluster if d > vehicle_params['SMALL'].capacity]
        if big_demands:
            return [vehicle_params['BIG']]*len(vehicles)
        else:
            return vehicles

    def run_vrp(self, distance_matrix_for_cluster, demands_for_cluster, vehicles, cluster_size, start_point):
        #vehicles = self.replace_big_with_small(demands_for_cluster, vehicles)
        print("Vehicles are %s" % vehicles)
        number_of_vehicles = len(vehicles)
        distance_provider = lambda x, y: distance_matrix_for_cluster[x][y]
        demand_provider = lambda x, y: demands_for_cluster[y]
        routing = pywrapcp.RoutingModel(cluster_size, number_of_vehicles, start_point)
        self.routing = routing
        
        # Add a distance dimension (We are setting a limit of 500 KM on total distance)
        #routing.AddDimension(distance_provider, 0 , 500000, True, "Distance")
        #distance_dimension = routing.GetDimensionOrDie("Distance")
        #distance_dimension.SetGlobalSpanCostCoefficient(100)
        # Set max distance per vehicle, since we have different types of vehicles. 
        # Some vehicles may travel more than others.
        #for i, vehicle in enumerate(vehicles):
        #    distance_dimension.SetSpanUpperBoundForVehicle(vehicle.max_distance, i)
        
        # Add vehicle constraits (This is to make sure that a vehicle isn't trying to deliver more than it's capacity)
        routing.AddDimensionWithVehicleCapacity(demand_provider, 0, [v.capacity for v in vehicles], True, "Capacity")
        # Add vehicle cost calculation (Not sure if this is used in route optimization)
        # We don't use this cost. We ill calculate cost based on routes later.
        cost_providers = [v.cost_provider(distance_matrix_for_cluster) for v in vehicles]
        for i, vehicle in enumerate(vehicles):
            routing.SetFixedCostOfVehicle(vehicle.labour, i)
            routing.SetArcCostEvaluatorOfVehicle(cost_providers[i], i)

        

        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        self.assignment = routing.SolveWithParameters(search_parameters)
    
    def add_time_window_constraints(self, time_evaluator, num_vehicles):
        """Supposed to add time constraints but NOT TESTED & NOT USED.
           We only have an overall limit on time (all clusters combined) and
           I am not sure how to add time constraint since we evaluate each cluster separately"""
        time = "Time"
        horizon = 240
        self.routing.AddDimension(
            time_evaluator,
            horizon, # allow waiting time
            horizon, # maximum time per vehicle
            False, # don't force start cumul to zero since we are giving TW to start nodes
            time)
        time_dimension = self.routing.GetDimensionOrDie(time)
        for location_idx in range(num_vehicles):
            if location_idx == 0:
                continue
            index = self.routing.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(0, 240)
            self.routing.AddToAssignment(time_dimension.SlackVar(index))
        #for vehicle_id in range(num_vehicles):
        #    index = self.routing.Start(vehicle_id)
        #    time_dimension.CumulVar(index).SetRange(0, 0)
        #    self.routing.AddToAssignment(time_dimension.SlackVar(index))

    def calculate_total_and_print(self, distance_matrix_for_cluster, cluster, vehicles, cluster_label):
        """Once a VRP model is solved this function will go through each vehicles and print out the route for it.
           We also calculate the total distance, time and cost and return them"""
        routes = []
        number_of_vehicles = len(vehicles)
        time_matrix_for_cluster = self.get_time_matrix_for_cluster(cluster)
        total_dist = 0
        total_cost = 0
        max_time = 0
        for vehicle_id in range(number_of_vehicles):
            index = self.routing.Start(vehicle_id)
            route = []
            plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
            route_dist = 0
            route_time = 0
            while not self.routing.IsEnd(index):
                node_index = self.routing.IndexToNode(index)
                route.append(cluster[node_index])
                next_node_index = self.routing.IndexToNode(
                    self.assignment.Value(self.routing.NextVar(index)))
                route_dist += distance_matrix_for_cluster[node_index][next_node_index]
                route_time += time_matrix_for_cluster[node_index][next_node_index]
                plan_output += ' {node_index} -> '.format(
                    node_index=cluster[node_index])
                index = self.assignment.Value(self.routing.NextVar(index))

            node_index = self.routing.IndexToNode(index)
            route.append(cluster[node_index])
            routes.append(route)
            total_dist += route_dist
            cost = 0 if route_dist == 0 else (route_dist*vehicles[vehicle_id].rate + vehicles[vehicle_id].labour)
            total_cost += cost
            max_time = max(max_time, route_time)
            plan_output += ' {node_index}\n'.format(
                node_index=cluster[node_index])
            plan_output += 'Distance of the route {0}: {dist}\n'.format(
                vehicle_id,
                dist=route_dist)
            plan_output += 'Time of the route {0}: {time}\n'.format(
                vehicle_id,
                time=route_time)
            plan_output += 'Cost of the route {0}: {cost}\n'.format(
                vehicle_id,
                cost=cost)
            print(plan_output)
        
        self.routes_for_cluster[cluster_label] = routes
        print('Total Distance of all routes in cluster: {dist}'.format(dist=total_dist))
        print('Max Time of all routes in cluster: {time}'.format(time=max_time))
        print('Total Cost of all routes in cluster: {cost}'.format(cost=total_cost))
        return total_dist, max_time, total_cost
    
    def plot_clusters(self):
        """ Plots dots for each service point with a color based on which cluster it belongs to.
            All points in same cluster have same color."""
        palette = sns.color_palette()
        cluster_colors = [palette[col] for col in self.cluster.labels_]
        plot_kwds = {'alpha' : 0.8, 's' : 80, 'linewidths':0}
        plt.scatter(self.ys, self.xs, c=cluster_colors, **plot_kwds)
    
    def plot_route(self, route, color):
        """ Plots lines which conenct service points with a color based on which cluster it belongs to.
            All points in same cluster have same color."""
        for i in range(len(route)-1):
            sp1, sp2 = self.sps[route[i]], self.sps[route[i+1]]
            plt.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]], c = color)
    
    def plot_routes(self):
        """ Uses plot_route() """
        palette = sns.color_palette()
        labels = set(self.cluster.labels_)
        # -1 is for the main cluster - Factory to depot
        labels.add(label_for_main_cluster)
        for cluster_label in labels:
            for route in self.routes_for_cluster[cluster_label]:
                self.plot_route(route, palette[cluster_label])
        

    def plot_final(self):
        """ Plots everything and shows it """
        self.plot_clusters()
        self.plot_routes()
        plt.show()

# Keep trying until you get a solution with a desired savings %
savings = 0
solution = None
while True:
    solution = Solution()
    solution.find_cluster()
    solution.prepare_for_vrp()
    solution.run_vrp_all()
    if solution.savings > savings:
        savings = solution.savings
        if solution.savings > 31:
            break
    print ("Max Savings = %s" % savings)
solution.plot_final()