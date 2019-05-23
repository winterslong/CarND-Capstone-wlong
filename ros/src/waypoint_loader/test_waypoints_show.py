#!/usr/bin/env python

import os
import csv
import tf
import numpy as np
import matplotlib.pyplot as plt
"""
self.lights.len=8
lights[0] : [1172.183, 1186.299],
lights[1] : [1584.065, 1156.953], 
lights[2] : [2126.353, 1550.636], 
lights[3] : [2178.291, 1819.328],
lights[4] : [1469.499, 2946.97],
lights[5] : [797.9147, 2905.59],
lights[6] : [160.8088, 2279.929], 
lights[7] : [363.378, 1553.731], 
"""
traffic_lights = np.array([
	[1172.183, 1186.299],
	[1584.065, 1156.953], 
	[2126.353, 1550.636], 
	[2178.291, 1819.328],
	[1469.499, 2946.97],
	[797.9147, 2905.59],
	[160.8088, 2279.929], 
	[363.378, 1553.731]])
tl_x = traffic_lights[:,0]
tl_y = traffic_lights[:,1]

stoplines=np.array([
	[1148.56, 1184.65],
	[1559.2, 1158.43],
	[2122.14, 1526.79],
	[2175.237, 1795.71],
	[1493.29, 2947.67],
	[821.96, 2905.8],
	[161.76, 2303.82],
	[351.84, 1574.65]])
stop_x = stoplines[:,0]
stop_y = stoplines[:,1]

CSV_HEADER = ['x', 'y', 'z', 'yaw']
waypointsX = []
waypointsY = []
waypointsYaw = []
def quaternion_from_yaw(yaw):
	return tf.transformations.quaternion_from_euler(0., 0., yaw)
def load_waypoints(fname):
	with open(fname) as wfile:
		reader = csv.DictReader(wfile, CSV_HEADER)
		for wp in reader:
			waypointsX.append(float(wp['x']))
			waypointsY.append(float(wp['y']))
			waypointsYaw.append(quaternion_from_yaw(float(wp['yaw'])))

path = "wp_yaw_const_wlong.csv"
print "new_waypoint_loader: path=", path
if os.path.isfile(path):
	load_waypoints(path)

print "waypointsX.len=", len(waypointsX)
print "waypointsY.len=", len(waypointsY)
print "waypointsYaw.len=", len(waypointsYaw)
	
plt.axes().set(xlabel='x', ylabel='y', title='base waypoints/stoplines')
plt.plot(tl_x, tl_y, 'o', color='red')
plt.plot(stop_x, stop_y, 'x', color='blue')
plt.plot(waypointsX, waypointsY, color='green')
plt.show()
	
	
	
	