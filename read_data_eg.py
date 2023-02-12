
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', '-dp', type=str, default='.', help='json file path')
parser.add_argument('--min_detections', '-md', type=int, default=15, help='minimum number of detections to filter')

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.data_path, 'r') as fp:
        trajectory_logger = json.load(fp)

    ride_counter = 0
    for ids, trajectory in trajectory_logger.items():
        if trajectory["number_of_detections"] > args.min_detections:
            ride_counter += 1

            start_time = trajectory["timestamps"][0]
            end_time = trajectory["timestamps"][-1]
            ride_time = end_time - start_time

            centroid_start = trajectory["centroid"][0]
            centroid_end = trajectory["centroid"][-1]
            centroid_diff = np.subtract(centroid_end, centroid_start)

            dist = np.sqrt(centroid_diff[0] ** 2 + centroid_diff[0] ** 2)

            print("Ride ID %d, Ride Duration %.2f, Ride Displacement %.2f pixels, "
                  "Number of Detections %d" % (int(ids), ride_time, dist, trajectory["number_of_detections"]))
    print("############################")
    print("Total valid rides %d" % ride_counter)
