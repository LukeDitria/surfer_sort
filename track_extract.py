from sort import *
import numpy as np
from tqdm import trange, tqdm
import pandas as pd
from collections import defaultdict


def extract_tracks(data_csv, max_age=10, iou_threshold=0.1, bb_size=(128, 64), frame_size=(3840, 2160)):
    """
    This function uses the "SORT" algorithm to match bounding box detections across frames to create
    unique trajectories
    This function takes in a csv file of bounding box predictions per frame, it assumes that each frame has at least 1
    detection of an object and therefore there is a unique timestamp for every frame.
    """

    # Load the csv file with pandas
    pd_dataframe = pd.read_csv(data_csv)

    # Create a list of all the unique timestamps to create the "timeline"
    unique_times = np.sort(np.unique(pd_dataframe["timestamp"].to_numpy()))

    # Use the list of timestamps to estimate the "frame index" for each set of detections
    pd_dataframe["frames_indx"] = 0
    for i in range(len(unique_times)):
        pd_dataframe.loc[pd_dataframe['timestamp'] == unique_times[i], 'frames_indx'] = i

    # Remove all non-surfers AFTER we estimate the frame index (some frames have NO surfers)
    pd_dataframe = pd_dataframe.drop(pd_dataframe[pd_dataframe["class"] < 5].index)

    # create instance of SORT algorithm
    # max_age -> number of frames to wait (until the next possible detection of
    # an object) before it stops trying to track it
    # iou_threshold -> the threshold to use to determine whether a "predicted" bounding box
    # location matches a real detection
    mot_tracker = Sort(max_age=max_age, iou_threshold=iou_threshold)

    # Create a dictionary for each unique object id using defaultdict
    trajectory_logger = defaultdict(lambda: {"timestamps": [],
                                             "bounding_boxes": [],
                                             "centroid": [],
                                             "velocities": [],
                                             "number_of_detections": 0})

    # step through each frame
    for i in trange(pd_dataframe['frames_indx'].max() + 1):
        # extract all detections corresponding to the current frame
        frame_detections = pd_dataframe[pd_dataframe['frames_indx'] == i]

        # If there are any detections (of surfers) in the current frame give them to the tracker
        if len(frame_detections) > 0:

            # Calculate centroid position
            centroid_x = (frame_detections["x0"] + (frame_detections["x1"] - frame_detections["x0"]) / 2).to_numpy()
            centroid_y = (frame_detections["y0"] + (frame_detections["y1"] - frame_detections["y0"]) / 2).to_numpy()

            # Convert to pixels (not strictly necessary)
            centroid_x *= frame_size[0]
            centroid_y *= frame_size[1]

            # Get confidence
            score = frame_detections["confidence"].to_numpy()

            # "Inflate bounding box, to improve likelihood of intersection"
            # Use centroid positions to create larger BB of points --> [x0,y0,x1,y1]
            detections = np.stack([centroid_x - bb_size[0] // 2,
                                   centroid_y - bb_size[1] // 2,
                                   centroid_x + bb_size[0] // 2,
                                   centroid_y + bb_size[1] // 2,
                                   score],
                                  1).astype(int)
        else:
            # Even if there are no detections we need to sill tell the tracker, so pass an empty array
            detections = np.empty((0, 5))

        # Update the tracker with the detections
        tracks_ = mot_tracker.update(detections)

        # If the tracker thinks that any of the current detections correspond to an object it has already seen, it
        # will return the predicted position of those objects (as well as the object ID number)

        # We log the positions of each unique ID number to build up the trajectory
        for track in tracks_:
            timestamp = frame_detections["timestamp"].to_numpy()[0]

            # The last value is the ID
            id = int(track[-1])
            bbox = track[:4].tolist()
            velocities = track[4:6].tolist()

            centroid = [bbox[0] + (bbox[2] - bbox[0]) / 2,
                        bbox[1] + (bbox[3] - bbox[1]) / 2]

            # Add the new data point to the trajectory
            trajectory_logger[id]["timestamps"].append(timestamp)
            trajectory_logger[id]["bounding_boxes"].append(bbox)
            trajectory_logger[id]["centroid"].append(centroid)
            trajectory_logger[id]["velocities"].append(velocities)

            # Increment the number of detections for this object
            trajectory_logger[id]["number_of_detections"] += 1

    # Return the dictionary of trajectories
    return trajectory_logger
