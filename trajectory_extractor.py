import argparse
import track_extract as te
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_source_dir', '-srd', type=str, default='.', help='root source dir')
parser.add_argument('--root_save_dir', '-svd', type=str, default='.', help='root save dir')
parser.add_argument('--iou_thresh', type=float, default=0.1, help='IOU threshold for bb tracker')
parser.add_argument('--max_age', type=int, default=10, help='Max number of frames to wait for tracking')

parser.add_argument('--bb_size', type=int, default=64, help='Artificial bounding box size')
parser.add_argument('--bb_ratio', type=int, default=2, help='x/ bounding box aspect ratio')
parser.add_argument("--restart_job", '-rj', action='store_true', help="overwrite all previous extractions")

args = parser.parse_args()

if __name__ == '__main__':
    save_dir = os.path.join(args.root_save_dir, args.root_source_dir.split("/")[-1])
    valid_file_ext = ['csv']
    print('EXTRACTING!')

    # Use os.walk() to loop through the nested directories
    for root, dirs, files in os.walk(args.root_source_dir):
        if len(files) > 0:
            sub_dir = save_dir + root.replace(args.root_source_dir, '')
            # Loop through the list of files
            for file in files:
                file_path = os.path.join(root, file)
                file_type = file.split(".")[-1]

                if not file.split(".")[-2] == "detections":
                    continue

                new_file_name = "_".join(file.split(".")[:-1]) + ".json"
                new_file_path = os.path.join(sub_dir, new_file_name)

                if not os.path.isfile(new_file_path) or args.restart_job:
                    if file_type.lower() in valid_file_ext:
                        if not os.path.isdir(sub_dir):
                            os.makedirs(sub_dir)

                        trajectories = te.extract_tracks(data_csv=file_path, max_age=args.max_age,
                                                         iou_threshold=args.iou_thresh,
                                                         bb_size=(args.bb_ratio * args.bb_size, args.bb_size),
                                                         frame_size=(3840, 2160))

                        # Save the trajectories dict as a json file
                        with open(new_file_path, 'w') as fp:
                            json.dump(trajectories, fp)

    print('COMPLETE!')
    print('SAVED TO %s' % args.root_save_dir)

