# Adapted from: https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py

from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(760, 260), (1230, 260), (1310,820), (700, 820)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255,0,0),  # BGR Value
        "text_color": (255,255,255),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(209, 254),(665, 258),(573, 818),(5, 806),(5, 750)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255,0,0),  # BGR Value
        "text_color": (255,255,255),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(1325, 260),(1917, 260),(1917, 820),(1441, 820)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255,0,0),  # BGR Value
        "text_color": (255,255,255),  # Region Text Color
    },
]
"""
Args:
    weights (str): Model weights path.
    source (str): Video file path.
    device (str): processing device cpu, 0, 1
    view_img (bool): Show results.
    save_img (bool): Save results.
    exist_ok (bool): Overwrite existing files.
    classes (list): classes to detect and track
    line_thickness (int): Bounding box thickness.
    track_thickness (int): Tracking line thickness
    region_thickness (int): Region thickness.
"""

WEIGHTS="u_yolo_artifacts/yolov8l_1920_noaug/weights/best.pt"
DEVICE="cpu"
view_img=False
save_img=True
classes=None
line_thickness=2
track_thickness=2
region_thickness=2

    
"""
Run Region counting on a video using YOLOv8 and ByteTrack.

Supports movable region for real time counting inside specific area.
Supports multiple regions counting.
Regions can be Polygons or rectangle in shape

"""

vid_frame_count = 0

# Setup Model
model = YOLO(f"{WEIGHTS}")

# Extract classes names
names = model.model.names

# Video setup
videocapture = cv2.VideoCapture("data/WoodLineVideoShort.mp4")
frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"MJPG") # [MJPG, mp4v]

# Output setup
video_writer = cv2.VideoWriter("WoodLineVideoShort.mp4", fourcc, fps, (frame_width, frame_height))

# Iterate over video frames
while videocapture.isOpened():
    success, frame = videocapture.read()
    if not success:
        break
    vid_frame_count += 1

    # Extract the results
    results = model.track(frame, persist=True, classes=classes)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        annotator = Annotator(frame, line_width=line_thickness, example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            annotator.box_label(box, str(names[cls]), color=colors(cls, True))
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

            track = track_history[track_id]  # Tracking Lines plot
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

            # Check if detection inside region
            for region in counting_regions:
                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                    region["counts"] += 1

    # Draw regions (Polygons/Rectangles)
    for region in counting_regions:
        region_label = str(region["counts"])
        region_color = region["region_color"]
        region_text_color = region["text_color"]

        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

        text_size, _ = cv2.getTextSize(
            region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
        )
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

    if view_img:
        if vid_frame_count == 1:
            cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
        cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

    if save_img:
        video_writer.write(frame.astype('uint8'))

    for region in counting_regions:  # Reinitialize count for each region
        region["counts"] = 0

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

del vid_frame_count
video_writer.release()
videocapture.release()
cv2.destroyAllWindows()
