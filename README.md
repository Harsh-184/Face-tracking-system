Face Tracking

What it does:

Tracks faces from a webcam or video file using MediaPipe.

Two modes: fast Face Detection (bounding boxes) or detailed Face Mesh (468 landmarks).

Can save an annotated MP4 and a CSV log of detections.

Prereqs:

Use your project’s Python 3.11 virtual environment with OpenCV and MediaPipe installed.

How to run (webcam):

Face Detection (boxes): python -m src.face_track --source 0 --save-video --csv

Face Mesh (landmarks): python -m src.face_track --source 0 --mesh --save-video --csv

How to run (video file):

Replace the path with your file: python -m src.face_track --source "C:\path\to\clip.mp4" --mesh --save-video --csv

CLI options:

--source : 0 for webcam, or full path to a video file

--mesh : enable Face Mesh instead of simple detection

--max-faces : maximum number of faces for mesh mode (default 2)

--outdir : folder for outputs (default “output”)

--save-video : write faces_annotated.mp4 with drawings

--csv : write faces.csv with basic per-frame logs

Outputs (written to the outdir):

faces_annotated.mp4 (only if you pass --save-video)

faces.csv (only if you pass --csv). For detection: time_s, frame, face_idx, xmin, ymin, xmax, ymax (relative 0–1). For mesh: time_s, frame, face_idx, landmark_count.

Tips:

If the webcam window is black or fails to open, try a different index (use --source 1, 2, …), close other apps using the camera, or use a video file.

If a video path won’t open, check the path and try a different backend by re-encoding to a standard H.264 MP4.

Performance: disable --save-video and/or reduce resolution if playback is choppy.
