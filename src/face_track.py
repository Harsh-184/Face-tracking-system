# src/face_track.py
import argparse
import os
import csv
import cv2
import numpy as np
import mediapipe as mp


def open_capture(source):
    """
    Robustly open a video source.
    - If source is an int → try common Windows webcam backends then ANY.
    - If source is a path → try FFMPEG/MSMF/ANY (covers most codecs).
    """
    if isinstance(source, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)  # last-ditch
    else:
        for backend in (cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Normalize source type
    src = args.source
    if src.isdigit():
        src = int(src)

    cap = open_capture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Optional video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_mp4 = os.path.join(args.outdir, "faces_annotated.mp4")
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H))

    # MediaPipe setup
    if args.mesh:
        face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=args.max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        draw = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles
    else:
        face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    # Optional CSV
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = os.path.join(args.outdir, "faces.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        if args.mesh:
            csv_writer.writerow(["time_s", "frame", "face_idx", "landmark_count"])
        else:
            csv_writer.writerow(["time_s", "frame", "face_idx", "xmin", "ymin", "xmax", "ymax"])  # relative coords

    frame_idx = 0
    win_name = "Face tracking (q=quit)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_s = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if args.mesh:
            res = face.process(rgb)
            if res.multi_face_landmarks:
                for i, lmset in enumerate(res.multi_face_landmarks):
                    # Draw landmarks (tesselation + contours)
                    draw.draw_landmarks(
                        frame,
                        lmset,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=styles.get_default_face_mesh_tesselation_style(),
                    )
                    draw.draw_landmarks(
                        frame,
                        lmset,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
                    )
                    if csv_writer:
                        csv_writer.writerow([f"{t_s:.3f}", frame_idx, i, 468])
        else:
            res = face.process(rgb)
            if res.detections:
                for i, det in enumerate(res.detections):
                    bbox = det.location_data.relative_bounding_box
                    # relative → pixel
                    x1 = int(max(0, bbox.xmin) * W)
                    y1 = int(max(0, bbox.ymin) * H)
                    x2 = int(min(1.0, bbox.xmin + bbox.width) * W)
                    y2 = int(min(1.0, bbox.ymin + bbox.height) * H)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"face {i}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if csv_writer:
                        # store relative coords 0..1 for portability
                        csv_writer.writerow([f"{t_s:.3f}", frame_idx, i,
                                             bbox.xmin, bbox.ymin,
                                             bbox.xmin + bbox.width, bbox.ymin + bbox.height])

        # HUD
        cv2.putText(frame, f"t={t_s:.2f}s  fps~{fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Saved] {os.path.join(args.outdir, 'faces_annotated.mp4')}")
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print(f"[Saved] {os.path.join(args.outdir, 'faces.csv')}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 for webcam, or full path to a video file")
    ap.add_argument("--mesh", action="store_true", help="Enable Face Mesh (468 landmarks) instead of simple face boxes")
    ap.add_argument("--max-faces", type=int, default=2, help="Max faces for mesh mode")
    ap.add_argument("--outdir", default="output", help="Folder for saved artifacts")
    ap.add_argument("--save-video", action="store_true", help="Save annotated MP4")
    ap.add_argument("--csv", action="store_true", help="Write faces.csv log")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
