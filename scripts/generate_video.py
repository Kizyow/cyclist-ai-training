from ultralytics import YOLO
import cv2

model_best = YOLO("model_best_transfer_learning_no_freeze.pt")  # ou yolov8n.pt, yolov8m.pt, etc.
model_best.to('cuda')

video_path = "video_cut.mp4"

cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Prédiction
    results = model_best.predict(frame, conf=0.6, iou=0.7, agnostic_nms=True)

    # Annote
    annotated_frame = results[0].plot()

    # Afficher
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()