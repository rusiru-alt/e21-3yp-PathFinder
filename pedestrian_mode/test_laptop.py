from ultralytics import YOLO
import cv2

# -----------------------------
# CONFIG
# -----------------------------
CONF_THRESHOLD = 0.4

VERY_CLOSE_TH = 0.10
CLOSE_TH = 0.03
MIN_AREA_TH = 0.005

CROWD_COUNT_TH = 5
CLOSE_COUNT_TH = 2

IMG_SIZE = 320

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# OPEN LAPTOP CAMERA
# 0 = default webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height

prev_output = ""

print("Press 'q' to quit.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    boxes = results[0].boxes

    people = []
    close_count = 0
    very_close_count = 0

    for box in boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_ratio = area / frame_area

        if area_ratio < MIN_AREA_TH:
            continue

        if area_ratio > VERY_CLOSE_TH:
            distance = "very_close"
            very_close_count += 1
        elif area_ratio > CLOSE_TH:
            distance = "close"
            close_count += 1
        else:
            distance = "far"

        center_x = (x1 + x2) / 2

        if center_x < frame_width / 3:
            direction = "left"
        elif center_x < 2 * frame_width / 3:
            direction = "center"
        else:
            direction = "right"

        people.append({
            "area": area,
            "direction": direction,
            "distance": distance
        })

        # draw box for laptop testing
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{direction}, {distance}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    total_people = len(people)

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if (
        total_people >= CROWD_COUNT_TH or
        close_count >= CLOSE_COUNT_TH or
        very_close_count >= 1
    ):
        output = "CROWDED"
    elif total_people > 0:
        closest = max(people, key=lambda p: p["area"])
        output = f"1 person {closest['direction']}"
    else:
        output = "no people"

    if output != prev_output:
        print(output)
        prev_output = output

    # show summary on screen
    debug_text = f"{output} | P:{total_people} C:{close_count} VC:{very_close_count}"
    cv2.putText(
        frame,
        debug_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Laptop Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()