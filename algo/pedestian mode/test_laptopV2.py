from ultralytics import YOLO
import cv2

# -----------------------------
# CONFIG
# -----------------------------
CONF_THRESHOLD = 0.4

VERY_CLOSE_TH = 0.10
CLOSE_TH = 0.03
MIN_AREA_TH = 0.005

IMG_SIZE = 320

# Crowd logic thresholds
CROWDED_NEARBY_TH = 3          # 3 or more nearby people -> crowded
HEAVY_CROWDED_NEARBY_TH = 4    # 4 or more nearby people -> heavily crowded
MIN_ZONES_FOR_CROWD = 2        # must occupy at least 2 zones (left/center/right)

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
    far_count = 0
    close_count = 0
    very_close_count = 0
    occupied_zones = set()

    # -----------------------------
    # PROCESS DETECTIONS
    # -----------------------------
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

        # ignore very tiny detections
        if area_ratio < MIN_AREA_TH:
            continue

        # distance category
        if area_ratio > VERY_CLOSE_TH:
            distance = "very_close"
            very_close_count += 1
        elif area_ratio > CLOSE_TH:
            distance = "close"
            close_count += 1
        else:
            distance = "far"
            far_count += 1

        # horizontal position
        center_x = (x1 + x2) / 2

        if center_x < frame_width / 3:
            direction = "left"
            occupied_zones.add("left")
        elif center_x < 2 * frame_width / 3:
            direction = "center"
            occupied_zones.add("center")
        else:
            direction = "right"
            occupied_zones.add("right")

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
    nearby_people = [p for p in people if p["distance"] in ["close", "very_close"]]
    nearby_count = len(nearby_people)
    zone_count = len(occupied_zones)

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if nearby_count >= HEAVY_CROWDED_NEARBY_TH and zone_count >= MIN_ZONES_FOR_CROWD:
        output = "heavily crowded area"

    elif nearby_count >= CROWDED_NEARBY_TH and zone_count >= MIN_ZONES_FOR_CROWD:
        output = "crowded area"

    elif total_people > 0:
        # if not crowded, always give individual guidance
        # choose the closest person (largest area)
        closest = max(people, key=lambda p: p["area"])
        output = f"1 person {closest['direction']}"

    else:
        output = "no people"

    # only print when output changes
    if output != prev_output:
        print(output)
        prev_output = output

    # show summary on screen
    debug_text = (
        f"{output} | P:{total_people} F:{far_count} "
        f"C:{close_count} VC:{very_close_count} Z:{zone_count}"
    )

    cv2.putText(
        frame,
        debug_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv2.imshow("Laptop Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()