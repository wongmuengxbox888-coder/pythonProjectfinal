import cv2
import math
import time
import os
import mediapipe as mp
import pygame

pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
cap = cv2.VideoCapture(0)


HOLD_TIME = 1
MIN_BOX = 40
PINCH_RATIO = 0.3
RELEASE_GRACE = 0.25
OK_RATIO = 0.2
BOX_COLOR = (0, 0, 255)   

SONGS = {
    9: "mysong/BLACKHEART.mp3",   
    1: "mysong/P9D.mp3",
}
GESTURES = {
    (0, 0, 1, 0, 0): (1, "fuck"),
    (1, 0, 0, 0, 1): (9, "carabao"),
}
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
def finger_states(lm):
    
    thumb = 1 if dist(lm[4], lm[17]) > dist(lm[3], lm[17]) else 0
    others = [1 if lm[tip].y < lm[pip].y else 0
              for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    return (thumb, *others)
def classify(lm):
    f = finger_states(lm)
    hand_size = dist(lm[0], lm[9])
    if dist(lm[4], lm[8]) < OK_RATIO * hand_size and f[2:] == (1, 1, 1):
        return (10, "OK")
    return GESTURES.get(f)


def play(n):
    path = SONGS.get(n)
    if path is None:
        print(f"ท่า {n} ยังไม่ได้ผูกเพลง (เพิ่มใน SONGS)")
        return
    if not os.path.exists(path):
        print(f"ไม่เจอไฟล์ {path}")
        return
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
locked_box = None
start = None
end = None
release_start = None
cand_gesture = None
cand_start = 0
now_playing = None
stop_start = None

cv2.namedWindow("Hand")
cv2.createTrackbar("Line", "Hand", 1, 15, lambda v: None)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()
    thick = max(1, cv2.getTrackbarPos("Line", "Hand"))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    found = {}
    if result.multi_hand_landmarks:
        for lms, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hd.classification[0].label
            found[label] = lms.landmark
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

    status = ""

    # ===== ยังไม่มีกล่อง: ประกบนิ้วมือขวาเพื่อวาด =====
    if locked_box is None:
        status = "Pinch thumb + index (right hand) and drag to draw a box"
        cur = None

        if "Right" in found:
            lm = found["Right"]
            hand_size = dist(lm[0], lm[9])
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            mid = ((tx + ix) // 2, (ty + iy) // 2)

            if dist(lm[4], lm[8]) < PINCH_RATIO * hand_size:
                cur = mid
                cv2.circle(frame, mid, 12, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

        if cur is not None:
            release_start = None
            if start is None:
                start = cur
            end = cur
            cv2.rectangle(frame, start, end, (0, 255, 255), thick)
            status = "Drawing... release to finish"

        elif start is not None:
            cv2.rectangle(frame, start, end, (0, 255, 255), thick)
            if release_start is None:
                release_start = now
            elif now - release_start >= RELEASE_GRACE:
                bx1, bx2 = sorted((start[0], end[0]))
                by1, by2 = sorted((start[1], end[1]))
                if bx2 - bx1 > MIN_BOX and by2 - by1 > MIN_BOX:
                    locked_box = (bx1, by1, bx2, by2)
                    cand_gesture = None
                    now_playing = None
                    stop_start = None
                start = None
                end = None
                release_start = None
    else:
        x1, y1, x2, y2 = locked_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thick)

        g = None
        if "Right" in found:
            palm = found["Right"][9]
            px, py = int(palm.x * w), int(palm.y * h)
            if x1 <= px <= x2 and y1 <= py <= y2:
                g = classify(found["Right"])
                if g:
                    cv2.putText(frame, f"{g[0]}: {g[1]}", (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if g and g[0] == cand_gesture:
            if now - cand_start >= HOLD_TIME and g[0] != now_playing:
                play(g[0])
                now_playing = g[0]
        else:
            cand_gesture = g[0] if g else None
            cand_start = now
            if g is None:
                now_playing = None   

        status = f"Playing: {now_playing if now_playing else '-'}"

        # ---- มือซ้าย: แบ 4 นิ้วค้าง = ลบกล่อง + หยุดเพลง ----
        if "Left" in found and all(finger_states(found["Left"])[1:]):
            if stop_start is None:
                stop_start = now
            elif now - stop_start >= HOLD_TIME:
                pygame.mixer.music.stop()
                locked_box = None
                now_playing = None
                cand_gesture = None
                stop_start = None
            else:
                status = "Hold left palm... deleting box"
        else:
            stop_start = None

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Line: {thick}   |   left palm = delete box + stop   |   e = exit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Hand", frame)
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()