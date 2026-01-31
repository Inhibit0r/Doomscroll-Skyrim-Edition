import cv2
import mediapipe as mp
import time
import subprocess
from pathlib import Path


# ================== CONFIG ==================
TIMER = 2.0              # задержка перед первым запуском
LOOK_DOWN = 0.25
DEBOUNCE = 0.45

LOOK_AT_CAMERA_MIN = 0.40
LOOK_AT_CAMERA_MAX = 0.60

DEBUG_EYES = True
# ============================================


def osascript(script: str) -> None:
    subprocess.run(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def play_video(video_path: Path) -> None:
    script = f'''
    tell application "QuickTime Player"
        activate
        set doc to open POSIX file "{video_path}"
        tell doc
            play
            set presenting to false
            tell front window
                set bounds to {{25, 45, 415, 825}}
            end tell
        end tell
    end tell
    '''
    osascript(script)


def close_video(video_path: Path) -> None:
    script = f'''
    tell application "QuickTime Player"
        repeat with d in documents
            try
                if (name of d) is "{video_path.name}" then
                    stop d
                    close d saving no
                end if
            end try
        end repeat
    end tell
    '''
    osascript(script)


def draw_warning(frame, text):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x1 + box_w + 2, y1 + box_h + 2), (80, 255, 160), 4)

    cv2.putText(
        frame,
        text,
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def main():
    video = Path("./assets/skyrim-skeleton.mp4").resolve()
    if not video.exists():
        print("Video not found")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("No webcam")
        return

    doomscroll_start = None
    video_open = False

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        now = time.time()
        doomscrolling = False
        looking_at_camera = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            left = [lm[145], lm[159]]
            right = [lm[374], lm[386]]
            l_iris = lm[468]
            r_iris = lm[473]

            l_ratio = (l_iris.y - left[1].y) / (left[0].y - left[1].y + 1e-6)
            r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)
            avg_ratio = (l_ratio + r_ratio) / 2

            threshold = DEBOUNCE if video_open else LOOK_DOWN
            doomscrolling = avg_ratio < threshold
            looking_at_camera = LOOK_AT_CAMERA_MIN <= avg_ratio <= LOOK_AT_CAMERA_MAX

            if DEBUG_EYES:
                box = 40
                lx = int((left[0].x + left[1].x) / 2 * w)
                ly = int((left[0].y + left[1].y) / 2 * h)
                rx = int((right[0].x + right[1].x) / 2 * w)
                ry = int((right[0].y + right[1].y) / 2 * h)

                cv2.rectangle(frame, (lx - box, ly - box), (lx + box, ly + box), (0, 255, 0), 2)
                cv2.rectangle(frame, (rx - box, ry - box), (rx + box, ry + box), (0, 255, 0), 2)

        else:
            # ЛИЦА НЕТ → ДУМСКРОЛЛ
            doomscrolling = True

        # ================= STATE MACHINE =================

        if doomscrolling:
            if doomscroll_start is None:
                doomscroll_start = now

            if (now - doomscroll_start) >= TIMER:
                if not video_open:
                    play_video(video)
                    video_open = True
        else:
            doomscroll_start = None

        # мгновенный перезапуск, если видео закончилось
        if doomscrolling and video_open:
            video_open = False

        # ОСОЗНАННОЕ ВОЗВРАЩЕНИЕ ВНИМАНИЯ
        if res.multi_face_landmarks and looking_at_camera:
            doomscroll_start = None
            video_open = False
            close_video(video)

        if doomscrolling:
            draw_warning(frame, "DOOMSCROLLING ALARM")

        cv2.imshow("lock in", frame)
        if cv2.waitKey(1) == 27:
            break

    close_video(video)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
