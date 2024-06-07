import cv2
import pygetwindow as gw
from ultralytics import YOLO
import pydirectinput
import numpy as np
import time
from mss import mss
from threading import Thread

mutex = False

def shoot():
    global mutex
    pydirectinput.press('g')
    pydirectinput.press('h', 4)
    mutex = False

model = YOLO('best.pt')
app_window = gw.getWindowsWithTitle('Tomb Raider: Anniversary')[0]
classNames = ["bat", "wolf", "bear", "raptor", "trex"]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 0)]

with mss() as sct:
    while True:
        start_time = time.time()
        left, top, width, height = app_window.left, app_window.top, app_window.width, app_window.height
        game_screen = sct.grab({"top": top, "left": left, "width": width, "height": height})
        frame = cv2.cvtColor(np.array(game_screen), cv2.COLOR_RGBA2RGB)
        results = model(frame,imgsz=320, conf=0.6)
        classes = results[0].cpu().boxes.cls.numpy()
        xyxy = results[0].cpu().boxes.xyxy.numpy()
        boundingBoxes = zip(classes, xyxy)
        for clss, box in boundingBoxes:
            if gw.getActiveWindowTitle() == 'Tomb Raider: Anniversary' and not mutex:
                    mutex = True
                    t1 = Thread(target=shoot)
                    t1.start()
            box = [int(x) for x in box]
            clss = int(clss)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[clss], 2)
            cv2.putText(frame, classNames[clss], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[clss], 2)
        cv2.imshow('Game Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        print(f"FPS: {1 / (end_time - start_time)}")
    cv2.destroyAllWindows()