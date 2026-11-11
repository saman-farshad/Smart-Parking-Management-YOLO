import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='data/parking.mp4', help='Video file or camera index, e.g. 0')
parser.add_argument('--output', default='config/spaces.json')
args = parser.parse_args()
source = int(args.video) if args.video.isdigit() else args.video
cap = cv2.VideoCapture(source)
ret, frame = cap.read()
cap.release()
if not ret:
    raise ValueError('Cannot read the video. Check --video.')

spaces = []
points = []
entry_line = []
mode = 'space'


def on_mouse(event, x, y, flags, param):
    global mode, points, entry_line
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'line':
            entry_line.append([x, y])
            if len(entry_line) == 2:
                mode = 'space'
        else:
            points.append([x, y])


cv2.namedWindow('Parking Setup')
cv2.setMouseCallback('Parking Setup', on_mouse)
while True:
    view = frame.copy()
    for space in spaces:
        polygon = np.array(space['points'], dtype=np.int32)
        cv2.polylines(view, [polygon], True, (0, 255, 0), 2)
        cv2.putText(view, space['name'], tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) > 0:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
        for pt in points:
            cv2.circle(view, tuple(pt), 4, (0, 255, 255), -1)
    if len(entry_line) > 0:
        cv2.circle(view, tuple(entry_line[0]), 6, (255, 0, 0), -1)
    if len(entry_line) == 2:
        cv2.line(view, tuple(entry_line[0]), tuple(entry_line[1]), (255, 0, 0), 2)

    cv2.rectangle(view, (0, 0), (view.shape[1], 75), (20, 20, 20), -1)
    cv2.putText(view, 'Click space corners -> N: finish space | L: entry line (2 clicks)',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)
    cv2.putText(view, 'U: undo point | S: save | ESC: cancel | mode: ' + mode,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)
    cv2.imshow('Parking Setup', view)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('n') and len(points) >= 3:
        spaces.append({'name': f'P{len(spaces)+1}', 'points': points[:]})
        points = []
    elif key == ord('l'):
        if points:
            print('Finish or undo the current polygon first.')
        else:
            entry_line = []
            mode = 'line'
    elif key == ord('u'):
        if mode == 'line' and entry_line:
            entry_line.pop()
        elif points:
            points.pop()
    elif key == ord('s'):
        if not spaces:
            print('Add at least one parking space, then press S.')
            continue
        settings = {
            'spaces': spaces,
            'entry_line': entry_line if len(entry_line) == 2 else None,
            'entry_direction': 'negative_to_positive',
            'lost_seconds': 2.0
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(settings, indent=2), encoding='utf-8')
        print('Saved:', args.output)
        print('If entry/exit are reversed, switch entry_direction in JSON.')
        break
    elif key == 27:
        break
cv2.destroyAllWindows()
