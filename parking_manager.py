import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def box_overlaps_space(box, polygon, threshold=0.20):
    x1, y1, x2, y2 = box

    parking_poly = np.array(polygon, dtype=np.float32)

    car_box = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    parking_area = cv2.contourArea(parking_poly)

    if parking_area <= 0:
        return False

    intersection_area, _ = cv2.intersectConvexConvex(
        parking_poly, car_box
    )

    overlap_ratio = intersection_area / parking_area

    return overlap_ratio >= threshold


def line_side(point, line):
    (x1, y1), (x2, y2) = line
    value = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
    if abs(value) < 30:  # Small dead zone to reduce counting jitter.
        return 0
    return 1 if value > 0 else -1


def duration_text(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}'


def write_reports(events, hourly_count, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(events, columns=['time', 'event', 'space', 'track_id', 'duration_seconds'])
    df.to_csv(output_path / 'parking_events.csv', index=False)
    df_hourly = pd.DataFrame(sorted(hourly_count.items()), columns=['hour', 'cars_arrived'])
    df_hourly.to_csv(output_path / 'hourly_count.csv', index=False)
    print('Saved reports in:', output_path)
    print(df_hourly.to_string(index=False) if not df_hourly.empty else 'No arrivals counted.')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/parking.mp4', help='Video path or camera index (0)')
    parser.add_argument('--config', default='config/spaces.json')
    parser.add_argument('--model', default='yolo11n.pt', help='Ultralytics pretrained detection model')
    parser.add_argument('--start-time', default=None, help='For files: YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
    spaces = settings['spaces']
    entry_line = settings.get('entry_line')
    direction = settings.get('entry_direction', 'negative_to_positive')
    lost_seconds = float(settings.get('lost_seconds', 2))
    if direction not in ['negative_to_positive', 'positive_to_negative']:
        raise ValueError('Invalid entry_direction in config.')

    is_camera = args.video.isdigit()
    source = int(args.video) if is_camera else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError('Cannot open camera/video: ' + args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path / 'parking_result.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create output video.')

    model = YOLO(args.model)  # We use pretrained detection; no custom training needed.
    start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S') if args.start_time else datetime.now()
    states = {s['name']: {'occupied': False, 'start_time': None,
                          'last_seen_seconds': None, 'track_id': None} for s in spaces}
    events = []
    hourly_count = defaultdict(int)
    last_track_side = {}
    counted_crossings = set()
    total_in, total_out = 0, 0
    frame_number = 0
    current_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = datetime.now() if is_camera else start_time + timedelta(seconds=frame_number / fps)
            seconds = (current_time - start_time).total_seconds()
            frame_number += 1
            hour = current_time.strftime('%Y-%m-%d %H:00')
            time_str = current_time.isoformat(timespec='seconds')

            results = model.track(
                    frame,
                    persist=True,
                    tracker='bytetrack.yaml',
                    classes=[2, 5, 7],
                    conf=0.15,
                    imgsz=1280,
                    verbose=False
            )
            boxes = results[0].boxes
            detected = []
            if boxes is not None and len(boxes) > 0:
                coords = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coords)
                for coords_one, track_id in zip(coords, track_ids):
                    x1, y1, x2, y2 = coords_one.tolist()
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detected.append({'id': track_id, 'center': center, 'box': (x1, y1, x2, y2)})
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f'Car ID: {track_id}', (x1, max(22, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if entry_line is not None and track_id is not None:
                        side = line_side(center, entry_line)
                        old_side = last_track_side.get(track_id)
                        if side != 0:
                            if old_side is not None and old_side != side:
                                is_entry = (old_side < side) if direction == 'negative_to_positive' else (old_side > side)
                                event_type = 'entry' if is_entry else 'exit'
                                if (track_id, event_type) not in counted_crossings:
                                    counted_crossings.add((track_id, event_type))
                                    events.append([time_str, event_type, '', track_id, ''])
                                    if is_entry:
                                        total_in += 1
                                        hourly_count[hour] += 1
                                    else:
                                        total_out += 1
                            last_track_side[track_id] = side

            for space in spaces:
                name = space['name']
                state = states[name]
                found = [d for d in detected if box_overlaps_space(d['box'], space['points'])]
                if found:
                    # If possible, keep the currently assigned tracking ID.
                    car = next((d for d in found if d['id'] == state['track_id']), found[0])
                    if not state['occupied']:
                        state['occupied'] = True
                        state['start_time'] = current_time
                        state['track_id'] = car['id']
                        events.append([time_str, 'parked', name, car['id'], ''])
                        if entry_line is None:
                            total_in += 1
                            hourly_count[hour] += 1
                    state['last_seen_seconds'] = seconds
                elif state['occupied'] and (seconds - state['last_seen_seconds'] >= lost_seconds):
                    parked_seconds = (current_time - state['start_time']).total_seconds()
                    events.append([time_str, 'left_space', name, state['track_id'], round(parked_seconds, 1)])
                    if entry_line is None:
                        total_out += 1
                    state.update({'occupied': False, 'start_time': None,
                                  'last_seen_seconds': None, 'track_id': None})

                polygon = np.array(space['points'], dtype=np.int32)
                color = (0, 0, 255) if state['occupied'] else (0, 255, 0)
                cv2.polylines(frame, [polygon], True, color, 2)
                label = name + (' : BUSY' if state['occupied'] else ' : FREE')
                if state['occupied']:
                    label += ' ' + duration_text((current_time - state['start_time']).total_seconds())
                cv2.putText(frame, label, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 2)

            if entry_line is not None:
                cv2.line(frame, tuple(entry_line[0]), tuple(entry_line[1]), (255, 0, 255), 2)
            occupied = sum(s['occupied'] for s in states.values())
            cv2.rectangle(frame, (0, 0), (width, 65), (25, 25, 25), -1)
            cv2.putText(frame, f'Occupied: {occupied}  Free: {len(spaces) - occupied}',
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Arrivals: {total_in}  Departures: {total_out}  This hour: {hourly_count[hour]}',
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
            if not args.no_show:
                cv2.imshow('Smart Parking Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        for name, state in states.items():
            if state['occupied']:
                duration = (current_time - state['start_time']).total_seconds()
                events.append([current_time.isoformat(timespec='seconds'), 'still_parked_at_end',
                               name, state['track_id'], round(duration, 1)])
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        write_reports(events, hourly_count, output_path)


if __name__ == '__main__':
    run()
