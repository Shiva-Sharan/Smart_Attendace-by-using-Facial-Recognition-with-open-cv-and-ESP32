import cv2
import face_recognition
import os
import csv
from datetime import datetime
import urllib.request
import numpy as np

# ESP32-CAM stream URL (change to your actual IP)
ESP32_URL = "http://192.168./stream"

# Attendance settings
KNOWN_FACES_DIR = r"C:\5077\ATTENDANCE1\image_folder"
ATTENDANCE_FILE = r"C:\5077\ATTENDANCE1\Attendance.csv"
attendance_marked = set()

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"⚠️ No face found in {filename}")

# Open ESP32 MJPEG stream
stream = urllib.request.urlopen(ESP32_URL)
bytes_stream = b''

print("📡 Connecting to ESP32-CAM stream...")

while True:
    bytes_stream += stream.read(1024)
    a = bytes_stream.find(b'\xff\xd8')  # JPEG start
    b = bytes_stream.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_stream[a:b+2]
        bytes_stream = bytes_stream[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin() if face_distances.size > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in attendance_marked:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ATTENDANCE_FILE, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, now])
                    print(f"✅ Attendance marked for {name} at {now}")
                    attendance_marked.add(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ESP32-CAM Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
