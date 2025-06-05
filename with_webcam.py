import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Paths
KNOWN_FACES_DIR = r"C:\5077\ATTENDANCE1\image_folder"
ATTENDANCE_FILE = r"C:\5077\ATTENDANCE1\Attendance.csv"

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]  # Use file name (without extension) as person's name

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"⚠️ No face found in image: {filename}")

# Track attendance
attendance_marked = set()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("🎥 Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

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

            # Mark attendance only once
            if name not in attendance_marked:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(ATTENDANCE_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, now])
                print(f"✅ Attendance marked for {name} at {now}")
                attendance_marked.add(name)

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show video
    cv2.imshow("Face Recognition - Multiple Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
