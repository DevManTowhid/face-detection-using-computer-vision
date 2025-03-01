import cv2
import winsound  # For beep sound (Windows only)
import threading  # To prevent sound blocking video processing

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam (Use CAP_DSHOW to fix webcam lag on Windows)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()
    
    if not ret:
        print("❌ Error: Couldn't read frame!")
        break

    # Convert frame to grayscale (Haarcascade works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using a fast detection method
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        winsound.Beep(100,200)
   

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display face details (coordinates and size)
        details = f"X:{x}, Y:{y}, W:{w}, H:{h}"
        cv2.putText(frame, details, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

        face_detected = True  # Set flag to true when face is detected

    # Show the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
