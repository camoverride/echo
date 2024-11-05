import cv2

# Set the RTSP URL for the Amcrest camera
rtsp_url = "rtsp://admin:admin123@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Display the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Amcrest Camera Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
