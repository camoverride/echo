import cv2

# Set the RTSP URL
rtsp_url = "rtsp://admin:admin123@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    # Retrieve frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Retrieve frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Display stream details
    print(f"Resolution: {width}x{height}")
    print(f"Frame Rate: {fps} FPS")

# Release the capture when done
cap.release()
