from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO(r'P:\SmartHacks\apple\runs\detect\train\weights\best.pt')

class_name=['Apple']


def apple_count(video_path):
    video_path_out = '{}_out.mp4'.format(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    threshold = 0.5

    # Variable to keep track of the maximum number of apples detected in any frame
    max_apple_count = 0

    while ret:
        # Perform inference on the frame
        results = model(frame)[0]

        # Initialize apple counter for the current frame
        apple_count = 0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Count apples (assuming class ID for apple is 0)
                if int(class_id) == 0:  # Adjust this based on your class labels
                    apple_count += 1

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Update the maximum apple count if the current frame has more apples
        if apple_count > max_apple_count:
            max_apple_count = apple_count

        # Display apple count on the frame (optional)
        cv2.putText(frame, f'Apples: {apple_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # Write the frame with bounding boxes and apple count
        out.write(frame)
        ret, frame = cap.read()
    return(max_apple_count)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    

    # Print the maximum apple count detected in any frame
    



# In[ ]:




