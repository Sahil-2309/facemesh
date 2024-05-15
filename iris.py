import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def get_gradient_color(landmark_index, total_landmarks):
    start_color = (255, 0, 127)
    end_color = (127, 0, 255)

    weight = landmark_index / (total_landmarks - 1)

    b = int(start_color[0] * (1 - weight) + end_color[0] * weight)
    g = int(start_color[1] * (1 - weight) + end_color[1] * weight)
    r = int(start_color[2] * (1 - weight) + end_color[2] * weight)

    return b, g, r

def get_connection_color(landmark_idx, landmark):
    num_landmarks = len(landmark.landmark)
    return get_gradient_color(landmark_idx, num_landmarks)

def update_frame():
    global face_count  

    ret, frame = cap.read()

    if ret:
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks is not None:
            face_count = len(results.multi_face_landmarks)
        else:
            face_count = 0

        if show_face_mesh and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=get_connection_color(0, face_landmarks))
                )

        cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

      
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image=img)

        label.imgtk = img
        label.config(image=img)

        label.after(10, update_frame)
    else:
        root.quit()

face_count = 0
show_face_mesh = True

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=8,
                                 min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Face Mesh Detection")

label = tk.Label(root)
label.pack()

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
