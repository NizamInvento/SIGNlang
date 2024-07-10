import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import pickle

class LoginUI:
    def __init__(self, root, gesture_ui):
        self.root = root
        self.gesture_ui = gesture_ui
        self.root.title("Login")
        
        # Create login UI elements
        self.label_username = tk.Label(root, text="Username:")
        self.label_username.grid(row=0, column=0, padx=5, pady=5)
        self.entry_username = tk.Entry(root)
        self.entry_username.grid(row=0, column=1, padx=5, pady=5)
        
        self.label_password = tk.Label(root, text="Password:")
        self.label_password.grid(row=1, column=0, padx=5, pady=5)
        self.entry_password = tk.Entry(root, show="*")
        self.entry_password.grid(row=1, column=1, padx=5, pady=5)
        
        self.button_login = tk.Button(root, text="Login", command=self.login)
        self.button_login.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
    def login(self):
        # Perform login authentication (replace with your authentication logic)
        username = self.entry_username.get()
        password = self.entry_password.get()
        if username == "admin" and password == "admin":
            self.root.destroy()  # Close login window
            self.gesture_ui.start_recognition()  # Start gesture recognition UI
        else:
            tk.messagebox.showerror("Login Failed", "Invalid username or password")

class GestureRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        
        # Load SVM model
        model_dict = pickle.load(open('./svm_model.p', 'rb'))
        self.model = model_dict['model']
        self.labels_dict = {
            0: 'HOME', 1: 'THUMBS UP', 2: 'HELP q', 
            3: 'D', 4: 'E', 98: 'super aayind', 
            99: 'I love you ttooo', 100: 'Not recognisable image', 
            101: 'Arun Sir njangade aiswaryam'
        }
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        # Create UI elements
        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition, state=tk.DISABLED)
        self.start_button.pack()
        
        self.stop_button = tk.Button(root, text="Stop Recognition", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack()
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.cap = cv2.VideoCapture(0)
        self.recognizing = False
        self.update()
        
    def start_recognition(self):
        self.recognizing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
    def stop_recognition(self):
        self.recognizing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
    def update(self):
        if self.recognizing:
            ret, frame = self.cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, 
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    centroid_x = np.mean(x_)
                    centroid_y = np.mean(y_)
                    normalized_x = [x - centroid_x for x in x_]
                    normalized_y = [y - centroid_y for y in y_]

                    data_aux.extend(normalized_x)
                    data_aux.extend(normalized_y)

                    concatenated_data_aux = np.concatenate(data_aux)
                    prediction = self.model.predict([concatenated_data_aux])
                    predicted_character = self.labels_dict[int(prediction[0])]
                    self.canvas.delete("all")
                    self.canvas.create_text(50, 50, text=predicted_character, font=("Helvetica", 20), fill="red")

            self.root.after(10, self.update)
        else:
            self.root.after(10, self.update)

# Main application
def main():
    root = tk.Tk()
    login_window = tk.Toplevel(root)
    gesture_ui = GestureRecognitionUI(root)
    login_ui = LoginUI(login_window, gesture_ui)
    root.mainloop()

if __name__ == "__main__":
    main()
