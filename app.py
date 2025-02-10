import time
import threading
import cv2
from PIL import Image, ImageTk
from tkinter import Label, Button, Tk, filedialog
from model import ClassificationModel


class CropImageClassificationApp:
    """
    ----------------------------------------------------------------------------------------------
    
    # Crop Image Classification Application 
    
    
    The Crop Image Classification Application is a GUI tool designed for image-based classification tasks, particularly in identifying and classifying crop types or related categories. The application makes use of a pre-trained machine learning model to predict the class of an image and its associated confidence score.

    The tool supports two main functionalities:
    - Capturing images using a webcam.
    - Loading existing images from the local system.

    This application is useful in agricultural or related domains where classification of crops is essential for decision-making or analysis.
    
    ----------------------------------------------------------------------------------------------
    """
    
    def __init__(self, window=Tk(), fn_predict=ClassificationModel().predict):

        # Prediction function for the model
        self.fn_predict = fn_predict
        # Path to save captured image
        self.save_image_path = "image.jpg"
        self.eval_image_path = lambda: self.save_image_path

        self.window = window
        self.window.title("Crop Image Classification")
        self.window.geometry("700x700")
        self.window.resizable(0, 0)

        self.render_capture_layout()

    def render_capture_layout(self):
        """
        Renders the main image capture layout.
        """
        self.window.configure(bg="#0000ff")
        self.take_picture = False
        self.picture_taken = False

        # Remove all widgets from the window before adding new ones
        for widget in self.window.winfo_children():
            widget.place_forget()

        file_chooser_btn = Button(self.window, width=20, text="Select image", font=(
            "Arial", 15), bg="#ff0000", command=self.open_file)
        file_chooser_btn.place(x=10, y=560)
        self.capture_image_button = Button(self.window, width=20, text="Capture", font=(
            "Arial", 15), bg="#ff0000", command=self.capture_picture)
        self.ImageLabel = Label(self.window, bg="white")
        self.ImageLabel.place(x=0, y=0)
        self.capture_image_button.place(x=360, y=560)
        self.app_main()

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a File",
            filetypes=(
                ("JPG Files", "*.jpg"),
                ("JPEG Files", "*.jpeg"),
                ("PNG Files", "*.png"),
            )
        )
        if file_path:
            print(f"Selected File: {file_path}")
            self.eval_image_path = lambda: file_path
            self.picture_taken = True

    def capture_picture(self):
        """
        Capture a picture from the camera and save it to file.
        """
        if not self.picture_taken:
            print('Taking a Picture')
            self.take_picture = True
            print("Saving the Picture")
            cv2.imwrite(self.save_image_path, self.last_frame)
            self.take_picture = False
            self.picture_taken = True
            self.eval_image_path = lambda: self.save_image_path
            print("Picture Saved")

    def change_to_results_layout(self, predicted_class, prediction_confidence):
        """
        Change the UI to display prediction results.
        """
        for widget in self.window.winfo_children():
            widget.place_forget()

        text_color = 'black'
        text_bg = 'green'
        font = ("Monospace", 20)
        self.window.configure(bg="green")

        self.ImageLabel = Label(self.window, bg="white")
        self.ImageLabel.place(x=20, y=300)
        Label(self.window, text="Image Classification Results", font=(
            "Arial", 30, "bold"), bg=text_bg).place(x=50, y=30)
        Label(self.window, text=f"Predicted Class: '{predicted_class.upper()}'", font=font, bg=text_bg).place(
            x=150, y=100)
        Label(self.window, text=f"Prediction Confidence: {prediction_confidence:.2f}%", font=font, bg=text_bg).place(
            x=150, y=150)
        Button(self.window, text="Re-capture",
               bg="#ff0000", command=self.render_capture_layout).place(x=100, y=250)
        Button(self.window, text="Exit",
               bg="#ff0000", command=self.window.quit).place(x=350, y=250)

        frame = cv2.imread(self.eval_image_path())
        picture = Image.fromarray(frame)
        picture = picture.resize((300, 300), resample=0)
        picture = ImageTk.PhotoImage(picture)
        self.ImageLabel.configure(image=picture)
        self.ImageLabel.photo = picture
        self.window.update()

    def app_main(self):
        """
        Start the main camera stream and inference in a separate thread.
        """
        self.render_thread = threading.Thread(target=self.StartCamera)
        self.render_thread.daemon = True
        self.render_thread.start()

    def StartCamera(self):
        """
        Start the camera feed and handle real-time display.
        """
        self.camera = cv2.VideoCapture(0)
        self.last_frame = None
        while True:
            if self.picture_taken:
                self.camera.release()
                print("Predicting the Image")
                predicted_class, prediction_confidence = self.fn_predict(
                    self.eval_image_path())
                print(
                    f"Predicted Class: {predicted_class.upper()}, Prediction Confidence: {prediction_confidence:.2f}")
                self.change_to_results_layout(
                    predicted_class, prediction_confidence)
                break

            ret, frame = self.camera.read()
            if ret and not self.take_picture:
                self.last_frame = frame
                self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                picture = Image.fromarray(frame)
                picture = picture.resize((500, 400), resample=0)
                picture = ImageTk.PhotoImage(picture)
                self.ImageLabel.configure(image=picture)
                self.ImageLabel.photo = picture
                self.window.update()
                time.sleep(0.001)
                
    def run(self):
        """
        Run the main application loop.
        """
        print("Starting the Crop Image Classification App...")
        self.window.mainloop()
                
                
def run_app():
    app = CropImageClassificationApp()
    app.run()

if __name__ == "__main__":
    run_app()