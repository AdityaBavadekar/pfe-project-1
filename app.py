import time
import threading
import cv2
from PIL import Image, ImageTk 
from tkinter import Label, Button, Tk, PhotoImage
        

class CropImageClassificationApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Crop Image Detection cum Classification")
        self.window.geometry("500x400")
        self.window.configure(bg="#0000ff")
        self.window.resizable(1, 1)
        Label(self.window, width=400, height = 30,  bg="black").place(x=0, y =320)
        self.capture_image_button = Button(self.window, width = 20, text = "Capture", font = ("Arial", 15),bg = "#ff0000", command=self.capture_picture)
        self.ImageLabel = Label(self.window, width = 500, height= 320, bg = "#4682B4")
        self.ImageLabel.place(x=0, y=0)
        self.capture_image_button.place(x = 150, y = 360)
        
        self.take_picture = False
        self.picture_taken = False
        self.app_main()

    def capture_picture(self):
        if not self.picture_taken:
            print('Taking a Picture')
            self.take_picture = True
            print("Saving the Picture")
            cv2.imwrite("image.jpg", self.last_frame)
            self.take_picture = False
            self.picture_taken = True
            exit(-1)

 
    def app_main(self):
        self.render_thread = threading.Thread(target=self.StartCamera)
        self.render_thread.daemon = True
        self.render_thread.start()


    def StartCamera(self):
        self.camera = cv2.VideoCapture(0)
        self.last_frame = None
        while True:
            ret, frame = self.camera.read()
            if ret and not self.take_picture:
                self.last_frame = frame
                picture = Image.fromarray(frame)
                picture = picture.resize((500, 400), resample=0)
                picture = ImageTk.PhotoImage(picture)
                self.ImageLabel.configure(image = picture)
                self.ImageLabel.photo = picture
                self.window.update()
                time.sleep(0.001)
            else:
                print("Your camera stopped")
                break
            

root = Tk()
App = CropImageClassificationApp(root)
root.mainloop()