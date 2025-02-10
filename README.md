# **Crop Image Classification Application**  

The Crop Image Classification Application is a GUI tool designed for image-based classification tasks, particularly in identifying and classifying crop types or related categories. The application makes use of a pre-trained machine learning model to predict the class of an image and its associated confidence score.

The tool supports two main functionalities:
- Capturing images using a webcam.
- Loading existing images from the local system.

This application is useful in agricultural or related domains where classification of crops is essential for decision-making or analysis.


## **Index**
- [Demo](#demo)  
- [Installation](#installation)  
- [How to Use](#how-to-use)  
- [Technologies Used](#technologies-used)  

<!-- ## Demo   -->


## Installation

### Prerequisites
1. Python 3.8 or later (Not Python 13, it is not suported by Tensorflow).  
2. A webcam or plant images.

### Steps
1. **Install packages**:  
   Run the following command to install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model**:
   Downalod the model and save to `saved_models/` directory.
    - [Download Link](https://drive.google.com/file/d/1-x4UlKuHSOj5KjrRdSglWi9T4cmmrj5m/view?usp=drive_link)

4. **Run**:
   ```bash
   python app.py
   ```

## Technologies Used
- Python, Tkinter, TensorFlow/Keras, OpenCV, PIL (Pillow)