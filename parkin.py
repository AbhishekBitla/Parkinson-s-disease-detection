import os
import cv2
import numpy as np
from skimage import feature
from tkinter import Tk, Label, Button, Canvas, filedialog, messagebox
from PIL import Image, ImageTk
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Function to quantify the image using HOG features
def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

# Function to load and split the dataset
def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)
        
    return np.array(data), np.array(labels)

# Function to train the models
def train_models(dataset_path):
    models = {
        "Rf": {
            "classifier": RandomForestClassifier(random_state=1),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        },
        "Xgb": {
            "classifier": XGBClassifier(),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        }
    }
    
    trainingPath = os.path.join(dataset_path, "training")
    testingPath = os.path.join(dataset_path, "testing")
    
    trainX, trainY = load_split(trainingPath)
    testX, testY = load_split(testingPath)
    
    print(f"trainX shape: {trainX.shape}")  # Check the shape of trainX
    print(f"trainY shape: {trainY.shape}")  # Check the shape of trainY
    
    if trainX.shape[0] == 0 or trainY.shape[0] == 0:
        raise ValueError("No data loaded. Check your dataset path and content.")
    
    # Encode labels
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)

    # Train each model and calculate its metrics
    for model in models:
        print(f"Training {model}...")
        models[model]["classifier"].fit(trainX, trainY)
        predictions = models[model]["classifier"].predict(testX)
        cm = confusion_matrix(testY, predictions).ravel()
        tn, fp, fn, tp = cm
        models[model]["accuracy"] = (tp + tn) / float(cm.sum())
        models[model]["sensitivity"] = tp / float(tp + fn)
        models[model]["specificity"] = tn / float(tn + fp)

    return models

# Function to predict on a single image
def predict_single_image(model, image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    thresh = cv2.threshold(resized, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(thresh)
    prediction = model.predict([features])[0]
    label = "Parkinson's" if prediction == 1 else "Healthy"
    return label, image

# GUI Class for the application
class ParkinsonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detection")
        
        self.label = Label(self.root, text="Browse an image to predict:")
        self.label.pack(pady=20)
        
        self.canvas = Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        self.browse_button = Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=20)

        self.predict_button = Button(self.root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                     filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
        if self.image_path:
            self.load_image()

    def load_image(self):
        self.img = Image.open(self.image_path)
        self.img = self.img.resize((400, 400), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def predict_image(self):
        if hasattr(self, 'image_path'):
            try:
                model = self.train_model()
                label, output_image = predict_single_image(model, self.image_path)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                output_image = Image.fromarray(output_image)
                output_image = output_image.resize((400, 400), Image.LANCZOS)
                output_image = ImageTk.PhotoImage(output_image)
                self.canvas.create_image(0, 0, anchor="nw", image=output_image)
                prediction_label = f"Prediction: {label}"
                messagebox.showinfo("Prediction", prediction_label)
            except Exception as e:
                messagebox.showerror("Error", f"Prediction error: {str(e)}")
        else:
            messagebox.showerror("Error", "Please browse an image first.")

    def train_model(self):
        dataset_path = "C:/vs_code/virtual enve/drawings"
        models = train_models(dataset_path)
        return models['Rf']['classifier']  # Returning RandomForestClassifier for prediction

# Main function to run the application
if __name__ == "__main__":
    root = Tk()
    app = ParkinsonApp(root)
    root.mainloop()