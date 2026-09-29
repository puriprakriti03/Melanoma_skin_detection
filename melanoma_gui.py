import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("melanoma_model.h5")

# Constants
IMG_SIZE = 224
class_names = ["Benign Skin Cancer", "Melanoma Skin Cancer"]

# Create main window
root = tk.Tk()
root.title("Melanoma Detection System")
root.geometry("900x650")
root.configure(bg="#f9c74f")

# GUI Image panels
input_panel = tk.Label(root)
filtered_panel = tk.Label(root)
preprocessed_panel = tk.Label(root)
output_mask_panel = tk.Label(root)

# Global variable to store file path
file_path = None

# ----------- Functions ------------

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def convert_grayscale(img):
    return img.convert("L")

def dummy_mask(img):
    return img.convert("1")

def browse_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    img = Image.open(file_path).resize((200, 200))

    # Input Image
    img_tk = ImageTk.PhotoImage(img)
    input_panel.config(image=img_tk)
    input_panel.image = img_tk

    # Filtered Image (same as input for now)
    filtered_panel.config(image=img_tk)
    filtered_panel.image = img_tk

    # Preprocess (Grayscale)
    gray_img = convert_grayscale(img)
    gray_tk = ImageTk.PhotoImage(gray_img)
    preprocessed_panel.config(image=gray_tk)
    preprocessed_panel.image = gray_tk

    # Output Mask (Dummy)
    bw_img = dummy_mask(img)
    bw_tk = ImageTk.PhotoImage(bw_img)
    output_mask_panel.config(image=bw_tk)
    output_mask_panel.image = bw_tk

    result_label.config(text="")

def detect():
    if not file_path:
        messagebox.showerror("Error", "Please browse an image first.")
        return

    img_arr, original_img = preprocess_image(file_path)
    prediction = model.predict(img_arr)[0]
    print("Raw prediction:", prediction) 
    # Softmax - 2 class prediction
    class_index = np.argmax(prediction)
    confidence = round(prediction[class_index] * 100, 2)

    # Display result
    result_text = f"{class_names[class_index]} ({confidence}%)"
    result_label.config(text=result_text, fg="green" if class_index == 0 else "red")

# ----------- GUI Layout ------------

tk.Label(root, text="Input Image", bg="#f9c74f", font=("Arial", 10, "bold")).place(x=100, y=20)
input_panel.place(x=50, y=50)

tk.Label(root, text="Filtered Image", bg="#f9c74f", font=("Arial", 10, "bold")).place(x=330, y=20)
filtered_panel.place(x=280, y=50)

tk.Label(root, text="Preprocess Image", bg="#f9c74f", font=("Arial", 10, "bold")).place(x=100, y=280)
preprocessed_panel.place(x=50, y=310)

tk.Label(root, text="Output Image", bg="#f9c74f", font=("Arial", 10, "bold")).place(x=330, y=280)
output_mask_panel.place(x=280, y=310)

# Right menu
menu_frame = tk.Frame(root, bg="#f8961e")
menu_frame.place(x=600, y=80, width=250, height=250)

tk.Button(menu_frame, text="Browse Image", command=browse_image, height=2, width=20).pack(pady=10)
tk.Button(menu_frame, text="Detect Cancer", command=detect, height=2, width=20).pack(pady=10)
tk.Button(menu_frame, text="Exit", command=root.destroy, height=2, width=20).pack(pady=10)

# Prediction result label
result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="#f9c74f")
result_label.place(x=150, y=550)

# Start GUI loop
root.mainloop()



