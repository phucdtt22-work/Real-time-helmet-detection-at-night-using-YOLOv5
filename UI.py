from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter as tk
import threading
import time
from ultralytics import YOLO
import numpy as np
import os
from tkinter import messagebox
from increase_light import gamma_correction, enhance_low_light_image_new_algorithm, histogram_equalization
import torch
r = Tk()
r.geometry('1300x750')
r.title("Helmet Detector")

video = StringVar()
captured_helmets = []
count = 0
running = False
fps = 0
frame_count = 0
prev_time = time.time()
previous_boxes = None

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO('best_final.pt')

with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

def video_option():
    global cap
    video_name = video.get()
    video_path = f"{video_name}.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        label_success_video = Label(box2, text=f"Video path updated to: {video_path}", font="arial 10", fg="red")
        label_success_video.place(x=10, y=170)
    else:
        print(f"Video {video_name} not found in the directory.")
        messagebox.showerror("Video Not Found", f"Video {video_name} not found in the directory.")

def start():
    global running
    if not running:
        running = True
        threading.Thread(target=update_frame, daemon=True).start()

def end():
    global running, cap
    running = False
    if cap.isOpened():
        cap.release()

def open_dis_image():
    threading.Thread(target=dis_image, daemon=True).start()

def is_similar_box(new_box, existing_boxes, threshold=50):
    for box in existing_boxes:
        if all(abs(new_box[i] - box[i]) < threshold for i in range(4)):
            return True
    return False

def dis_image():
    r_image = Toplevel(r)
    r_image.geometry('800x600')
    r_image.title("Image No Helmet")

    def display_images():
        for widget in r_image.winfo_children():
            widget.destroy()
        img_no_helmet_path = "img_no_helmet"
        row_count = 0
        col_count = 0
        for filename in os.listdir(img_no_helmet_path):
            file_path = os.path.join(img_no_helmet_path, filename)
            img = Image.open(file_path)
            img = img.resize((int(img.width * 0.65), int(img.height * 0.65)))
            img_tk = ImageTk.PhotoImage(img)

            canvas = Canvas(r_image, width=img.width, height=img.height + 20)
            canvas.grid(row=row_count, column=col_count, padx=10, pady=10)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk
            canvas.create_text(img.width / 2, img.height + 10, text=filename, fill="black", font=("Arial", 7),
                               anchor="center")

            col_count += 1
            if col_count == 6:
                col_count = 0
                row_count += 1

        r_image.after(5000, display_images)

    display_images()

def calculate_fps():
    global frame_count, prev_time
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    prev_time = current_time
    frame_count = 0
    return fps

def are_boxes_similar(boxes11, boxes22, threshold=200):
    if boxes11 is None or boxes22 is None:
        return False
    if len(boxes11) != len(boxes22):
        return False
    for box11, box22 in zip(boxes11, boxes22):
        if not all(abs(box11[i] - box22[i]) < threshold for i in range(4)):
            return False
    return True
previous_boxes1 = []  
previous_boxes2 = [] 

def update_frame():
    global captured_helmets, count, fps, frame_count, prev_time, previous_boxes
    global previous_boxes1, previous_boxes2
    skip_frames = 0
    while running:
        if skip_frames > 0:
            for _ in range(3):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            skip_frames = 0
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized1 = cv2.resize(frame_rgb, (box1.winfo_width(), box1.winfo_height()))
        frame_resized2 = cv2.resize(frame_rgb, (box3.winfo_width(), box3.winfo_height()))

        if frame_count % 20 == 0:
            algorithm1 = selected_algorithm1.get()
            if algorithm1 == "Gamma Correction":
                frame_rgb1 = gamma_correction(frame_resized1, sc_gamma1.get())
            elif algorithm1 == "Histogram Equalization":
                frame_rgb1 = histogram_equalization(frame_resized1)
            elif algorithm1 == "Enhance Low Light":
                frame_rgb1 = enhance_low_light_image_new_algorithm(frame_resized1, gamma=sc_gamma1.get(), omega=sc_omega1.get())
            else:
                frame_rgb1 = frame_resized1

            results1 = model.predict(frame_rgb1)
            boxes1 = np.array(results1[0].cpu().boxes.data)


            algorithm2 = selected_algorithm2.get()
            if algorithm2 == "Gamma Correction":
                frame_rgb2 = gamma_correction(frame_resized2, sc_gamma2.get())
            elif algorithm2 == "Histogram Equalization":
                frame_rgb2 = histogram_equalization(frame_resized2)
            elif algorithm2 == "Enhance Low Light":
                frame_rgb2 = enhance_low_light_image_new_algorithm(frame_resized2, gamma=sc_gamma2.get(), omega=sc_omega2.get())
            else:
                frame_rgb2 = frame_resized2

            results2 = model.predict(frame_rgb2)
            boxes2 = np.array(results2[0].cpu().boxes.data)

            if (are_boxes_similar(boxes1, previous_boxes1) or len(boxes1) == 0) and \
               (are_boxes_similar(boxes2, previous_boxes2) or len(boxes2) == 0):
                skip_frames = 3
                continue

            previous_boxes1 = boxes1
            previous_boxes2 = boxes2

            directory = 'img_no_helmet'

            for box in boxes1:
                x1, y1, x2, y2, _, d = box.astype(int)
                c = class_list[d]

                x1_scaled1 = int(x1 * frame_resized1.shape[1] / frame_rgb1.shape[1])
                y1_scaled1 = int(y1 * frame_resized1.shape[0] / frame_rgb1.shape[0])
                x2_scaled1 = int(x2 * frame_resized1.shape[1] / frame_rgb1.shape[1])
                y2_scaled1 = int(y2 * frame_resized1.shape[0] / frame_rgb1.shape[0])

                if d == 1:
                    if not is_similar_box((x1, y1, x2, y2), captured_helmets):
                        helmet_img = frame_rgb1[y1:y2, x1:x2]
                        resize = cv2.resize(helmet_img, (128, 128))
                        resize_rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                        file_path = f"{directory}/pic_{count}.jpg"
                        cv2.imwrite(file_path, resize_rgb)
                        captured_helmets.append((x1, y1, x2, y2))
                        count += 1
                    cv2.rectangle(frame_rgb1, (x1_scaled1, y1_scaled1), (x2_scaled1, y2_scaled1), (0, 0, 255), 1)
                    cv2.putText(frame_rgb1, f'{c}', (x1_scaled1, y1_scaled1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                elif d == 0:
                    cv2.rectangle(frame_rgb1, (x1_scaled1, y1_scaled1), (x2_scaled1, y2_scaled1), (255, 0, 0), 1)
                    cv2.putText(frame_rgb1, f'{c}', (x1_scaled1, y1_scaled1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            for box in boxes2:
                x1, y1, x2, y2, _, d = box.astype(int)
                c = class_list[d]

                x1_scaled2 = int(x1 * frame_resized2.shape[1] / frame_rgb2.shape[1])
                y1_scaled2 = int(y1 * frame_resized2.shape[0] / frame_rgb2.shape[0])
                x2_scaled2 = int(x2 * frame_resized2.shape[1] / frame_rgb2.shape[1])
                y2_scaled2 = int(y2 * frame_resized2.shape[0] / frame_rgb2.shape[0])

                if d == 0:
                    cv2.rectangle(frame_rgb2, (x1_scaled2, y1_scaled2), (x2_scaled2, y2_scaled2), (255, 0, 0), 1)
                    cv2.putText(frame_rgb2, f'{c}', (x1_scaled2, y1_scaled2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                elif d == 1:
                    cv2.rectangle(frame_rgb2, (x1_scaled2, y1_scaled2), (x2_scaled2, y2_scaled2), (0, 0, 255), 1)
                    cv2.putText(frame_rgb2, f'{c}', (x1_scaled2, y1_scaled2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        

        frame_count += 1
        fps = calculate_fps()
        fps_label.config(text=f"FPS: {fps:.2f}")
        print(f"Frame count: {frame_count}, FPS: {fps:.2f}")    
        update_ui(frame_rgb1, frame_rgb2)

def update_ui(frame_rgb1, frame_rgb2):
    image1 = Image.fromarray(frame_rgb1)
    image2 = Image.fromarray(frame_rgb2)
    bg1 = ImageTk.PhotoImage(image1)
    bg2 = ImageTk.PhotoImage(image2)
    canvas.create_image(0, 0, anchor=tk.NW, image=bg1)
    canvas.image = bg1
    canvas1.create_image(0, 0, anchor=tk.NW, image=bg2)
    canvas1.image = bg2

box1 = LabelFrame(r, text="Video Capture")
box2 = LabelFrame(r, text="Option")
box3 = LabelFrame(r, text="Video Capture 2")
box4 = LabelFrame(r, text="Option")

image = Image.open("helmet.png")
image = image.resize((int(image.width * 1.25), int(image.height * 1.25)))
bg = ImageTk.PhotoImage(image)

canvas = Canvas(box1, width=image.width, height=image.height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor=NW, image=bg)

canvas1 = Canvas(box3, width=image.width, height=image.height)
canvas1.pack(fill="both", expand=True)
canvas1.create_image(0, 0, anchor=NW, image=bg)

box1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
box2.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
box3.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
box4.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

lb_video = Label(box2, text="Video Options: ", font="arial 10")
lb_video.grid(row=3, column=0, padx=10, pady=10)
entry_video = Entry(box2, textvariable=video)
entry_video.grid(row=3, column=1, padx=10, pady=10)
bttn_update_video = Button(box2, text="Update Video", font="arial 10", width=10, command=video_option)
bttn_update_video.grid(row=4, column=0, padx=10, pady=10)

algorithm_options = ["Not processed yet", "Gamma Correction", "Histogram Equalization", "Enhance Low Light"]
selected_algorithm1 = StringVar(value=algorithm_options[0])
selected_algorithm2 = StringVar(value=algorithm_options[0])

lb1 = Label(box2, text="Algorithm :")
lb1.grid(row=0, column=2, padx=5, pady=5)

algorithm_menu1 = OptionMenu(box2, selected_algorithm1, *algorithm_options)
algorithm_menu1.grid(row=0, column=3, padx=5, pady=3)

lb2 = Label(box4, text="Algorithm :")
lb2.grid(row=0, column=3, padx=5, pady=5)

algorithm_menu2 = OptionMenu(box4, selected_algorithm2, *algorithm_options)
algorithm_menu2.grid(row=0, column=4, padx=5, pady=3)

#Scale_box 2
lb_gamma1 = Label(box2, text="gamma: ", font="arial 10")
lb_gamma1.place(x=295 , y=135)
sc_gamma1 = Scale(box2,from_= 0, to = 1,resolution=0.1, length= 130, orient="horizontal")
sc_gamma1.place(x=350 , y=120)

lb_omega1 = Label(box2, text="omega: ", font="arial 10")
lb_omega1.place(x=295 , y=175)
sc_omega1 = Scale(box2, from_= 0, to = 1,resolution=0.1, length= 130, orient="horizontal")
sc_omega1.place(x=350 , y=160)

#Scale_box 2
lb_gamma2 = Label(box4, text="gamma: ", font="arial 10")
lb_gamma2.place(x=10 , y=135)
sc_gamma2 = Scale(box4,from_= 0, to = 1,resolution=0.1, length= 130, orient="horizontal")
sc_gamma2.place(x=65 , y=120)

lb_omega2 = Label(box4, text="omega: ", font="arial 10")
lb_omega2.place(x=10 , y=175)
sc_omega2 = Scale(box4, from_= 0, to = 1,resolution=0.1, length= 130, orient="horizontal")
sc_omega2.place(x=65 , y=160)



bttn_start = Button(box2, text="Start", font="arial 10", width=10, command=start)
bttn_start.grid(row=0, column=0, padx=10, pady=10)

bttn_stop = Button(box2, text="Stop", font="arial 10", width=10, command=end)
bttn_stop.grid(row=1, column=0, padx=10, pady=10)

bttn_dis_image = Button(box2, text="Display Image", font="arial 10", width=10, command=open_dis_image)
bttn_dis_image.grid(row=4, column=1, padx=10, pady=10)

fps_label = Label(box2, text="FPS: 0.00", font="arial 10")
fps_label.grid(row=2, column=0, padx=10, pady=10)

r.mainloop()
