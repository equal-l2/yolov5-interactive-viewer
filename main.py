#!/usr/bin/env python3

from PIL import Image, ImageTk
from tkinter import ttk, filedialog
import cv2
import os
import tkinter
import yolov5


class YoloV5InteractiveViewer:
    def __init__(self, root):
        root.geometry("1600x600")
        root.title("YOLOv5 Interactive Viewer")

        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, sticky=tkinter.NSEW)
        # configureしないと伸びない
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # place widgets
        self.left_sidebar = ttk.Frame(mainframe)
        self.left_sidebar.grid(column=0, row=0, sticky=tkinter.NS + tkinter.W)
        self.left_sidebar["borderwidth"] = 1

        self.image_view = tkinter.Canvas(mainframe)
        self.image_view.grid(column=1, row=0, sticky=tkinter.NSEW)
        self.image_view.configure(bg="gray")

        self.right_sidebar = ttk.Frame(mainframe)
        self.right_sidebar.grid(column=2, row=0, sticky=tkinter.NS + tkinter.E)
        self.right_sidebar["borderwidth"] = 1

        # configureしないと伸びない
        mainframe.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=10)
        mainframe.columnconfigure(2, weight=1)
        mainframe.rowconfigure(0, weight=1)

        self.tk_image = None  # need to alive
        self.model = None
        self.pil_image = None
        self.realpathes = []
        self.image_index = 0
        self.file_list = None

        self.configure_left_sidebar()
        self.configure_right_sidebar()

    def configure_left_sidebar(self):
        folder_button = ttk.Button(
            self.left_sidebar, text="Load Folder", command=self.load_folder
        )
        folder_button.pack()

        def update_index(e):
            select = e.widget.curselection()
            self.image_index = select[0]
            self.run_detect()

        self.file_list = tkinter.Listbox(self.left_sidebar, height=10)
        self.file_list.pack()
        self.file_list.bind("<<ListboxSelect>>", update_index)

    def configure_right_sidebar(self):
        fit_button = ttk.Button(
            self.right_sidebar, text="Fit Image", command=self.fit_image
        )
        fit_button.pack()
        model_button = ttk.Button(
            self.right_sidebar, text="Open Model", command=self.open_model
        )
        model_button.pack()
        confidence_frame = ttk.Frame(self.right_sidebar)
        confidence_frame.pack()

        ttk.Label(confidence_frame, text="Confidence:").pack()

        self.confidence_scale = ttk.Scale(confidence_frame, from_=0.0, to=1.0)
        self.confidence_scale.pack()

    def fit_image(self):
        if self.pil_image is not None:
            width = self.image_view.winfo_width()
            height = self.image_view.winfo_height()
            copied = self.pil_image.copy()
            copied.thumbnail((width, height))
            print(width, height)
            self.tk_image = ImageTk.PhotoImage(image=copied)
            self.image_view.delete("all")
            self.image_view.create_image(0, 0, image=self.tk_image, anchor="nw")

    def open_model(self):
        filename = filedialog.askopenfilename()
        print(f"Load {filename}")
        self.model = yolov5.load(filename)
        self.run_detect()

    def load_folder(self):
        folder = filedialog.askdirectory()
        images = []
        self.realpathes = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1]
            valid_image_ext = [".jpg", ".jpeg", ".png"]
            if ext.lower() in valid_image_ext:
                realpath = os.path.join(folder, f)
                self.realpathes.append(realpath)
                images.append(f)

        tk_imagelist = tkinter.StringVar(value=images)
        if self.file_list is not None:
            self.file_list["listvariable"] = tk_imagelist

    def run_detect(self):
        if self.model is not None and self.image_index < len(self.realpathes):
            filename = self.realpathes[self.image_index]
            print(f"Run detect for {filename}")
            cv2_image = cv2.imread(filename)
            cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB, dst=cv2_image)
            detected = self.model(cv2_image, size=1280)
            rendered = detected.render()[0]
            self.pil_image = Image.fromarray(rendered)
            self.fit_image()


print("Initializing...")
root = tkinter.Tk()
view = YoloV5InteractiveViewer(root)
print("Initialized")
root.mainloop()
