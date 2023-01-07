#!/usr/bin/env python3

from PIL import Image, ImageTk, ImageColor
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import os
import tkinter
import yolov5


class LineConfigs(ttk.LabelFrame):
    def __init__(self, root, text: str, color: str, width: int):
        ttk.LabelFrame.__init__(self, root, text=text)

        self.width_entry = IntEntry(self, label="Line width", init=width)
        self.width_entry.pack()

        color_frame = ttk.Frame(self)
        color_frame.pack()
        self.color = color
        ttk.Label(color_frame, text="Line color").pack(side=tkinter.LEFT)
        ttk.Button(color_frame, text="Choose...", command=self.choose_line_color).pack(
            side=tkinter.LEFT
        )
        self.color_label = tkinter.Label(color_frame, text="     ", bg=self.color)
        self.color_label.pack(side=tkinter.LEFT)

    def choose_line_color(self):
        colors = colorchooser.askcolor()
        if colors is not None and colors[1] is not None:
            self.color = colors[1]
            self.color_label["bg"] = self.color

    def get(self):
        width = self.width_entry.get()
        if width is None:
            return None
        else:
            return (ImageColor.getrgb(self.color), width)


class IntEntry(ttk.Frame):
    def __init__(self, root, label: str, init: int):
        ttk.Frame.__init__(self, root)

        self.value = tkinter.StringVar(value=str(init))
        ttk.Label(self, text=label).pack(side=tkinter.LEFT)
        ttk.Entry(
            self,
            textvariable=self.value,
        ).pack(side=tkinter.LEFT)

    def get(self):
        s = self.value.get()
        if not s.isdigit():
            return None

        try:
            val = int(s)
            if str(val) == s and val > 0:
                return val
            else:
                return None
        except Exception as e:
            print(e)
            return None

    def set(self, val: int):
        self.value.set(str(val))


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
        mainframe.columnconfigure(1, weight=10)
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
        ttk.Button(
            self.left_sidebar, text="Load folder", command=self.load_folder
        ).pack()

        def update_index(e):
            select = e.widget.curselection()
            if len(select) < 1:
                return
            self.image_index = select[0]
            self.run_detect()

        self.file_list = tkinter.Listbox(self.left_sidebar, height=10)
        self.file_list.pack()
        self.file_list.bind("<<ListboxSelect>>", update_index)

        ttk.Button(self.left_sidebar, text="Save picture", command=self.save).pack()

    def save(self):
        if self.image_index >= len(self.realpathes) or self.pil_image is None:
            return

        original_basename = os.path.basename(self.realpathes[self.image_index])
        original_name, ext = os.path.splitext(original_basename)
        new_name = original_name + "_detected" + ext

        real_new_name = filedialog.asksaveasfilename(initialfile=new_name)

        if real_new_name == "":
            # canceled
            return

        self.pil_image.save(real_new_name)

    def configure_right_sidebar(self):
        parent = self.right_sidebar
        ttk.Button(parent, text="Fit image", command=self.fit_image).pack()

        model_frame = ttk.Frame(parent)
        model_frame.pack()

        ttk.Button(model_frame, text="Open model", command=self.load_model).pack(
            side=tkinter.LEFT
        )
        self.model_name = ttk.Label(model_frame)
        self.model_name.pack(side=tkinter.LEFT)

        self.confidence = tkinter.DoubleVar()
        scale = tkinter.Scale(
            parent,
            from_=0,
            to=1.0,
            resolution=0.01,
            label="Confidence",
            orient=tkinter.HORIZONTAL,
            variable=self.confidence,
        )
        scale.set(0.25)  # the default value from YOLOv5 detect.py
        scale.pack()

        self.bb_config = LineConfigs(
            parent, text="Bounding Boxes", color="#FF0000", width=2
        )
        self.bounds_config = LineConfigs(
            parent, text="Upper/Lower Bounds", color="#00FF00", width=1
        )
        self.bb_config.pack()
        self.bounds_config.pack()

        self.upper_pixel = IntEntry(
            self.bounds_config, label="Upper Bound Px", init=287
        )
        self.lower_pixel = IntEntry(
            self.bounds_config, label="Lower Bound Px", init=1000
        )
        self.upper_pixel.pack()
        self.lower_pixel.pack()

        ttk.Button(parent, text="Run Detection", command=self.run_detect).pack()

        # "Advanced" settings
        self.use_yolo_render = tkinter.BooleanVar()
        self.use_yolo_render.trace_add("write", self.handle_own_renderer_checkbutton)
        ttk.Checkbutton(
            parent, text="Use YOLOv5-pip renderer", variable=self.use_yolo_render
        ).pack()

        self.show_label = tkinter.BooleanVar()
        ttk.Checkbutton(parent, text="Show label", variable=self.show_label).pack()

    def handle_own_renderer_checkbutton(self, *_):
        # enable / disable all control associated with the own renderer
        pass

    def fit_image(self):
        if self.pil_image is not None:
            width = self.image_view.winfo_width()
            height = self.image_view.winfo_height()
            copied = self.pil_image.copy()
            copied.thumbnail((width, height))
            # print(width, height)
            self.tk_image = ImageTk.PhotoImage(image=copied)
            self.image_view.delete("all")
            self.image_view.create_image(0, 0, image=self.tk_image, anchor="nw")

    def load_model(self):
        filename = filedialog.askopenfilename(title="Choose Model")
        if filename == "":
            # canceled
            return

        print(f"Load {filename}")
        try:
            self.model = yolov5.load(filename)
        except Exception as e:
            messagebox.showerror(message=f"Failed to load the model: {e}")
            return

        self.model_name["text"] = os.path.basename(filename)
        self.run_detect()

    def load_folder(self):
        if self.model is None:
            messagebox.showerror(message="You need to load a model first")
            return

        folder = filedialog.askdirectory(title="Choose Image Folder")
        if folder == "":
            # canceled
            return

        # get all images in the folder
        images = []
        self.realpathes = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1]
            valid_image_ext = [".jpg", ".jpeg", ".png"]
            if ext.lower() in valid_image_ext:
                realpath = os.path.join(folder, f)
                self.realpathes.append(realpath)
                images.append(f)

        # populate file list
        tk_imagelist = tkinter.StringVar(value=images)
        if self.file_list is not None:
            self.file_list["listvariable"] = tk_imagelist

    def ensure_detect_params(self):
        if self.model is None:
            messagebox.showerror(message="Model is not loaded")
            return None

        if self.image_index >= len(self.realpathes):
            # logic error (not because of users)
            # usually occurs when no image is loaded
            return None

        bb_params = self.bb_config.get()
        if bb_params is None:
            messagebox.showerror(
                message="Bounding Boxes: Line width must be a positive interger"
            )
            return None

        bounds_params = self.bounds_config.get()
        if bounds_params is None:
            messagebox.showerror(
                message="Upper/Lower Bounds: Line width must be a positive interger"
            )
            return None

        confidence = self.confidence.get()
        if confidence == 0:
            messagebox.showerror(
                message="Confidence == 0 will result in castrophe, stopping"
            )
            return None

        upper_pixel = self.upper_pixel.get()
        if upper_pixel is None:
            messagebox.showerror(
                message="Upper Bounds: Pixel must be a positive interger"
            )
            return None

        lower_pixel = self.lower_pixel.get()
        if lower_pixel is None:
            messagebox.showerror(
                message="Lower Bounds: Pixel must be a positive interger"
            )
            return None

        filename = self.realpathes[self.image_index]
        return (
            self.model,
            filename,
            confidence,
            bb_params,
            bounds_params,
            lower_pixel,
            upper_pixel,
        )

    def run_detect(self):
        params = self.ensure_detect_params()
        if params is None:
            return
        (
            model,
            filename,
            confidence,
            bb_params,
            bounds_params,
            lower_pixel,
            upper_pixel,
        ) = params

        show_label = self.show_label.get()
        print(
            f"Run detect for {filename} with confidence {confidence} and label is {show_label}"
        )

        cv2_image = cv2.imread(filename)
        cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB, dst=cv2_image)

        detected = model(cv2_image, size=1280)

        if self.use_yolo_render.get():
            rendered = detected.render(labels=show_label)[0]
            self.pil_image = Image.fromarray(rendered)
        else:
            values = detected.pandas().xyxy[0]

            # coords in dataframe are float, so they need to be cast into int
            # (because cv2 accepts int coords only)
            values.round(0)
            values = values.astype({"xmin": int, "ymin": int, "xmax": int, "ymax": int})

            # draw bounding boxes
            (bb_color, bb_width) = bb_params
            for row in values.itertuples():
                cv2.rectangle(
                    cv2_image,
                    (row.xmin, row.ymin),
                    (row.xmax, row.ymax),
                    bb_color,
                    bb_width,
                )

            # draw bounds line
            (bounds_color, bounds_width) = bounds_params
            cv2.line(
                cv2_image,
                (0, upper_pixel),
                (cv2_image.shape[1], upper_pixel),
                bounds_color,
                bounds_width,
            )

            cv2.line(
                cv2_image,
                (0, lower_pixel),
                (cv2_image.shape[1], lower_pixel),
                bounds_color,
                bounds_width,
            )

            self.pil_image = Image.fromarray(cv2_image)

        self.fit_image()


print("Initializing...")
root = tkinter.Tk()
view = YoloV5InteractiveViewer(root)
print("Initialized")
root.mainloop()
