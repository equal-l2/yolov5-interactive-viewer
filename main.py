#!/usr/bin/env python3

from PIL import Image, ImageTk, ImageColor
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import os
import tkinter
import yolov5
import traceback
import typing
import numpy

# YOLOv5 parameters, from the default value in detect.py
CONFIDENCE_DEFAULT = 0.25
IOU_DEFAULT = 0.45

# our parameters
OUTSIDE_THRES_DEFAULT = 0.7
MASK_THRES_DEFAULT = OUTSIDE_THRES_DEFAULT
UPPER_BOUND_DEFAULT = 287
LOWER_BOUND_DEFAULT = 850

BOUNDS_COLOR_DEFAULT = "#00FF00"  # green
BBOXES_COLOR_DEFAULT = "#FF0000"  # red
OUTSIDER_COLOR_DEFAULT = "#9900FF"  # purple

# TODO: make configurable
TEXT_COLOR = (255, 0, 0)  # red


class LineConfigs(ttk.LabelFrame):
    def __init__(self, root: tkinter.Misc, text: str, color: str, width: int):
        ttk.LabelFrame.__init__(self, root, text=text)

        self.width_entry = IntEntry(self, label="Line width", init=width)
        self.width_entry.pack()

        self.colorpicker = ColorPicker(self, text="Line color", color=color)
        self.colorpicker.pack()

    def get(self):
        width = self.width_entry.get()
        if width is None:
            return None
        else:
            return (self.colorpicker.get(), width)


class IntEntry(ttk.Frame):
    def __init__(self, root: tkinter.Misc, label: str, init: int):
        ttk.Frame.__init__(self, root)

        self.value = tkinter.StringVar(value=str(init))
        ttk.Label(self, text=label).pack(side=tkinter.LEFT)
        ttk.Entry(
            self,
            textvariable=self.value,
        ).pack(side=tkinter.LEFT)

    def strip(self):
        s = self.value.get()
        self.value.set(s.strip())

    def get(self):
        self.strip()
        s = self.value.get()

        if not s.isdigit():
            return None

        try:
            val = int(s)
            if str(val) == s and val > 0:
                return val
            else:
                return None
        except:
            traceback.print_exc()
            return None

    def set(self, val: int):
        self.value.set(str(val))


class ColorPicker(ttk.Frame):
    def __init__(self, root: tkinter.Misc, text: str, color: str):
        ttk.Frame.__init__(self, root)

        ImageColor.getrgb(color)  # test the color is valid

        self.color = color
        ttk.Label(self, text=text).pack(side=tkinter.LEFT)
        ttk.Button(self, text="Choose...", command=self.choose_line_color).pack(
            side=tkinter.LEFT
        )
        self.color_label = tkinter.Label(self, text="     ", bg=self.color)
        self.color_label.pack(side=tkinter.LEFT)

    def choose_line_color(self):
        colors = colorchooser.askcolor()
        if colors is not None and colors[1] is not None:
            self.color = colors[1]
            self.color_label["bg"] = self.color

    def get(self):
        return ImageColor.getrgb(self.color)


class ZeroToOneScale(tkinter.Scale):
    def __init__(self, root: tkinter.Misc, label: str, init: float):
        tkinter.Scale.__init__(
            self,
            root,
            from_=0,
            to=1.0,
            resolution=0.01,
            label=label,
            orient=tkinter.HORIZONTAL,
        )
        self.set(init)


class YoloV5InteractiveViewer:
    def __init__(self, root: tkinter.Misc):
        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, sticky=tkinter.NSEW)

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

        self.tk_image = None  # need to be alive
        self.model = None
        self.pil_image = None
        self.realpathes: list[str] = []
        self.image_index: int = 0
        self.file_list = None
        self.mask = None

        self.configure_left_sidebar()
        self.configure_right_sidebar()

    def configure_left_sidebar(self):
        ttk.Button(
            self.left_sidebar, text="Load folder", command=self.load_folder
        ).pack()

        def update_index(e: typing.Any):
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

        try:
            self.pil_image.save(real_new_name)
            messagebox.showinfo(message=f"Successfully saved to {real_new_name}")
        except:
            traceback.print_exc()
            messagebox.showinfo(message=f"Failed to save to {real_new_name}")

    def configure_right_sidebar(self):
        parent = self.right_sidebar
        ttk.Button(parent, text="Fit image", command=self.fit_image).pack()

        model_config = ttk.LabelFrame(parent, text="Model")
        model_config.pack()

        model_frame = ttk.Frame(model_config)
        ttk.Button(model_frame, text="Load model", command=self.load_model).pack(
            side=tkinter.LEFT
        )
        self.model_name = ttk.Label(model_frame)
        self.model_name.pack(side=tkinter.LEFT)
        model_frame.pack()

        self.confidence = ZeroToOneScale(
            model_config, label="Confidence", init=CONFIDENCE_DEFAULT
        )
        self.confidence.pack()

        self.iou = ZeroToOneScale(model_config, label="IoU", init=IOU_DEFAULT)
        self.iou.pack()

        self.bb_config = LineConfigs(
            parent, text="Bounding Boxes", color=BBOXES_COLOR_DEFAULT, width=2
        )
        self.bb_config.pack()

        # mark outside of bounds
        self.outsider_config = LineConfigs(
            parent, text="Outsiders", color=OUTSIDER_COLOR_DEFAULT, width=2
        )
        self.outsider_thres = ZeroToOneScale(
            self.outsider_config, label="Min overlap %", init=OUTSIDE_THRES_DEFAULT
        )
        self.outsider_thres.pack()
        self.outsider_config.pack()

        # mask settings
        mask_config = ttk.LabelFrame(parent, text="Mask")

        self.enable_mask = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_config, text="Apply mask as postprocess", variable=self.enable_mask
        ).pack()

        load_mask_frame = ttk.Frame(mask_config)
        ttk.Button(load_mask_frame, text="Load mask", command=self.load_mask).pack(
            side=tkinter.LEFT
        )
        self.mask_name = ttk.Label(load_mask_frame)
        self.mask_name.pack(side=tkinter.LEFT)
        load_mask_frame.pack()

        self.mask_thres = ZeroToOneScale(
            mask_config, label="Min overlap %", init=MASK_THRES_DEFAULT
        )
        self.mask_thres.pack()

        mask_config.pack()

        self.bounds_config = LineConfigs(
            parent, text="Upper/Lower Bounds", color=BOUNDS_COLOR_DEFAULT, width=1
        )
        self.bounds_config.pack()

        self.upper_pixel = IntEntry(
            self.bounds_config, label="Up Px", init=UPPER_BOUND_DEFAULT
        )
        self.lower_pixel = IntEntry(
            self.bounds_config, label="Lo Px", init=LOWER_BOUND_DEFAULT
        )
        self.upper_pixel.pack()
        self.lower_pixel.pack()

        misc_frame = ttk.LabelFrame(parent, text="Misc")
        misc_frame.pack()

        self.hide_outsiders = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame, text="Hide outsiders", variable=self.hide_outsiders
        ).pack()

        self.show_confidence = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame,
            text="Show confidence with bounding boxes",
            variable=self.show_confidence,
        ).pack()

        self.show_filename = tkinter.BooleanVar(value=True)
        ttk.Checkbutton(
            misc_frame, text="Show filename in the picture", variable=self.show_filename
        ).pack()

        self.disable_bounds = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame, text="Disable upper/lower bounds", variable=self.disable_bounds
        ).pack()

        ttk.Button(parent, text="Run Detection", command=self.run_detect).pack()

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

        print(f"Load model {filename}")
        try:
            self.model = yolov5.load(filename)
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load the model")
            return

        self.model_name["text"] = os.path.basename(filename)

    def load_mask(self):
        filename = filedialog.askopenfilename(title="Choose Mask")
        if filename == "":
            # canceled
            return

        print(f"Load mask {filename}")
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load the mask")
            return

        self.mask = mask
        self.mask_name["text"] = os.path.basename(filename)

    def load_folder(self):
        if self.model is None:
            messagebox.showerror(message="You need to load a model first")
            return

        folder = filedialog.askdirectory(title="Choose Image Folder")
        if folder == "":
            # canceled
            return

        # get all images in the folder
        images: list[tuple[str, str]] = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1]
            valid_image_ext = [".jpg", ".jpeg", ".png"]
            if ext.lower() in valid_image_ext:
                realpath = os.path.join(folder, f)
                images.append((f, realpath))

        # sort files by name
        images.sort(key=lambda y: y[0])

        image_names = [name for (name, _) in images]
        self.realpathes = [realpath for (_, realpath) in images]

        # populate file list
        tk_imagelist = tkinter.StringVar(value=image_names)  # type: ignore (mismatch between value and image_names)
        if self.file_list is not None:
            self.file_list["listvariable"] = tk_imagelist

    def get_realpath(self):
        if self.image_index >= len(self.realpathes):
            # logic error (not because of users)
            # usually occurs when no image is loaded
            return None
        return self.realpathes[self.image_index]

    def ensure_detect_params(self):
        bb_params = self.bb_config.get()
        if bb_params is None:
            messagebox.showerror(
                message="Bounding Boxes: Line width must be a positive interger"
            )
            return None

        outsider_params = self.outsider_config.get()
        if outsider_params is None:
            messagebox.showerror(
                message="Outsiders: Line width must be a positive interger"
            )
            return None

        bounds_params = self.bounds_config.get()
        if bounds_params is None:
            messagebox.showerror(
                message="Upper/Lower Bounds: Line width must be a positive interger"
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

        return (
            bb_params,
            outsider_params,
            bounds_params,
            lower_pixel,
            upper_pixel,
        )

    def run_detect(self):
        if self.model is None:
            messagebox.showerror(message="Model is not loaded")
            return

        filename = self.get_realpath()
        if filename is None:
            print("filename is None")
            return

        self.cv2_image = cv2.imread(filename)
        cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB, dst=self.cv2_image)

        self.model.conf = self.confidence.get()  # type: ignore (mismatch between float and Tensor | Module)
        self.model.iou = self.iou.get()
        detected = self.model(self.cv2_image, size=1280)

        values = detected.pandas().xyxy[0]

        # coords in dataframe are float, so they need to be cast into int
        # (because cv2 accepts int coords only)
        values.round(0)
        values = values.astype({"xmin": int, "ymin": int, "xmax": int, "ymax": int})

        self.values = values
        self.render_result()

    def render_result(self):
        if self.values is None:
            return

        params = self.ensure_detect_params()
        if params is None:
            return
        (
            bb_params,
            outsider_params,
            bounds_params,
            lower_pixel,
            upper_pixel,
        ) = params

        # check mask
        enable_mask = self.enable_mask.get()
        if enable_mask:
            if self.mask is None:
                messagebox.showerror(message="Mask is not loaded")
                return

            mask_dim = self.mask.shape[0:2]
            image_dim = self.cv2_image.shape[0:2]
            if mask_dim != image_dim:
                messagebox.showerror(
                    message=f"Dimension mismatch\nInput:{image_dim}\nMask:{mask_dim}"
                )
                return

        cv2_image_copy = self.cv2_image.copy()

        # draw filename
        if self.show_filename.get():
            filename = self.get_realpath()
            if filename is None:
                filename = "ERROR: NO NAME"

            cv2.putText(
                cv2_image_copy,
                text=os.path.basename(filename),
                org=(10, cv2_image_copy.shape[0] - 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=TEXT_COLOR,
                thickness=2,
            )

        # draw bounding boxes and bounds
        (bb_color, bb_width) = bb_params
        (outsider_color, outsider_width) = outsider_params
        bounds = [(0, upper_pixel), (self.cv2_image.shape[1], lower_pixel)]
        for row in self.values.itertuples():
            bb_area = (row.xmax - row.xmin) * (row.ymax - row.ymin)

            if bb_area <= 0:
                # not sure if this can happen
                continue

            outsider_thres = self.outsider_thres.get()
            is_outsider = False

            # handle bounds
            if not self.disable_bounds.get():
                # compute the area of intersection
                bb = [(row.xmin, row.ymin), (row.xmax, row.ymax)]
                max_of_x_min = max(bounds[0][0], bb[0][0])
                max_of_y_min = max(bounds[0][1], bb[0][1])
                min_of_x_max = min(bounds[1][0], bb[1][0])
                min_of_y_max = min(bounds[1][1], bb[1][1])
                w = min_of_x_max - max_of_x_min
                h = min_of_y_max - max_of_y_min
                intersect = w * h if w > 0 and h > 0 else 0

                intersect_ratio = intersect / bb_area

                if intersect_ratio < outsider_thres:
                    is_outsider = True

                # draw bound rectangle
                (bounds_color, bounds_width) = bounds_params
                cv2.rectangle(
                    cv2_image_copy,
                    (0, upper_pixel),
                    (cv2_image_copy.shape[1], lower_pixel),
                    bounds_color,
                    bounds_width,
                )

            # handle mask
            if enable_mask:
                if self.mask is None:
                    # this should never happen
                    messagebox.showerror(message="Internal error: Mask is None")
                    return

                mask_cropped = self.mask[row.ymin : row.ymax, row.xmin : row.xmax]
                whites = numpy.sum(mask_cropped == 255)
                mask_intersect_ratio = whites / bb_area
                if mask_intersect_ratio < self.mask_thres.get():
                    is_outsider = True

            box_color = outsider_color if is_outsider else bb_color
            box_width = outsider_width if is_outsider else bb_width

            hide_detect = self.hide_outsiders.get() and is_outsider

            if not hide_detect:
                cv2.rectangle(
                    cv2_image_copy,
                    (row.xmin, row.ymin),
                    (row.xmax, row.ymax),
                    box_color,
                    box_width,
                )
                if self.show_confidence.get():
                    cv2.putText(
                        cv2_image_copy,
                        text=f"{row.confidence:.2f}",
                        org=(row.xmin, row.ymin - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=TEXT_COLOR,
                        thickness=2,
                    )

        self.pil_image = Image.fromarray(cv2_image_copy)

        self.fit_image()


print("Initializing...")
root = tkinter.Tk()
root.geometry("1600x1000")
root.title("YOLOv5 Interactive Viewer")
# configureしないと伸びない
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

view = YoloV5InteractiveViewer(root)
print("Initialized")
root.mainloop()
