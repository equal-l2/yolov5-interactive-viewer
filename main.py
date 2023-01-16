#!/usr/bin/env python3

from PIL import Image, ImageTk, ImageColor
from tkinter import ttk, filedialog, messagebox, colorchooser
import json
import os
import tkinter
import traceback
import typing

import cv2
import yolov5

from structs import RgbTuple, LineParam, AppConfig
import consts
import logic


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
            return LineParam(self.colorpicker.get(), width)

    def set(self, color: str, width: int):
        self.colorpicker.set(color)
        self.width_entry.set(width)


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

        ttk.Label(self, text=text).pack(side=tkinter.LEFT)
        ttk.Button(self, text="Choose...", command=self.choose_line_color).pack(
            side=tkinter.LEFT
        )
        self.color_label = tkinter.Label(self, text="     ")
        self.color_label.pack(side=tkinter.LEFT)

        self.set(color)

    def choose_line_color(self):
        colors = colorchooser.askcolor()
        if colors is not None and colors[1] is not None:
            self._set_color(colors[1])

    def _set_color(self, color: str):
        self.color = color
        self.color_label["bg"] = color

    def get(self) -> RgbTuple:
        return ImageColor.getrgb(self.color)[0:3]

    def set(self, color: str):
        ImageColor.getrgb(color)  # test the color is valid
        self._set_color(color)


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

    def _update_image_index(self, new_index: int):
        self.image_index = new_index
        self.run_detect()

    def set_image_index(self, new_index: int):
        if self.file_list is None:
            return

        file_list = self.file_list
        if 0 <= new_index and new_index < file_list.size():
            file_list.see(new_index)
            file_list.selection_clear(0, tkinter.END)
            file_list.selection_set(new_index)
            self._update_image_index(new_index)

    def configure_left_sidebar(self):
        parent = self.left_sidebar
        ttk.Button(parent, text="Load folder", command=self.load_folder).pack()

        def on_list_selected(e: typing.Any):
            select = e.widget.curselection()
            if len(select) < 1:
                return
            self._update_image_index(select[0])

        self.file_list = tkinter.Listbox(parent, height=10)
        self.file_list.pack()
        self.file_list.bind("<<ListboxSelect>>", on_list_selected)

        def modify_index(delta: int):
            if self.file_list is None:
                return
            file_list = self.file_list
            select = file_list.curselection()
            if len(select) < 1:
                return
            current = select[0]
            next = current + delta
            self.set_image_index(next)

        nav_frame = ttk.Frame(parent)
        ttk.Button(nav_frame, text="<-", command=lambda: modify_index(-1)).grid(
            row=0, column=0
        )
        ttk.Button(nav_frame, text="->", command=lambda: modify_index(1)).grid(
            row=0, column=2
        )
        nav_frame.pack()

        ttk.Separator(parent, orient=tkinter.HORIZONTAL).pack()

        ttk.Button(parent, text="Save picture", command=self.save).pack()

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

    def get_config(self) -> typing.Optional[AppConfig]:
        # check params
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

        return AppConfig(
            confidence=self.confidence.get(),
            iou=self.iou.get(),
            bb_color=bb_params.color,
            bb_width=bb_params.width,
            show_confidence=self.show_confidence.get(),
            outsider_color=outsider_params.color,
            outsider_width=outsider_params.width,
            outsider_thres=self.outsider_thres.get(),
            hide_outsiders=self.hide_outsiders.get(),
            bounds_color=bounds_params.color,
            bounds_width=bounds_params.width,
            upper_pixel=upper_pixel,
            lower_pixel=lower_pixel,
            disable_bounds=self.disable_bounds.get(),
            mask_thres=self.mask_thres.get(),
        )

    def from_config(self, app_config: AppConfig):
        self.confidence.set(app_config.confidence)
        self.iou.set(app_config.iou)
        self.bb_config.set(
            color=logic.rgb2hex(app_config.bb_color), width=app_config.bb_width
        )
        self.show_confidence.set(app_config.show_confidence)
        self.outsider_config.set(
            color=logic.rgb2hex(app_config.outsider_color),
            width=app_config.outsider_width,
        )
        self.outsider_thres.set(app_config.outsider_thres)
        self.hide_outsiders.set(app_config.hide_outsiders)
        self.bounds_config.set(
            color=logic.rgb2hex(app_config.bounds_color),
            width=app_config.bounds_width,
        )
        self.upper_pixel.set(app_config.upper_pixel)
        self.lower_pixel.set(app_config.lower_pixel)
        self.disable_bounds.set(app_config.disable_bounds)
        self.mask_thres.set(app_config.mask_thres)

    def export_config(self):
        real_new_name = filedialog.asksaveasfilename(initialfile="config.json")

        if real_new_name == "":
            # canceled
            return

        try:
            app_config = self.get_config()
            if app_config is None:
                return
            with open(real_new_name, "w") as f:
                f.write(app_config.json())
            messagebox.showinfo(message=f"Successfully saved to {real_new_name}")
        except:
            traceback.print_exc()
            messagebox.showinfo(message=f"Failed to save to {real_new_name}")

    def import_config(self):
        filename = filedialog.askopenfilename(title="Choose config")
        if filename == "":
            # canceled
            return

        try:
            with open(filename, "r") as f:
                config_json = json.load(f)
            app_config = AppConfig.parse_obj(config_json)

            self.from_config(app_config)
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load config")
            return

    def configure_right_sidebar(self):
        parent = self.right_sidebar

        config_io = ttk.LabelFrame(parent, text="Config")
        ttk.Button(config_io, text="Export config", command=self.export_config).pack()
        ttk.Button(config_io, text="Import config", command=self.import_config).pack()
        config_io.pack()

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
            model_config, label="Confidence", init=consts.CONFIDENCE_DEFAULT
        )
        self.confidence.pack()

        self.iou = ZeroToOneScale(model_config, label="IoU", init=consts.IOU_DEFAULT)
        self.iou.pack()

        self.bb_config = LineConfigs(
            parent, text="Bounding Boxes", color=consts.BBOXES_COLOR_DEFAULT, width=2
        )
        self.bb_config.pack()

        # mark outside of bounds
        self.outsider_config = LineConfigs(
            parent, text="Outsiders", color=consts.OUTSIDER_COLOR_DEFAULT, width=2
        )
        self.outsider_thres = ZeroToOneScale(
            self.outsider_config,
            label="Min overlap %",
            init=consts.OUTSIDE_THRES_DEFAULT,
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
            mask_config, label="Min overlap %", init=consts.MASK_THRES_DEFAULT
        )
        self.mask_thres.pack()

        mask_config.pack()

        self.bounds_config = LineConfigs(
            parent,
            text="Upper/Lower Bounds",
            color=consts.BOUNDS_COLOR_DEFAULT,
            width=1,
        )
        self.bounds_config.pack()

        self.upper_pixel = IntEntry(
            self.bounds_config, label="Up Px", init=consts.UPPER_BOUND_DEFAULT
        )
        self.lower_pixel = IntEntry(
            self.bounds_config, label="Lo Px", init=consts.LOWER_BOUND_DEFAULT
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

        ttk.Button(parent, text="Rerender", command=self.render_result).pack()

        ttk.Separator(parent).pack()

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
            self.set_image_index(0)

    def get_realpath(self):
        if self.image_index >= len(self.realpathes):
            # logic error (not because of users)
            # usually occurs when no image is loaded
            return None
        return self.realpathes[self.image_index]

    def run_detect(self):
        if self.model is None:
            messagebox.showerror(message="Model is not loaded")
            return

        filename = self.get_realpath()
        if filename is None:
            print("filename is None")
            return

        config = self.get_config()
        if config is None:
            return

        self.cv2_image = cv2.imread(filename)
        cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB, dst=self.cv2_image)

        values = logic.run_detect(self.model, self.cv2_image, config)

        self.values = values
        self.render_result()

    def render_result(self):
        # check detect is done
        if self.values is None:
            return

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

        # set filename
        filename = None
        if self.show_filename.get():
            realpath = self.get_realpath()
            if realpath is None:
                filename = "ERROR: NO NAME"
            else:
                filename = os.path.basename(realpath)

        config = self.get_config()
        if config is None:
            return

        image_copied = self.cv2_image.copy()
        logic.draw_result(
            values=self.values,
            cv2_image=image_copied,
            filename=filename,
            mask=self.mask if enable_mask else None,
            text_color=consts.TEXT_COLOR,
            config=config,
        )

        self.pil_image = Image.fromarray(image_copied)

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
