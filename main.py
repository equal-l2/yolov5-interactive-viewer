#!/usr/bin/env python3

from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import json
import os
import tkinter
import traceback
import typing

import cv2

from structs import AppConfig, ModelParam
import consts
import logic
from widgets import ZeroToOneScale, LoadFileButton, LineConfig


class ModelParamControl(ttk.Frame):
    def __init__(self, root: tkinter.Misc):
        ttk.Frame.__init__(self, root)

        self.confidence = ZeroToOneScale(
            self, label="Confidence", init=consts.CONFIDENCE_DEFAULT
        )
        self.confidence.pack()

        self.iou = ZeroToOneScale(self, label="IoU", init=consts.IOU_DEFAULT)
        self.iou.pack()

        self.augment = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            self, text="Enable augmentation on inference", variable=self.augment
        ).pack()

    def get(self) -> ModelParam:
        return ModelParam(self.confidence.get(), self.iou.get(), self.augment.get())

    def set(self, confidence: float, iou: float, augment: bool):
        self.confidence.set(confidence)
        self.iou.set(iou)
        self.augment.set(augment)


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
        self.model: typing.Optional[consts.MODEL_TYPE] = None
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

        ttk.Button(parent, text="Save picture", command=self.save_image).pack()

    def save_image(self):
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
                message="Bounds: Line width must be a positive interger"
            )
            return None

        model_param = self.model_param_control.get()

        return AppConfig(
            confidence=model_param.confidence,
            iou=model_param.iou,
            augment=model_param.augment,
            bb_color=bb_params.color,
            bb_width=bb_params.width,
            show_confidence=self.show_confidence.get(),
            outsider_color=outsider_params.color,
            outsider_width=outsider_params.width,
            hide_outsiders=self.hide_outsiders.get(),
            bounds_color=bounds_params.color,
            bounds_width=bounds_params.width,
            mask_thres=self.mask_thres.get(),
        )

    def from_config(self, app_config: AppConfig):
        self.model_param_control.set(
            app_config.confidence, app_config.iou, app_config.augment
        )

        self.bb_config.set(
            color=logic.rgb2hex(app_config.bb_color), width=app_config.bb_width
        )

        self.outsider_config.set(
            color=logic.rgb2hex(app_config.outsider_color),
            width=app_config.outsider_width,
        )

        self.bounds_config.set(
            color=logic.rgb2hex(app_config.bounds_color),
            width=app_config.bounds_width,
        )
        self.mask_thres.set(app_config.mask_thres)

        self.hide_outsiders.set(app_config.hide_outsiders)
        self.show_confidence.set(app_config.show_confidence)

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
        filename = filedialog.askopenfilename(
            title="Choose config", filetypes=[("json", "*.json")]
        )
        if filename == "":
            # canceled
            return

        try:
            with open(filename, "r") as f:
                config_json = json.load(f)
            app_config = AppConfig.parse_obj(config_json)

            self.from_config(app_config)
            messagebox.showinfo(message=f"Successfully loaded from {filename}")
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load config")

    def configure_right_sidebar(self):
        parent = self.right_sidebar

        config_io = ttk.LabelFrame(parent, text="Config")
        ttk.Button(config_io, text="Export config", command=self.export_config).pack()
        ttk.Button(config_io, text="Import config", command=self.import_config).pack()
        config_io.pack()

        # model config BEGIN
        model_config = ttk.LabelFrame(parent, text="Model")
        model_config.pack()

        self.load_model_button = LoadFileButton(
            model_config, "Load model", self.load_model
        )
        self.load_model_button.pack()

        self.model_param_control = ModelParamControl(model_config)
        self.model_param_control.pack()
        # model config END

        # mask config BEGIN
        mask_config_frame = ttk.LabelFrame(parent, text="Mask")

        self.enable_mask = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_config_frame,
            text="Apply mask as postprocess",
            variable=self.enable_mask,
        ).pack()

        self.load_mask_button = LoadFileButton(
            mask_config_frame, "Load mask", self.load_mask
        )
        self.load_mask_button.pack()

        self.bounds_config = LineConfig(
            mask_config_frame,
            color=consts.BOUNDS_COLOR_DEFAULT,
            width=1,
        )
        self.bounds_config.pack()

        self.mask_thres = ZeroToOneScale(
            mask_config_frame, label="Min overlap %", init=consts.MASK_THRES_DEFAULT
        )
        self.mask_thres.pack()

        mask_config_frame.pack()
        # mask config END

        # bb config BEGIN
        bb_config_frame = ttk.LabelFrame(parent, text="Bounding boxes")
        self.bb_config = LineConfig(
            bb_config_frame, color=consts.BBOXES_COLOR_DEFAULT, width=2
        )
        self.bb_config.pack()
        bb_config_frame.pack()
        # bb config END

        # mark outside of the mask
        outsider_config_frame = ttk.LabelFrame(parent, text="Outsiders")
        self.outsider_config = LineConfig(
            outsider_config_frame, color=consts.OUTSIDER_COLOR_DEFAULT, width=2
        )
        self.outsider_config.pack()
        outsider_config_frame.pack()

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

        self.show_filename = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame, text="Show filename in the picture", variable=self.show_filename
        ).pack()

        ttk.Button(parent, text="Rerender", command=self.render_result).pack()

        ttk.Separator(parent).pack()

        ttk.Button(parent, text="Run Detection", command=self.run_detect).pack()

    def fit_image(self):
        """scale the shown image to fit to the window"""
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
        from yolov5.helpers import load_model

        filename = filedialog.askopenfilename(title="Choose Model")
        if filename == "":
            # canceled
            return

        print(f"Load model {filename}")
        try:
            self.model = load_model(filename)
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load the model")
            return

        self.load_model_button.set_filename(os.path.basename(filename))

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
        self.load_mask_button.set_filename(os.path.basename(filename))

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
