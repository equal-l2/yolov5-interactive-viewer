#!/usr/bin/env python3
from __future__ import annotations

from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import json
import os
import tkinter
import traceback
import typing
from typing import Optional

import cv2

from structs import AppConfig, ModelParam, ViewerInitConfig
import consts
import logic
from widgets import ZeroToOneScale, LoadFileButton, LineConfig, TkCommand


class ModelParamControl(ttk.Frame):
    confidence: ZeroToOneScale
    iou: ZeroToOneScale
    augment: tkinter.BooleanVar

    command: TkCommand

    def __init__(self, root: tkinter.Misc, *, command: TkCommand = None):
        ttk.Frame.__init__(self, root)

        self.command = command

        self.confidence = ZeroToOneScale(
            self, label="Confidence", init=consts.CONFIDENCE_DEFAULT, command=command
        )
        self.confidence.pack()

        self.iou = ZeroToOneScale(
            self, label="IoU", init=consts.IOU_DEFAULT, command=command
        )
        self.iou.pack()

        self.augment = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            self,
            text="Enable augmentation on inference",
            variable=self.augment,
            command=self._button_command,
        ).pack()

    def get(self) -> ModelParam:
        return ModelParam(self.confidence.get(), self.iou.get(), self.augment.get())

    def set(self, confidence: float, iou: float, augment: bool):
        self.confidence.set(confidence)
        self.iou.set(iou)
        self.augment.set(augment)

    def _button_command(self):
        if self.command is not None:
            self.command()


class YoloV5InteractiveViewer(ttk.Frame):
    # internal states
    model: Optional[consts.MODEL_TYPE] = None
    image_index: int = 0
    mask: Optional[logic.Mask] = None
    cv2_image: Optional[logic.Cv2Image]
    pil_image: Optional[Image.Image]
    values: logic.DetectValues

    # tk objects
    left_sidebar: LeftSidebar
    image_view: tkinter.Canvas
    tk_image: Optional[ImageTk.PhotoImage] = None
    load_model_button: LoadFileButton
    model_param_control: ModelParamControl
    load_mask_button: LoadFileButton
    mask_border_config: LineConfig
    mask_thres: ZeroToOneScale
    bb_config: LineConfig
    outsider_config: LineConfig

    # tk string variables
    enable_mask: tkinter.BooleanVar
    show_mask_border: tkinter.BooleanVar
    show_outsiders: tkinter.BooleanVar
    show_confidence: tkinter.BooleanVar
    show_filename: tkinter.BooleanVar

    def __init__(self, root: tkinter.Misc, init_config: Optional[ViewerInitConfig]):
        ttk.Frame.__init__(self, root)

        # place widgets
        self.left_sidebar = LeftSidebar(self)
        self.left_sidebar.grid(column=0, row=0, sticky=tkinter.NS + tkinter.W)
        self.left_sidebar["borderwidth"] = 1

        self.image_view = tkinter.Canvas(self)
        self.image_view.grid(column=1, row=0, sticky=tkinter.NSEW)
        self.image_view.configure(bg="gray")

        right_sidebar = ttk.Frame(self)
        right_sidebar.grid(column=2, row=0, sticky=tkinter.NS + tkinter.E)
        right_sidebar["borderwidth"] = 1

        # configureしないと伸びない
        self.columnconfigure(1, weight=10)
        self.rowconfigure(0, weight=1)

        self.configure_right_sidebar(right_sidebar)

        if init_config is not None:
            self.handle_init_config(init_config)

    def handle_init_config(self, init_config: ViewerInitConfig):
        if init_config.model_path is not None:
            self.load_model(init_config.model_path)
        if init_config.mask_path is not None:
            self.load_mask(init_config.mask_path)
        if init_config.config_path is not None:
            self.import_config(init_config.config_path)

    def update_image_index(self, new_index: int):
        self.image_index = new_index
        self.run_detect()

    def clear_image(self):
        self.pil_image = None
        self.cv2_image = None
        self.tk_image = None
        self.image_view.delete("all")

    def save_image(self):
        if self.pil_image is None:
            return

        original_path = self.get_realpath()
        if original_path is None:
            return

        original_basename = os.path.basename(original_path)
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

    def get_config(self) -> Optional[AppConfig]:
        bb_params = self.bb_config.get()
        outsider_params = self.outsider_config.get()
        mask_border_params = self.mask_border_config.get()
        model_param = self.model_param_control.get()

        return AppConfig(
            confidence=model_param.confidence,
            iou=model_param.iou,
            augment=model_param.augment,
            bb_color=bb_params.color,
            bb_width=bb_params.width,
            show_outsiders=self.show_outsiders.get(),
            outsider_color=outsider_params.color,
            outsider_width=outsider_params.width,
            mask_thres=self.mask_thres.get(),
            show_mask_border=self.show_mask_border.get(),
            mask_border_color=mask_border_params.color,
            mask_border_width=mask_border_params.width,
            show_confidence=self.show_confidence.get(),
        )

    def from_config(self, app_config: AppConfig):
        self.model_param_control.set(
            app_config.confidence, app_config.iou, app_config.augment
        )

        self.bb_config.set(
            color=logic.rgb2hex(app_config.bb_color), width=app_config.bb_width
        )

        self.show_outsiders.set(app_config.show_outsiders)
        self.outsider_config.set(
            color=logic.rgb2hex(app_config.outsider_color),
            width=app_config.outsider_width,
        )

        self.mask_thres.set(app_config.mask_thres)
        self.show_mask_border.set(app_config.show_mask_border)
        self.mask_border_config.set(
            color=logic.rgb2hex(app_config.mask_border_color),
            width=app_config.mask_border_width,
        )

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
                f.write(app_config.json(indent=2))
            messagebox.showinfo(message=f"Successfully saved to {real_new_name}")
        except:
            traceback.print_exc()
            messagebox.showinfo(message=f"Failed to save to {real_new_name}")

    def config_import_dialog(self):
        filename = filedialog.askopenfilename(
            title="Choose config", filetypes=[("json", "*.json")]
        )
        if filename == "":
            # canceled
            return
        self.import_config(filename)

    def import_config(self, filename: str):
        try:
            with open(filename, "r") as f:
                config_json = json.load(f)
            app_config = AppConfig.parse_obj(config_json)

            self.from_config(app_config)
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load config")

    def configure_right_sidebar(self, parent: tkinter.Misc):
        config_io = ttk.LabelFrame(parent, text="Config")
        ttk.Button(config_io, text="Export config", command=self.export_config).pack()
        ttk.Button(
            config_io, text="Import config", command=self.config_import_dialog
        ).pack()
        config_io.pack()

        # model config
        model_config = ttk.LabelFrame(parent, text="Model")
        model_config.pack()

        self.load_model_button = LoadFileButton(
            model_config, "Load model", command=self.model_load_dialog
        )
        self.load_model_button.pack()

        self.model_param_control = ModelParamControl(model_config)
        self.model_param_control.pack()

        # mask config
        mask_config_frame = ttk.LabelFrame(parent, text="Mask")

        self.enable_mask = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_config_frame,
            text="Apply mask as postprocess",
            variable=self.enable_mask,
        ).pack()

        self.load_mask_button = LoadFileButton(
            mask_config_frame, "Load mask", command=self.mask_load_dialog
        )
        self.load_mask_button.pack()

        self.show_mask_border = tkinter.BooleanVar(value=True)
        ttk.Checkbutton(
            mask_config_frame,
            text="Show borders of the mask",
            variable=self.show_mask_border,
        ).pack()

        self.mask_border_config = LineConfig(
            mask_config_frame,
            color=consts.BOUNDS_COLOR_DEFAULT,
            width=consts.BOUNDS_WIDTH_DEFAULT,
        )
        self.mask_border_config.pack()

        self.mask_thres = ZeroToOneScale(
            mask_config_frame, label="Threshold", init=consts.MASK_THRES_DEFAULT
        )
        self.mask_thres.pack()

        mask_config_frame.pack()
        # mask config END

        # bb config BEGIN
        bb_config_frame = ttk.LabelFrame(parent, text="Bounding boxes")
        self.bb_config = LineConfig(
            bb_config_frame,
            color=consts.BBOXES_COLOR_DEFAULT,
            width=consts.BBOXES_WIDTH_DEFAULT,
        )
        self.bb_config.pack()
        bb_config_frame.pack()
        # bb config END

        # mark outside of the mask
        outsider_config_frame = ttk.LabelFrame(parent, text="Outsiders")

        self.show_outsiders = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            outsider_config_frame, text="Show outsiders", variable=self.show_outsiders
        ).pack()

        self.outsider_config = LineConfig(
            outsider_config_frame,
            color=consts.OUTSIDER_COLOR_DEFAULT,
            width=consts.OUTSIDER_WIDTH_DEFAULT,
        )
        self.outsider_config.pack()
        outsider_config_frame.pack()

        misc_frame = ttk.LabelFrame(parent, text="Misc")
        misc_frame.pack()

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
        # clear image view
        self.image_view.delete("all")

        if self.pil_image is not None:
            width = self.image_view.winfo_width()
            height = self.image_view.winfo_height()
            copied = self.pil_image.copy()
            copied.thumbnail((width, height))
            # print(width, height)
            self.tk_image = ImageTk.PhotoImage(image=copied)
            self.image_view.delete("all")
            self.image_view.create_image(0, 0, image=self.tk_image, anchor="nw")

    def model_load_dialog(self):
        filename = filedialog.askopenfilename(title="Choose Model")
        if filename == "":
            # canceled
            return

        self.load_model(filename)

    def load_model(self, filename: str):
        from yolov5.helpers import load_model

        print(f"Load model {filename}")
        try:
            self.model = load_model(filename)
        except:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load the model")
            return

        self.load_model_button.set_filename(os.path.basename(filename))

    def mask_load_dialog(self):
        filename = filedialog.askopenfilename(title="Choose Mask")
        if filename == "":
            # canceled
            return
        self.load_mask(filename)

    def load_mask(self, filename: str):
        print(f"Load mask {filename}")
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load the mask")
            return

        self.mask = logic.Mask(mask)
        self.load_mask_button.set_filename(os.path.basename(filename))
        self.enable_mask.set(True)

    def get_realpath(self) -> Optional[str]:
        return self.left_sidebar.get_realpath(self.image_index)

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
        if self.cv2_image is None:
            # image is not loaded yet
            return

        # check detect is done
        if self.values is None:
            return

        # check mask
        enable_mask = self.enable_mask.get()
        if enable_mask:
            if self.mask is None:
                messagebox.showerror(message="Mask is not loaded")
                return

            mask_dim = self.mask.img.shape[0:2]
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


class LeftSidebar(ttk.Frame):
    class RootProxy:
        root: YoloV5InteractiveViewer

        def __init__(self, root: YoloV5InteractiveViewer):
            self.root = root

        def update_image_index(self, index: int):
            self.root.update_image_index(index)

        def has_model(self) -> bool:
            return self.root.model is not None

        def clear_image(self):
            self.root.clear_image()

    proxy: RootProxy
    file_list: tkinter.Listbox
    realpathes: list[str] = []

    def __init__(self, root: YoloV5InteractiveViewer):
        ttk.Frame.__init__(self, root)
        self.proxy = self.RootProxy(root)

        # load folder button
        ttk.Button(self, text="Load folder", command=self.load_folder).pack()

        # file list
        def on_list_selected(e: typing.Any):
            select = e.widget.curselection()
            if len(select) < 1:
                return
            self.proxy.update_image_index(select[0])

        self.file_list = tkinter.Listbox(self, height=10)
        self.file_list.pack()
        self.file_list.bind("<<ListboxSelect>>", on_list_selected)

        # next/prev button
        def modify_index(delta: int):
            select = self.file_list.curselection()
            if len(select) < 1:
                return
            current_idx: int = select[0]
            next_idx = current_idx + delta
            self.set_image_index(next_idx)

        nav_frame = ttk.Frame(self)
        ttk.Button(nav_frame, text="<-", command=lambda: modify_index(-1)).grid(
            row=0, column=0
        )
        ttk.Button(nav_frame, text="->", command=lambda: modify_index(1)).grid(
            row=0, column=2
        )
        nav_frame.pack()

        ttk.Separator(self, orient=tkinter.HORIZONTAL).pack()

        # save picture button
        ttk.Button(self, text="Save picture", command=root.save_image).pack()

    def set_image_index(self, new_index: int):
        file_list = self.file_list
        if 0 <= new_index and new_index < file_list.size():
            file_list.see(new_index)
            file_list.selection_clear(0, tkinter.END)
            file_list.selection_set(new_index)
            self.proxy.update_image_index(new_index)

    def load_folder(self):
        if not self.proxy.has_model():
            messagebox.showerror(message="You need to load a model first")
            return

        folder = filedialog.askdirectory(title="Choose Image Folder")
        if folder == "":
            # canceled
            return

        self.proxy.clear_image()
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
        self.file_list["listvariable"] = tk_imagelist
        self.set_image_index(0)

    def get_realpath(self, index: int) -> Optional[str]:
        if index >= len(self.realpathes):
            # logic error (not because of users)
            # usually occurs when no image is loaded
            return None
        return self.realpathes[index]


class RightSidebar(ttk.Frame):
    def __init__(self, root: YoloV5InteractiveViewer):
        ttk.Frame.__init__(self, root)


if __name__ == "__main__":
    print("Initializing...")
    root = tkinter.Tk()
    root.geometry("1600x1000")
    root.title("YOLOv5 Interactive Viewer")
    # configureしないと伸びない
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    init_config = None
    if os.path.isfile("init.json"):
        try:
            with open("init.json", "r") as f:
                config_json = json.load(f)
            init_config = ViewerInitConfig.parse_obj(config_json)
        except:
            traceback.print_exc()
            messagebox.showerror(message=f"Failed to load init config")

    view = YoloV5InteractiveViewer(root, init_config)
    view.grid(column=0, row=0, sticky=tkinter.NSEW)
    print("Initialized")
    root.mainloop()
