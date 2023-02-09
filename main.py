#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tkinter
import traceback
import typing
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import consts
import logic
from structs import AppConfig, ModelParam, ViewerInitConfig
from widgets import LineConfig, LineParam, LoadFileButton, TkCommand, ZeroToOneScale


class YoloV5InteractiveViewer(ttk.Frame):
    # internal states
    model: consts.MODEL_TYPE | None = None
    image_index: int = 0
    mask: logic.Mask | None = None
    cv2_image: logic.Cv2Image | None = None
    pil_image: Image.Image | None = None
    values: logic.DetectValues | None = None

    # tk objects
    left_sidebar: LeftSidebar
    right_sidebar: RightSidebar
    image_view: tkinter.Canvas
    tk_image: ImageTk.PhotoImage | None = None

    def __init__(
        self,
        root: tkinter.Misc,
        init_config: ViewerInitConfig | None,
    ) -> None:
        ttk.Frame.__init__(self, root)

        # place widgets
        self.left_sidebar = LeftSidebar(self, self)
        self.left_sidebar.grid(column=0, row=0, sticky=tkinter.NS + tkinter.W)
        self.left_sidebar["borderwidth"] = 1

        self.image_view = tkinter.Canvas(self)
        self.image_view.grid(column=1, row=0, sticky=tkinter.NSEW)
        self.image_view.configure(bg="gray")
        self.image_view.bind("<Configure>", func=lambda _: self.fit_image())

        self.right_sidebar = RightSidebar(self, self)
        self.right_sidebar.grid(column=2, row=0, sticky=tkinter.NS + tkinter.E)
        self.right_sidebar["borderwidth"] = 1

        # configureしないと伸びない
        self.columnconfigure(1, weight=10)
        self.rowconfigure(0, weight=1)

        if init_config is not None:
            self.handle_init_config(init_config)

    def handle_init_config(self, init_config: ViewerInitConfig) -> None:
        if init_config.model_path is not None:
            self.load_model(init_config.model_path)
        if init_config.mask_path is not None:
            self.load_mask(init_config.mask_path)
        if init_config.config_path is not None:
            self.import_config(init_config.config_path)

    def update_image_index(self, new_index: int) -> None:
        self.image_index = new_index
        self.run_detect()

    def clear_image(self) -> None:
        self.pil_image = None
        self.cv2_image = None
        self.tk_image = None
        self.image_view.delete("all")

    def save_image(self) -> None:
        if self.pil_image is None:
            return

        original_path = self.get_realpath()
        if original_path is None:
            return

        original_basename: str = Path(original_path).name
        original_name, ext = os.path.splitext(original_basename)  # noqa: PTH122
        new_name: str = original_name + "_detected" + ext

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

    def export_config(self, filename: str) -> None:
        try:
            app_config = self.right_sidebar.get_config()
            if app_config is None:
                return
            with Path(filename).open("w") as f:
                f.write(app_config.json(indent=2))
            messagebox.showinfo(message=f"Successfully saved to {filename}")
        except:
            traceback.print_exc()
            messagebox.showinfo(message=f"Failed to save to {filename}")

    def import_config(self, filename: str) -> None:
        try:
            with Path(filename).open() as f:
                config_json = json.load(f)
            app_config = AppConfig.parse_obj(config_json)

            self.right_sidebar.from_config(app_config)
        except:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load config")

    def fit_image(self) -> None:
        """scale the shown image to fit to the window"""
        if self.pil_image is not None:
            # clear image view
            self.image_view.delete("all")

            width = self.image_view.winfo_width()
            height = self.image_view.winfo_height()

            copied = self.pil_image.copy()
            copied.thumbnail((width, height))

            self.tk_image = ImageTk.PhotoImage(image=copied)

            self.image_view.create_image(0, 0, image=self.tk_image, anchor="nw")

    def load_model(self, filename: str) -> None:
        from yolov5.helpers import load_model

        print(f"Load model {filename}")
        try:
            self.model = load_model(filename)
        except:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load the model")
            return

        self.right_sidebar.set_model_filename(Path(filename).name)

    def load_mask(self, filename: str) -> None:
        print(f"Load mask {filename}")
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load the mask")
            return

        self.mask = logic.Mask(mask)
        self.right_sidebar.set_mask_filename(Path(filename).name)
        self.right_sidebar.turn_mask_on()

    def get_realpath(self) -> str | None:
        return self.left_sidebar.get_realpath(self.image_index)

    def run_detect(self) -> None:
        if self.model is None:
            messagebox.showerror(message="Model is not loaded")
            return

        filename = self.get_realpath()
        if filename is None:
            print("filename is None")
            return

        config = self.right_sidebar.get_config()
        if config is None:
            return

        self.cv2_image = cv2.imread(filename)
        if self.cv2_image is None:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load the mask")
            return
        cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB, dst=self.cv2_image)

        values = logic.run_detect(self.model, self.cv2_image, config)

        self.values = values  # noqa: PD011
        self.render_result()

    def render_result(self) -> None:
        if self.cv2_image is None:
            # image is not loaded yet
            return

        # check detect is done
        if self.values is None:  # noqa: PD011
            return

        # check mask
        enable_mask = self.right_sidebar.is_mask_on()
        if enable_mask:
            if self.mask is None:
                messagebox.showerror(message="Mask is not loaded")
                return

            mask_dim = self.mask.img.shape[0:2]
            image_dim = self.cv2_image.shape[0:2]
            if mask_dim != image_dim:
                messagebox.showerror(
                    message=f"Dimension mismatch\nInput:{image_dim}\nMask:{mask_dim}",
                )
                return

        # set filename
        filename = None
        if self.right_sidebar.is_filename_shown():
            realpath = self.get_realpath()
            if realpath is None:
                filename = "ERROR: NO NAME"
            else:
                filename = Path(realpath).name

        config = self.right_sidebar.get_config()
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
    file_list: tkinter.Listbox
    realpathes: list[str] = []

    proxy: Proxy

    class Proxy:
        control: YoloV5InteractiveViewer

        def __init__(self, control: YoloV5InteractiveViewer) -> None:
            self.control = control

        def update_image_index(self, index: int) -> None:
            self.control.update_image_index(index)

        def has_model(self) -> bool:
            return self.control.model is not None

        def clear_image(self) -> None:
            self.control.clear_image()

        def save_image(self) -> None:
            self.control.save_image()

    def __init__(self, root: tkinter.Misc, control: YoloV5InteractiveViewer) -> None:
        ttk.Frame.__init__(self, root)
        self.proxy = self.Proxy(control)

        # load folder button
        ttk.Button(self, text="Load folder", command=self.load_folder).pack()

        # file list
        def on_list_selected(_: typing.Any) -> None:
            select = self.file_list.curselection()
            if len(select) < 1:
                return
            self.proxy.update_image_index(select[0])

        self.file_list = tkinter.Listbox(self, height=10)
        self.file_list.pack()
        self.file_list.bind("<<ListboxSelect>>", on_list_selected)

        # next/prev button
        def modify_index(delta: int) -> None:
            select = self.file_list.curselection()
            if len(select) < 1:
                return
            current_idx: int = select[0]
            next_idx = current_idx + delta
            self.set_image_index(next_idx)

        nav_frame = ttk.Frame(self)
        ttk.Button(nav_frame, text="<-", command=lambda: modify_index(-1)).grid(
            row=0,
            column=0,
        )
        ttk.Button(nav_frame, text="->", command=lambda: modify_index(1)).grid(
            row=0,
            column=2,
        )
        nav_frame.pack()

        ttk.Separator(self, orient=tkinter.HORIZONTAL).pack()

        # save picture button
        ttk.Button(self, text="Save picture", command=self.proxy.save_image).pack()

    def set_image_index(self, new_index: int) -> None:
        file_list = self.file_list
        if 0 <= new_index < file_list.size():
            file_list.see(new_index)
            file_list.selection_clear(0, tkinter.END)
            file_list.selection_set(new_index)
            self.proxy.update_image_index(new_index)

    def load_folder(self) -> None:
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
            ext = Path(f).suffix
            valid_image_ext = [".jpg", ".jpeg", ".png"]
            if ext.lower() in valid_image_ext:
                realpath = str(Path(folder) / f)
                images.append((f, realpath))

        # sort files by name
        images.sort(key=lambda y: y[0])

        image_names = [name for (name, _) in images]
        self.realpathes = [realpath for (_, realpath) in images]

        # populate file list
        tk_imagelist = tkinter.StringVar(
            value=image_names,  # pyright: ignore [reportGeneralTypeIssues]
        )
        self.file_list["listvariable"] = tk_imagelist
        self.set_image_index(0)

    def get_realpath(self, index: int) -> str | None:
        if index >= len(self.realpathes):
            # logic error (not because of users)
            # usually occurs when no image is loaded
            return None
        return self.realpathes[index]


class RightSidebar(ttk.Frame):
    _model_config: ModelConfig
    _render_config: RenderConfig

    proxy: Proxy

    class Proxy:
        control: YoloV5InteractiveViewer

        def __init__(self, control: YoloV5InteractiveViewer) -> None:
            self.control = control

        def run_detect(self) -> None:
            self.control.run_detect()

        def render_result(self) -> None:
            self.control.render_result()

        def import_config(self, filename: str) -> None:
            self.control.import_config(filename)

        def export_config(self, filename: str) -> None:
            self.control.export_config(filename)

        def load_model(self, filename: str) -> None:
            self.control.load_model(filename)

        def load_mask(self, filename: str) -> None:
            self.control.load_model(filename)

    def __init__(self, root: tkinter.Misc, control: YoloV5InteractiveViewer) -> None:
        ttk.Frame.__init__(self, root)
        self.proxy = self.Proxy(control)

        config_io = ttk.LabelFrame(self, text="Config")
        ttk.Button(config_io, text="Export config", command=self.export_config).pack()
        ttk.Button(config_io, text="Import config", command=self.import_config).pack()
        config_io.pack()

        self._model_config = ModelConfig(self, control)
        self._model_config.pack()

        self._render_config = RenderConfig(
            self,
            control,
            command=self.proxy.render_result,
        )
        self._render_config.pack()

        ttk.Button(
            self,
            text="Re-run Detection",
            command=self.proxy.run_detect,
        ).pack()

    def import_config(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose config",
            filetypes=[("json", "*.json")],
        )
        if filename == "":
            # canceled
            return
        self.proxy.import_config(filename)

    def export_config(self) -> None:
        filename = filedialog.asksaveasfilename(initialfile="config.json")

        if filename == "":
            # canceled
            return

        self.proxy.export_config(filename)

    def get_config(self) -> AppConfig | None:
        render_params = self._render_config.get_param()
        model_param = self._model_config.get_param()

        return AppConfig(
            confidence=model_param.confidence,
            iou=model_param.iou,
            augment=model_param.augment,
            bb_color=render_params.bb_params.color,
            bb_width=render_params.bb_params.width,
            show_outsiders=render_params.show_outsiders,
            outsider_color=render_params.outsider_params.color,
            outsider_width=render_params.outsider_params.width,
            mask_thres=render_params.mask_thres,
            show_mask_border=render_params.show_mask_border,
            mask_border_color=render_params.mask_border_params.color,
            mask_border_width=render_params.mask_border_params.width,
            show_confidence=render_params.show_confidence,
        )

    def from_config(self, app_config: AppConfig) -> None:
        self._model_config.from_param(
            ModelParam(
                confidence=app_config.confidence,
                iou=app_config.iou,
                augment=app_config.augment,
            ),
        )

        render_params = RenderConfig.Param(
            bb_params=LineParam(app_config.bb_color, app_config.bb_width),
            outsider_params=LineParam(
                app_config.outsider_color,
                app_config.outsider_width,
            ),
            mask_border_params=LineParam(
                app_config.mask_border_color,
                app_config.mask_border_width,
            ),
            mask_thres=app_config.mask_thres,
            show_mask_border=app_config.show_mask_border,
            show_outsiders=app_config.show_outsiders,
            show_confidence=app_config.show_confidence,
        )
        self._render_config.from_param(render_params)

    def is_filename_shown(self) -> bool:
        return self._render_config.is_filename_shown()

    def is_mask_on(self) -> bool:
        return self._render_config.is_mask_on()

    def turn_mask_on(self) -> None:
        self._render_config.turn_mask_on()

    def set_model_filename(self, filename: str) -> None:
        self._model_config.set_model_filename(filename)

    def set_mask_filename(self, filename: str) -> None:
        self._render_config.set_mask_filename(filename)


class ModelConfig(ttk.Frame):
    _load_model_button: LoadFileButton
    _model_param_control: ModelParamControl

    _proxy: Proxy

    class Proxy:
        control: YoloV5InteractiveViewer

        def __init__(self, control: YoloV5InteractiveViewer) -> None:
            self.control = control

        def load_model(self, filename: str) -> None:
            self.control.load_model(filename)

    _command: TkCommand

    def __init__(
        self,
        root: tkinter.Misc,
        control: YoloV5InteractiveViewer,
        *,
        command: TkCommand = None,
    ) -> None:
        ttk.Frame.__init__(self, root)
        self._proxy = self.Proxy(control)
        self.command = command

        # model config
        model_config = ttk.LabelFrame(self, text="Model")
        model_config.pack()

        self._load_model_button = LoadFileButton(
            model_config,
            "Load model",
            command=self.load_model,
        )
        self._load_model_button.pack()

        self._model_param_control = ModelParamControl(model_config, command=command)
        self._model_param_control.pack()

    def load_model(self) -> None:
        filename = filedialog.askopenfilename(title="Choose Model")
        if filename == "":
            # canceled
            return

        self._proxy.load_model(filename)
        if self.command is not None:
            self.command()

    def get_param(self) -> ModelParam:
        return self._model_param_control.get()

    def from_param(self, param: ModelParam) -> None:
        self._model_param_control.from_param(param)

    def set_model_filename(self, filename: str) -> None:
        self._load_model_button.set_filename(filename)


class ModelParamControl(ttk.Frame):
    _confidence: ZeroToOneScale
    _iou: ZeroToOneScale
    _augment: tkinter.BooleanVar

    _command: TkCommand

    def __init__(self, root: tkinter.Misc, *, command: TkCommand = None) -> None:
        ttk.Frame.__init__(self, root)

        self._command = command

        self._confidence = ZeroToOneScale(
            self,
            label="Confidence",
            init=consts.CONFIDENCE_DEFAULT,
            command=command,
        )
        self._confidence.pack()

        self._iou = ZeroToOneScale(
            self,
            label="IoU",
            init=consts.IOU_DEFAULT,
            command=command,
        )
        self._iou.pack()

        self._augment = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            self,
            text="Enable augmentation on inference",
            variable=self._augment,
            command=self._button_command,
        ).pack()

    def get(self) -> ModelParam:
        return ModelParam(self._confidence.get(), self._iou.get(), self._augment.get())

    def set(self, confidence: float, iou: float, augment: bool) -> None:
        self._confidence.set(confidence)
        self._iou.set(iou)
        self._augment.set(augment)

    def from_param(self, param: ModelParam) -> None:
        self.set(param.confidence, param.iou, param.augment)

    def _button_command(self) -> None:
        if self._command is not None:
            self._command()


class RenderConfig(ttk.Frame):
    # Tk widgets
    _load_mask_button: LoadFileButton
    _mask_border_config: LineConfig
    _mask_thres: ZeroToOneScale
    _bb_config: LineConfig
    _outsider_config: LineConfig

    # Tk variables
    _show_mask_border: tkinter.BooleanVar
    _show_outsiders: tkinter.BooleanVar
    _show_confidence: tkinter.BooleanVar

    # volatile variables (don't save to config)
    _enable_mask: tkinter.BooleanVar
    _show_filename: tkinter.BooleanVar

    _proxy: Proxy

    class Proxy:
        control: YoloV5InteractiveViewer

        def __init__(self, control: YoloV5InteractiveViewer) -> None:
            self.control = control

        def load_mask(self, filename: str) -> None:
            self.control.load_mask(filename)

    _command: TkCommand

    @dataclass
    class Param:
        bb_params: LineParam
        outsider_params: LineParam
        mask_border_params: LineParam
        mask_thres: float
        show_mask_border: bool
        show_outsiders: bool
        show_confidence: bool

    def __init__(
        self,
        root: tkinter.Misc,
        control: YoloV5InteractiveViewer,
        *,
        command: TkCommand = None,
    ) -> None:
        ttk.Frame.__init__(self, root)
        self._proxy = self.Proxy(control)
        self.command = command

        # mask config
        mask_config_frame = ttk.LabelFrame(self, text="Mask")

        self._enable_mask = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_config_frame,
            text="Apply mask as postprocess",
            variable=self._enable_mask,
            command=self._run_command,
        ).pack()

        self._load_mask_button = LoadFileButton(
            mask_config_frame,
            "Load mask",
            command=self.load_mask,
        )
        self._load_mask_button.pack()

        self._show_mask_border = tkinter.BooleanVar(value=True)
        ttk.Checkbutton(
            mask_config_frame,
            text="Show borders of the mask",
            variable=self._show_mask_border,
            command=self._run_command,
        ).pack()

        self._mask_border_config = LineConfig(
            mask_config_frame,
            color=consts.BOUNDS_COLOR_DEFAULT,
            width=consts.BOUNDS_WIDTH_DEFAULT,
            command=command,
        )
        self._mask_border_config.pack()

        self._mask_thres = ZeroToOneScale(
            mask_config_frame,
            label="Threshold",
            init=consts.MASK_THRES_DEFAULT,
            command=command,
        )
        self._mask_thres.pack()

        mask_config_frame.pack()

        # bb config
        bb_config_frame = ttk.LabelFrame(self, text="Bounding boxes")
        self._bb_config = LineConfig(
            bb_config_frame,
            color=consts.BBOXES_COLOR_DEFAULT,
            width=consts.BBOXES_WIDTH_DEFAULT,
            command=command,
        )
        self._bb_config.pack()
        bb_config_frame.pack()

        # outsider config
        outsider_config_frame = ttk.LabelFrame(self, text="Outsiders")

        self._show_outsiders = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            outsider_config_frame,
            text="Show outsiders",
            variable=self._show_outsiders,
            command=self._run_command,
        ).pack()

        self._outsider_config = LineConfig(
            outsider_config_frame,
            color=consts.OUTSIDER_COLOR_DEFAULT,
            width=consts.OUTSIDER_WIDTH_DEFAULT,
            command=command,
        )
        self._outsider_config.pack()
        outsider_config_frame.pack()

        # misc.
        misc_frame = ttk.LabelFrame(self, text="Misc")
        misc_frame.pack()

        self._show_confidence = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame,
            text="Show confidence with bounding boxes",
            variable=self._show_confidence,
            command=self._run_command,
        ).pack()

        self._show_filename = tkinter.BooleanVar(value=False)
        ttk.Checkbutton(
            misc_frame,
            text="Show filename in the picture",
            variable=self._show_filename,
            command=self._run_command,
        ).pack()

    def _run_command(self) -> None:
        if self.command is not None:
            self.command()

    def get_param(self) -> Param:
        return self.Param(
            bb_params=self._bb_config.get(),
            outsider_params=self._outsider_config.get(),
            mask_border_params=self._mask_border_config.get(),
            mask_thres=self._mask_thres.get(),
            show_mask_border=self._show_mask_border.get(),
            show_outsiders=self._show_outsiders.get(),
            show_confidence=self._show_confidence.get(),
        )

    def from_param(self, param: Param) -> None:
        self._bb_config.from_param(param.bb_params)
        self._outsider_config.from_param(param.outsider_params)
        self._mask_border_config.from_param(param.mask_border_params)
        self._mask_thres.set(param.mask_thres)
        self._show_mask_border.set(param.show_mask_border)
        self._show_outsiders.set(param.show_outsiders)
        self._show_confidence.set(param.show_confidence)

    def is_filename_shown(self) -> bool:
        return self._show_filename.get()

    def is_mask_on(self) -> bool:
        return self._enable_mask.get()

    def turn_mask_on(self) -> None:
        return self._enable_mask.set(True)

    def set_mask_filename(self, filename: str) -> None:
        self._load_mask_button.set_filename(filename)

    def load_mask(self) -> None:
        filename = filedialog.askopenfilename(title="Choose Mask")
        if filename == "":
            # canceled
            return
        self._proxy.load_mask(filename)
        self._run_command()


if __name__ == "__main__":
    print("Initializing...")
    root = tkinter.Tk()
    root.geometry("1600x1000")
    root.title("YOLOv5 Interactive Viewer")
    # configureしないと伸びない
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    init_config: ViewerInitConfig | None = None
    if Path("init.json").is_file():
        try:
            with Path("init.json").open() as f:
                config_json = json.load(f)
            init_config = ViewerInitConfig.parse_obj(config_json)
        except:
            traceback.print_exc()
            messagebox.showerror(message="Failed to load init config")

    view = YoloV5InteractiveViewer(root, init_config)
    view.grid(column=0, row=0, sticky=tkinter.NSEW)
    print("Initialized")
    root.mainloop()
