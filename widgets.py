import tkinter
import typing
from tkinter import colorchooser, ttk

from PIL import ImageColor

from logic import rgb2hex
from structs import LineParam, RgbTuple

TkCommand = typing.Callable[[], None] | None


class LineConfig(ttk.Frame):
    def __init__(
        self,
        root: tkinter.Misc,
        color: str,
        width: int,
        command: TkCommand = None,
    ) -> None:
        ttk.Frame.__init__(self, root)

        self.width_scale = LineWidthScale(
            self,
            label="Line Width",
            init=width,
            command=command,
        )
        self.width_scale.pack()

        self.colorpicker = ColorPicker(
            self,
            text="Line color",
            color=color,
            command=command,
        )
        self.colorpicker.pack()

    def get(self) -> LineParam:
        return LineParam(self.colorpicker.get(), self.width_scale.get())

    def set(self, color: str, width: int) -> None:
        self.colorpicker.set(color)
        self.width_scale.set(width)

    def from_param(self, param: LineParam) -> None:
        color_str = rgb2hex(param.color)
        self.set(color_str, param.width)


class ColorPicker(ttk.Frame):
    color_label: tkinter.Label
    color: str
    command: TkCommand

    def __init__(
        self,
        root: tkinter.Misc,
        text: str,
        color: str,
        command: TkCommand = None,
    ) -> None:
        self.command = command
        ttk.Frame.__init__(self, root)

        ttk.Label(self, text=text).pack(side=tkinter.LEFT)
        ttk.Button(self, text="Choose...", command=self.handle_command).pack(
            side=tkinter.LEFT,
        )
        self.color_label = tkinter.Label(self, text="     ")
        self.color_label.pack(side=tkinter.LEFT)

        self.set(color)

    def choose_line_color(self) -> None:
        colors = colorchooser.askcolor()
        if colors[1] is not None:
            self._set_color(colors[1])

    def handle_command(self) -> None:
        self.choose_line_color()
        if self.command is not None:
            self.command()

    def _set_color(self, color: str) -> None:
        self.color = color
        self.color_label["bg"] = color

    def get(self) -> RgbTuple:
        return ImageColor.getrgb(self.color)[0:3]

    def set(self, color: str) -> None:
        ImageColor.getrgb(color)  # test the color is valid
        self._set_color(color)


class ZeroToOneScale(tkinter.Scale):
    command: TkCommand

    def __init__(
        self,
        root: tkinter.Misc,
        label: str,
        init: float,
        command: TkCommand = None,
    ) -> None:
        self.command = command
        tkinter.Scale.__init__(
            self,
            root,
            from_=0,
            to=1.0,
            resolution=0.01,
            label=label,
            orient=tkinter.HORIZONTAL,
            command=self._run_command,
        )
        self.set(init)

    def _run_command(self, _: str) -> None:
        if self.command is not None:
            self.command()


class LineWidthScale(tkinter.Scale):
    command: TkCommand

    def __init__(
        self,
        root: tkinter.Misc,
        label: str,
        init: int,
        command: TkCommand = None,
    ) -> None:
        from consts import LINE_WIDTH_MAX, LINE_WIDTH_MIN

        self.command = command

        tkinter.Scale.__init__(
            self,
            root,
            from_=LINE_WIDTH_MIN,
            to=LINE_WIDTH_MAX,
            resolution=1,
            label=label,
            orient=tkinter.HORIZONTAL,
            command=self._run_command,
        )
        self.set(init)

    def get(self) -> int:
        return int(super().get())

    def _run_command(self, _: str) -> None:
        if self.command is not None:
            self.command()


class LoadFileButton(ttk.Frame):
    def __init__(
        self,
        root: tkinter.Misc,
        text: str,
        command: typing.Callable[[], None],
    ) -> None:
        ttk.Frame.__init__(self, root)
        ttk.Button(self, text=text, command=command).pack(side=tkinter.LEFT)
        self.filename_label = ttk.Label(self)
        self.filename_label.pack(side=tkinter.LEFT)

    def set_filename(self, name: str) -> None:
        self.filename_label["text"] = name
