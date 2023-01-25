from PIL import ImageColor
from tkinter import ttk, colorchooser
import tkinter
import traceback

from structs import RgbTuple, LineParam


class LineConfigs(ttk.Frame):
    def __init__(self, root: tkinter.Misc, color: str, width: int):
        ttk.Frame.__init__(self, root)

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
