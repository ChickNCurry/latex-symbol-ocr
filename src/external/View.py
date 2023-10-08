import tkinter as tk
from tkinter import ttk
from typing import List

from PIL import Image, ImageTk

from src.controllers.Controller import Controller
from src.controllers.IControllerObserver import IControllerObserver
from src.domain.IPredictionsObserver import IPredictionsObserver
from src.domain.Prediction import Prediction


class View(IControllerObserver, IPredictionsObserver):
    def __init__(self, controller: Controller):
        self.controller = controller

        self.canvas_dims = (256, 256)
        self.render_dims = (64, 64)
        self.entry_width = 20

        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.buffers = [ImageTk.PhotoImage(Image.new(mode="RGB", size=self.render_dims, color="white"))] * 3
        self.markups = [tk.StringVar(), tk.StringVar(), tk.StringVar()]

        self.canvas = tk.Canvas(self.root, bg="white", width=self.canvas_dims[0], height=self.canvas_dims[1])
        self.canvas.bind("<B1-Motion>", lambda e: self.controller.draw(e, self.canvas_dims))
        self.button_predict = ttk.Button(self.root, text="predict", command=self.controller.predict)
        self.button_clear = ttk.Button(self.root, text="clear", command=self.controller.clear)
        self.label_1 = ttk.Label(self.root, text="1.")
        self.label_render_1 = ttk.Label(self.root, image=self.buffers[0])
        self.entry_markup_1 = ttk.Entry(self.root, width=self.entry_width, textvariable=self.markups[0])
        self.button_copy_1 = ttk.Button(self.root, text="copy", command=lambda: self.copy(0))
        self.label_2 = ttk.Label(self.root, text="2.")
        self.label_render_2 = ttk.Label(self.root, image=self.buffers[1])
        self.entry_markup_2 = ttk.Entry(self.root, width=self.entry_width, textvariable=self.markups[1])
        self.button_copy_2 = ttk.Button(self.root, text="copy", command=lambda: self.copy(1))
        self.label_3 = ttk.Label(self.root, text="3.")
        self.label_render_3 = ttk.Label(self.root, image=self.buffers[2])
        self.entry_markup_3 = ttk.Entry(self.root, width=self.entry_width, textvariable=self.markups[2])
        self.button_copy_3 = ttk.Button(self.root, text="copy", command=lambda: self.copy(2))
        self.labels_render = [self.label_render_1, self.label_render_2, self.label_render_3]

        self.canvas.grid(column=0, row=0, columnspan=3, rowspan=3)
        self.button_predict.grid(column=0, row=3, columnspan=3)
        self.button_clear.grid(column=3, row=3, columnspan=4)
        self.label_1.grid(column=3, row=0)
        self.label_render_1.grid(column=4, row=0)
        self.entry_markup_1.grid(column=5, row=0)
        self.button_copy_1.grid(column=6, row=0)
        self.label_2.grid(column=3, row=1)
        self.label_render_2.grid(column=4, row=1)
        self.entry_markup_2.grid(column=5, row=1)
        self.button_copy_2.grid(column=6, row=1)
        self.label_3.grid(column=3, row=2)
        self.label_render_3.grid(column=4, row=2)
        self.entry_markup_3.grid(column=5, row=2)
        self.button_copy_3.grid(column=6, row=2)

    def run(self):
        self.root.mainloop()

    def copy(self, index: int):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.markups[index].get())

    def update_drawing(self, x: int, y: int, brush_size: tuple[int, int]):
        self.canvas.create_oval((x - brush_size[0],
                                 y - brush_size[1],
                                 x + brush_size[0],
                                 y + brush_size[1]), fill="black")

    def update_clearing(self):
        self.canvas.delete("all")
        self.buffers = [ImageTk.PhotoImage(Image.new(mode="RGB", size=self.render_dims, color="white"))] * 3
        [self.labels_render[i].configure(image=self.buffers[i]) for i in range(3)]
        [m.set("") for m in self.markups]

    def update_predictions(self, predictions: List[Prediction]):
        assert len(predictions) == len(self.markups)
        for i in range(len(predictions)):
            self.markups[i].set(predictions[i].markup)
            render = predictions[i].render.resize(self.render_dims)
            self.buffers[i] = ImageTk.PhotoImage(render)
            self.labels_render[i].configure(image=self.buffers[i])
