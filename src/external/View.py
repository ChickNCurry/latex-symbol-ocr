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

        self.CANVAS_DIMS = (256, 256)
        self.RENDER_DIMS = (64, 64)
        self.ENTRY_WIDTH = 20
        self.TOP_K = 3
    
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.renders = [ImageTk.PhotoImage(
            Image.new(mode="RGB", size=self.RENDER_DIMS, color="white"))] * self.TOP_K
        self.markups = [tk.StringVar(), tk.StringVar(), tk.StringVar()]

        self.canvas = tk.Canvas(
            self.root, bg="white", width=self.CANVAS_DIMS[0], height=self.CANVAS_DIMS[1])
        self.canvas.bind(
            "<B1-Motion>", lambda e: self.controller.draw(e, self.CANVAS_DIMS))
        self.button_predict = ttk.Button(
            self.root, text="predict", command=self.controller.predict)
        self.button_clear = ttk.Button(
            self.root, text="clear", command=self.controller.clear)

        self.prediction_components = []
        for i in range(self.TOP_K):
            label_ranking = ttk.Label(self.root, text=f"{i + 1}.")
            label_render = ttk.Label(self.root, image=self.renders[i])
            entry_markup = ttk.Entry(
                self.root, width=self.ENTRY_WIDTH, textvariable=self.markups[i])
            button_copy = ttk.Button(
                self.root, text="copy", command=lambda: self._copy(i))
            self.prediction_components.append(
                (label_ranking, label_render, entry_markup, button_copy))

        self.canvas.grid(column=0, row=0, columnspan=3, rowspan=3)
        self.button_predict.grid(column=0, row=3, columnspan=3)
        self.button_clear.grid(column=3, row=3, columnspan=4)

        for i, c in enumerate(self.prediction_components):
            c[0].grid(column=3, row=i)
            c[1].grid(column=4, row=i)
            c[2].grid(column=5, row=i)
            c[3].grid(column=6, row=i)

    def run(self) -> None:
        self.root.mainloop()

    def update_drawing(self, coords: tuple[int, int], brush_size: tuple[int, int]) -> None:
        self.canvas.create_oval((coords[0] - brush_size[0],
                                 coords[1] - brush_size[1],
                                 coords[0] + brush_size[0],
                                 coords[1] + brush_size[1]), fill="black")

    def update_clearing(self) -> None:
        self.canvas.delete("all")
        self.renders = [ImageTk.PhotoImage(
            Image.new(mode="RGB", size=self.RENDER_DIMS, color="white"))] * 3
        for i in range(self.TOP_K):
            self.prediction_components[i][1].configure(image=self.renders[i])
        for m in self.markups:
            m.set("")

    def update_predictions(self, predictions: List[Prediction]) -> None:
        assert len(predictions) == self.TOP_K
        for i in range(self.TOP_K):
            self.markups[i].set(predictions[i].markup)
            render = predictions[i].render.resize(self.RENDER_DIMS)
            self.renders[i] = ImageTk.PhotoImage(render)
            self.prediction_components[i][1].configure(image=self.renders[i])

    def _copy(self, index: int) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.markups[index].get())
