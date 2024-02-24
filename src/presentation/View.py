import tkinter as tk
from tkinter import ttk
from typing import List, Tuple

from PIL import Image, ImageTk

from src.controllers.Controller import Controller
from src.controllers.models import IControllerObserver
from src.domain.models import IPredictionsObserver, Prediction
from src.presentation.models import PredictionComponent


class View(IControllerObserver, IPredictionsObserver):
    _CANVAS_DIMS = (256, 256)
    _RENDER_DIMS = (64, 64)
    _ENTRY_WIDTH = 20
    _TOP_K = 3

    def __init__(self, controller: Controller):
        self._controller = controller

        self._root = tk.Tk()
        self._root.protocol("WM_DELETE_WINDOW", self._root.destroy)
        self._root.resizable(False, False)

        self._renders = [self._create_blank_render() for _ in range(self._TOP_K)]
        self._markups = [tk.StringVar() for _ in range(self._TOP_K)]

        self._canvas = tk.Canvas(
            self._root,
            bg="white",
            width=self._CANVAS_DIMS[0],
            height=self._CANVAS_DIMS[1],
        )

        self._canvas.bind(
            "<B1-Motion>",
            lambda e: self._controller.draw(e, self._CANVAS_DIMS),
        )

        self._button_predict = ttk.Button(
            self._root, text="predict", command=self._controller.predict
        )

        self._button_clear = ttk.Button(
            self._root, text="clear", command=self._controller.clear
        )

        self._prediction_components = [
            self._create_prediction_component(i) for i in range(self._TOP_K)
        ]

        self._canvas.grid(column=0, row=0, columnspan=3, rowspan=3)
        self._button_predict.grid(column=0, row=3, columnspan=3)
        self._button_clear.grid(column=3, row=3, columnspan=4)

        for i, c in enumerate(self._prediction_components):
            c.label_ranking.grid(column=3, row=i)
            c.label_render.grid(column=4, row=i)
            c.entry_markup.grid(column=5, row=i)
            c.button_copy.grid(column=6, row=i)

    def _create_blank_render(self) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(
            Image.new(mode="RGB", size=self._RENDER_DIMS, color="white")
        )

    def _copy(self, index: int) -> None:
        self._root.clipboard_clear()
        self._root.clipboard_append(self._markups[index].get())

    def _create_prediction_component(self, index: int) -> PredictionComponent:
        label_ranking = ttk.Label(self._root, text=f"{index + 1}.")
        label_render = ttk.Label(self._root, image=self._renders[index])
        entry_markup = ttk.Entry(
            self._root, width=self._ENTRY_WIDTH, textvariable=self._markups[index]
        )
        button_copy = ttk.Button(
            self._root, text="copy", command=lambda: self._copy(index)
        )
        return PredictionComponent(
            label_ranking,
            label_render,
            entry_markup,
            button_copy,
        )

    def run(self) -> None:
        self._root.mainloop()

    def update_drawing(
        self, coords: Tuple[int, int], brush_size: Tuple[int, int]
    ) -> None:
        self._canvas.create_oval(
            (
                coords[0] - brush_size[0],
                coords[1] - brush_size[1],
                coords[0] + brush_size[0],
                coords[1] + brush_size[1],
            ),
            fill="black",
        )

    def update_clearing(self) -> None:
        self._canvas.delete("all")
        self._renders = [self._create_blank_render() for _ in range(self._TOP_K)]

        for i in range(self._TOP_K):
            self._prediction_components[i].label_render.configure(
                image=self._renders[i]
            )

        for m in self._markups:
            m.set("")

    def update_predictions(self, predictions: List[Prediction]) -> None:
        assert len(predictions) == self._TOP_K

        for i in range(self._TOP_K):
            self._markups[i].set(predictions[i].markup)

            render = predictions[i].render.resize(self._RENDER_DIMS)
            self._renders[i] = ImageTk.PhotoImage(render)

            self._prediction_components[i].label_render.configure(
                image=self._renders[i]
            )
