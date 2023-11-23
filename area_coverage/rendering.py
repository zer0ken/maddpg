import tkinter as tk
from enum import Enum

import numpy as np

class Color:
    BLACK = "#000000"
    WHITE = "#FFFFFF"

class Size:
    WIDTH = 800
    HEIGHT = 800
    INTERFACE_HEIGHT = 100
    PADDING = 100


class Renderer(tk.Tk):
    def __init__(self, n_row=10, n_col=10, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, width=Size.WIDTH, height=Size.HEIGHT)
        self.canvas.pack()
        
        self.n_row = n_row
        self.n_col = n_col
        
        self.rect = {}
        
        self.render()
        
    def render(self): #agents, obstacles, dirty_cells, visited_cells
        cell_size = min(40,
                        (Size.WIDTH - 2*Size.WIDTH) // self.n_col,
                        (Size.HEIGHT - 2*Size.PADDING) // self.n_row)
        x_start = (Size.WIDTH - cell_size*self.n_col) // 2 + Size.PADDING
        y_start = (Size.HEIGHT - cell_size*self.n_row) // 2 + Size.PADDING
        
        for row in range(self.n_row):
            for col in range(self.n_col):
                self.rect[row, col] = self.canvas.create_rectangle(
                    x_start + col*cell_size,
                    y_start + row*cell_size,
                    x_start + (col+1)*cell_size,
                    y_start + (row+1)*cell_size,
                    fill=Color.WHITE,
                    outline=Color.BLACK
                )
        


renderer = Renderer(n_row=2, n_col=3)
renderer.mainloop()