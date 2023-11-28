from tkinter import Frame, Tk, Canvas, N, S, E, W, Menu, Menubutton
from tkinter import ttk

class GUI(Frame):
    def __init__(self, root, row_num=10, col_num=10):
        Frame.__init__(self, root)
        root.title("MADDPG - Area Coverage")
        root.resizable(False, False)
        self.grid(row=0,column=0)
        
        self.rect_to_cell = {}
        self.cells = {}
        self.row_num = row_num
        self.col_num = col_num
        
        self.cell_size = 50
        self.canvas_width = self.cell_size * self.col_num - 1
        self.canvas_height = self.cell_size * self.row_num - 1
        
        self.map_btn = None
        self.obstacle_btn = None
        self.dirty_btn = None
        self.agent_btn = None
        self.init_menu()
        
        self.canvas = None
        self.init_canvas()

    def init_menu(self):
        # interface_frame = ttk.Frame(self, padding=(5,5,5,0))
        # interface_frame.grid(column=0, row=0, sticky=(N, S, E, W))
        menu = Menu(self.master)
        
        map_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='공간 설정', menu=map_menu)
        map_menu.add_command(label='5x5', command=lambda: self.set_map_size(5,5))
        map_menu.add_command(label='10x10', command=lambda: self.set_map_size(10,10))
        map_menu.add_command(label='15x15', command=lambda: self.set_map_size(15,15))
        map_menu.add_separator()
        map_menu.add_command(label='직접 입력')
        
        env_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='환경 설정', menu=env_menu)
        env_menu.add_command(label='장애물/벽 위치 설정')
        env_menu.add_command(label='청소할 구역 설정')
        env_menu.add_command(label='청소기 배치')
        env_menu.add_separator()
        env_menu.add_command(label='초기화')
        
        model_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='학습', menu=model_menu)
        model_menu.add_command(label='학습 시작')
        model_menu.add_command(label='학습 중단')
        
        self.master.config(menu=menu)
    
    def init_canvas(self):
        if self.canvas is None:
            canvas_frame = ttk.Frame(self, padding=(5,5,5,5))
            canvas_frame.grid(column=0, row=0, sticky=(N, S, E, W))
            canvas = Canvas(canvas_frame, 
                            width=self.canvas_width, height=self.canvas_height,
                            bg='white')
            canvas.grid(column=0, row=0)
            canvas.bind('<Button-1>', self.on_canvas_click)
        else:
            canvas = self.canvas
            canvas.config(width=self.canvas_width, height=self.canvas_height)

        for row in range(self.row_num):
            for col in range(self.col_num):
                if (row, col) in self.cells:
                    continue
                top = row * self.cell_size + 2
                left = col * self.cell_size + 2
                bottom = row * self.cell_size + self.cell_size
                right = col * self.cell_size + self.cell_size
                rect = canvas.create_rectangle(left,top,right,bottom,outline='gray',fill='')
                cell = Cell(canvas, rect, row, col)
                self.cells[row, col] = cell
                self.rect_to_cell[rect] = cell
        
        self.canvas = canvas

    def on_canvas_click(self, event):
        rect = self.canvas.find_closest(event.x, event.y)
        self.rect_to_cell[rect[0]].onclick({
            'event_test': 'We can pass any data from GUI to Cell',
        })
            
    def set_map_size(self, row_num, col_num):
        self.row_num = row_num
        self.col_num = col_num
        self.canvas_width = self.cell_size * self.col_num - 1
        self.canvas_height = self.cell_size * self.row_num - 1
        
        self.init_canvas()
        
        
class Cell:
    def __init__(self, canvas, rect, row, col):
        self.canvas = canvas
        self.rect = rect
        self.row = row
        self.col = col

    def onclick(self, event):
        print('clicked', self.row, self.col)
        print(event)


if __name__ == '__main__':
    tk = Tk()
    gui = GUI(tk)
    tk.mainloop()