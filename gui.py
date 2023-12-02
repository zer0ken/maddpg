from tkinter import Frame, Tk, Canvas, N, S, E, W, Menu, X, Label, BOTTOM, W
from tkinter import ttk
from tkinter.simpledialog import askinteger

import random

from colors import AGENT_COLORS

class GUI(Frame):
    OBSTACLE = 0
    DIRTY = 1
    AGENT = 2
    LEARNING = 3
    
    LITERAL = {
        OBSTACLE: '장애물/벽 위치 설정',
        DIRTY: '청소할 구역 설정',
        AGENT: '청소기 배치',
    }
    
    CELL_SIZE = 50
    
    def __init__(self, root, row_num=10, col_num=10):
        Frame.__init__(self, root)
        root.title("MAAC")
        root.resizable(False, False)
        self.grid(row=0,column=0)
        
        self.rect_to_cell = {}
        self.cells = {}
        self.agents = {}
        
        self.n_row = row_num
        self.n_col = col_num
        
        self.canvas_width = GUI.CELL_SIZE * self.n_col - 1
        self.canvas_height = GUI.CELL_SIZE * self.n_row - 1
        
        self.dragging = []
        self.dragging_rect = None
        self.dragging_line = None
        
        self.gui_mode = GUI.OBSTACLE
        
        self.map_btn = None
        self.obstacle_btn = None
        self.dirty_btn = None
        self.agent_btn = None
        self.map_input = None
        self.init_menu()
        
        self.canvas = None
        self.init_canvas()
        
        self.statusbar = Frame(root, bd=1, relief='sunken')
        self.statusbar.grid(row=1, column=0, sticky=(S, E, W))
        
        self.map_status = Label(self.statusbar, padx=5, bd=1, relief='sunken', anchor=W, 
                                text='{}×{}'.format(self.n_row, self.n_col))
        self.map_status.pack(side='left')
        
        self.env_status = Label(self.statusbar, padx=5, bd=1, relief='sunken', anchor=W,
                                text=GUI.LITERAL[self.gui_mode])
        self.env_status.pack(side='left', fill=X, expand=True)

    def init_menu(self):
        menu = Menu(self.master)
        
        map_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='공간', menu=map_menu)
        map_menu.add_command(label='5x5', command=lambda: self.set_map_size(5,5))
        map_menu.add_command(label='10x10', command=lambda: self.set_map_size(10,10))
        map_menu.add_command(label='15x15', command=lambda: self.set_map_size(15,15))
        map_menu.add_separator()
        map_menu.add_command(label='직접 입력', command=self.set_map_size)
        
        env_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='설정 모드', menu=env_menu)
        env_menu.add_command(label='장애물/벽 위치 설정', 
                             command=lambda: self.set_click_mode(GUI.OBSTACLE))
        env_menu.add_command(label='청소할 구역 설정',
                             command=lambda: self.set_click_mode(GUI.DIRTY))
        env_menu.add_command(label='청소기 배치',
                             command=lambda: self.set_click_mode(GUI.AGENT))
        
        reset_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='초기화', menu=reset_menu)
        reset_menu.add_command(label='장애물/벽 초기화',
                               command=lambda: self.remove_all(GUI.OBSTACLE))
        reset_menu.add_command(label='청소할 구역 초기화',
                               command=lambda: self.remove_all(GUI.DIRTY))
        reset_menu.add_command(label='청소기 배치 초기화',
                             command=lambda: self.remove_all(GUI.AGENT))
        
        model_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='학습', menu=model_menu)
        model_menu.add_command(label='학습 시작',
                               command=self.export_env)
        model_menu.add_command(label='학습 중단')
        model_menu.add_separator()
        model_menu.add_command(label='학습 초기화')
        
        self.master.config(menu=menu)
    
    def set_click_mode(self, mode):
        self.gui_mode = mode
        self.env_status.config(text=GUI.LITERAL[self.gui_mode])
        if mode == GUI.AGENT:
            self.canvas.config(cursor='arrow')
        else:
            self.canvas.config(cursor='crosshair')
    
    def remove_all(self, target):
        if target == GUI.AGENT:
            for agent in self.agents.values():
                agent.erase()
            self.agents.clear()
            return
        for cell in self.cells.values():
            if target == GUI.OBSTACLE:
                cell.obstacle = False
            elif target == GUI.DIRTY:
                cell.dirty = False
            cell.draw()
    
    def init_canvas(self):
        if self.canvas is None:
            canvas_frame = ttk.Frame(self, padding=(5,5,5,5))
            canvas_frame.grid(column=0, row=0, sticky=(N, S, E, W))
            canvas = Canvas(canvas_frame, 
                            width=self.canvas_width, height=self.canvas_height,
                            bg='white', cursor='crosshair')
            canvas.grid(column=0, row=0)
            canvas.bind('<ButtonRelease-1>', self.on_canvas_up)
            canvas.bind('<B1-Motion>', self.on_canvas_drag)
            canvas.bind('<Button-1>', self.on_canvas_down)
        else:
            canvas = self.canvas
            canvas.config(width=self.canvas_width, height=self.canvas_height)

        for row in range(self.n_row):
            for col in range(self.n_col):
                if (row, col) in self.cells:
                    self.cells[(row, col)].draw()
                    continue
                cell = Cell(canvas, row, col)
                self.cells[row, col] = cell
                self.rect_to_cell[cell.rect] = cell
        
        self.canvas = canvas

    def on_canvas_down(self, event):
        if self.gui_mode == GUI.LEARNING:
            return
        self.dragging.append((event.x, event.y))
        self.dragging.append((event.x, event.y))
    
    def on_canvas_drag(self, event):
        if self.gui_mode == GUI.LEARNING:
            return
        
        if self.dragging:
            self.dragging[-1] = (event.x, event.y)
        if self.gui_mode == GUI.AGENT:
            if self.dragging_line is None:
                self.dragging_line = self.canvas.create_line(
                    *self.dragging[0], *self.dragging[1], fill='gray', width=2)
            else:
                self.canvas.coords(self.dragging_line, *self.dragging[0], *self.dragging[1])
        else:
            if self.dragging_rect is None:
                self.dragging_rect = self.canvas.create_rectangle(
                    *self.dragging[0], *self.dragging[1], outline='gray', width=2)
            else:
                self.canvas.coords(self.dragging_rect, *self.dragging[0], *self.dragging[1])

    def on_canvas_up(self, event):
        if self.gui_mode == GUI.LEARNING:
            return
        
        if not self.dragging:
            return
        
        if self.dragging_rect is not None:
            self.canvas.delete(self.dragging_rect)
            self.dragging_rect = None
        if self.dragging_line is not None:
            self.canvas.delete(self.dragging_line)
            self.dragging_line = None
            
        if self.gui_mode == GUI.AGENT:
            down_shapes = self.canvas.find_overlapping(*self.dragging[0], *self.dragging[0])
            up_shapes = self.canvas.find_overlapping(*self.dragging[1], *self.dragging[1])
            
            down_cell = None 
            for shape in down_shapes:
                if shape in self.rect_to_cell:
                    down_cell = self.rect_to_cell[shape]
                    break
                
            up_cell = None
            for shape in up_shapes:
                if shape in self.rect_to_cell:
                    up_shapes = shape
                    up_cell = self.rect_to_cell[shape]
                    break

            if down_cell is None or up_cell is None:
                self.dragging.clear()
                return
            if up_cell.obstacle:
                self.dragging.clear()
                return
            
            down_pos = (down_cell.row, down_cell.col)
            up_pos = (up_cell.row, up_cell.col)
            
            if down_pos == up_pos:
                if up_pos not in self.agents:
                    self.agents[up_pos] = ColoredAgent(self.canvas, *up_pos)
                    print('new agent', up_pos)
                else:
                    self.agents[up_pos].erase()
                    del self.agents[up_pos]
                    print('remove agent', up_pos)
                    
            elif down_pos in self.agents and down_pos != up_pos and up_pos not in self.agents:
                self.move_agent(down_pos, up_pos)
        else:        
            rects = self.canvas.find_overlapping(*self.dragging[0], *self.dragging[1])
            for rect in rects:
                if rect not in self.rect_to_cell:
                    continue
                cell = self.rect_to_cell[rect]
                cell.onclick({
                    'mode': self.gui_mode,
                })
                
                cell_pos = (cell.row, cell.col)
                if cell_pos in self.agents and self.gui_mode == GUI.OBSTACLE:
                    self.agents[cell_pos].erase()
                    del self.agents[cell_pos]
        self.dragging.clear()

    def move_agent(self, from_, to):
        selected = self.agents[from_]
        del self.agents[from_]
        selected.row = to[0]
        selected.col = to[1]
        self.agents[to] = selected
        selected.draw()
        print('move agent', from_, 'to', to)
            
    def set_map_size(self, row_num=None, col_num=None):
        if row_num is None:
            row_num = askinteger('', '공간의 행 개수를 입력하세요.')
        if col_num is None:
            col_num = askinteger('', '공간의 열 개수를 입력하세요.')
        if row_num is None or col_num is None:
            return
        self.n_row = min(max(2, row_num), 20)
        self.n_col = min(max(2, col_num), 20)
        
        self.canvas_width = self.CELL_SIZE * self.n_col - 1
        self.canvas_height = self.CELL_SIZE * self.n_row - 1
        
        self.init_canvas()
        self.map_status.config(text='{}×{}'.format(self.n_row, self.n_col))
    
    def export_env(self):
        obstacles = []
        dirties = []
        agents = []
        for pos, cell in self.cells.items():
            if pos[0] >= self.n_row or pos[1] >= self.n_col:
                continue
            if cell.obstacle:
                obstacles.append((cell.row, cell.col))
            if cell.dirty:
                dirties.append((cell.row, cell.col))
        for pos in self.agents.keys():
            if pos[0] >= self.n_row or pos[1] >= self.n_col:
                continue
            agents.append(pos)
        
        print('@@@ Export environment')
        print('* obstacles:', obstacles)
        print('* dirties:', dirties)
        print('* agents:', agents)
    
    def render(self, agents, visited):
        for row in range(self.n_row):
            for col in range(self.n_col):
                if visited[row, col] != -1:
                    cell = self.cells[row, col]
                    color = self.agents[tuple(agents[visited[row, col]]['pos'])].color
                    cell.visited = True
                    cell.visited_color = color
                    cell.draw()
        for agent in agents:
            self.move_agent(agent['pos'], agent['new_pos'])
        pass
    

class Cell:
    def __init__(self, canvas, row, col):
        self.canvas = canvas
        self.row = row
        self.col = col
        self.dirty = False
        self.obstacle = False
        self.visited = False
        
        self.rect = None
        self.draw()
    
    def draw(self):
        if self.rect is None:
            left = self.col * GUI.CELL_SIZE + 2
            top = self.row * GUI.CELL_SIZE + 2
            right = (self.col + 1) * GUI.CELL_SIZE
            bottom = (self.row + 1) * GUI.CELL_SIZE
            self.rect = self.canvas.create_rectangle(left,top,right,bottom,outline='gray',fill='')
        
        color = 'white'
        if self.obstacle:
            color = 'black'
        elif self.visited:
            color = self.visited_color
        elif self.dirty:
            color ='gray80'
        self.canvas.itemconfig(self.rect, fill=color)

    def onclick(self, event):
        if event['mode'] == GUI.OBSTACLE:
            self.obstacle = not self.obstacle
        if event['mode'] == GUI.DIRTY:
            if self.obstacle:
                return
            self.dirty = not self.dirty
        self.draw()


class ColoredAgent:
    def __init__(self, canvas, row, col):
        self.color = random.choice(AGENT_COLORS)
        self.canvas = canvas
        self.row = row
        self.col = col
        
        self.oval = None
        self.draw()
    
    def erase(self):
        if self.oval is not None:
            self.canvas.delete(self.oval)
    
    def draw(self):
        left = self.col * GUI.CELL_SIZE + 2 + 4
        top = self.row * GUI.CELL_SIZE + 2 + 4
        right = (self.col + 1) * GUI.CELL_SIZE - 4
        bottom = (self.row + 1) * GUI.CELL_SIZE - 4
        if self.oval is None:
            self.oval = self.canvas.create_oval(left, top, right, bottom, fill=self.color, outline='black', width=1)
        else:
            self.canvas.coords(self.oval, left, top, right, bottom) 


if __name__ == '__main__':
    tk = Tk()
    gui = GUI(tk)
    tk.mainloop()