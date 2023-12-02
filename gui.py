from threading import Thread
from tkinter import Frame, Canvas, N, S, E, W, Menu, X, Label, W
from tkinter import ttk
from tkinter.simpledialog import askinteger

import random

import numpy as np

from colors import AGENT_COLORS
from environment import MAACEnv

class GUI(Frame):
    OBSTACLE = 0
    DIRTY = 1
    AGENT = 2
    FIXED = 3
    
    LITERAL = {
        OBSTACLE: '장애물/벽 위치 설정',
        DIRTY: '청소할 구역 설정',
        AGENT: '청소기 배치',
    }
    
    CELL_SIZE = 50
    
    def __init__(self, root, main, row_num=10, col_num=10):
        Frame.__init__(self, root)
        root.title("MAAC")
        root.resizable(False, False)
        
        self.main = main
        self.running_thread = None
        self.exported_env = False
        
        self.grid(row=0,column=0)
        
        self.rect_to_cell = {}
        self.pos_to_cell = {}
        self.pos_to_agent = {}
        self.idx_to_agent = {}
        
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
        
        self.statusbar = Frame(root, bd=1, relief='sunken')
        self.statusbar.grid(row=1, column=0, sticky=(S, E, W))
        
        self.map_status = Label(self.statusbar, padx=5, bd=1, relief='sunken', anchor=W, 
                                text='{}×{}'.format(self.n_row, self.n_col))
        self.map_status.pack(side='left')
        
        self.env_status = Label(self.statusbar, padx=5, bd=1, relief='sunken', anchor=W,
                                text=GUI.LITERAL[self.gui_mode])
        self.env_status.pack(side='left', fill=X, expand=True)

        self.canvas = None
        self.init_with_random_env()

    def init_with_random_env(self):
        env = MAACEnv()
        
        self.n_row = env.n_row
        self.n_col = env.n_col
        
        self.canvas_width = GUI.CELL_SIZE * self.n_col - 1
        self.canvas_height = GUI.CELL_SIZE * self.n_row - 1
        
        self.set_map_size(self.n_row, self.n_col)
        
        for pos in env.obstacle_pos:
            self.pos_to_cell[pos[0], pos[1]].obstacle = True
            self.pos_to_cell[pos[0], pos[1]].draw()
        
        for pos in env.dirty_pos:
            self.pos_to_cell[pos[0], pos[1]].dirty = True
            self.pos_to_cell[pos[0], pos[1]].draw()
        
        for pos in env.agent_pos:
            self.add_agent(tuple(pos))

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
                               command=self.start_learn)
        model_menu.add_command(label='학습 중단',
                               command=self.stop_learn)
        model_menu.add_separator()
        model_menu.add_command(label='학습 초기화')
        
        test_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='테스트', menu=test_menu)
        test_menu.add_command(label='테스트 시작')
        test_menu.add_command(label='테스트 중단')
        
        self.master.config(menu=menu)

    def set_click_mode(self, mode):
        self.gui_mode = mode
        self.env_status.config(text=GUI.LITERAL[self.gui_mode])
        if mode == GUI.AGENT:
            self.canvas.config(cursor='arrow')
        else:
            self.canvas.config(cursor='crosshair')
    
    def remove_all(self, target):
        if self.gui_mode == GUI.FIXED:
            return
        if target == GUI.AGENT:
            for agent in self.pos_to_agent.values():
                agent.erase()
            self.pos_to_agent.clear()
            return
        for cell in self.pos_to_cell.values():
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
                if (row, col) in self.pos_to_cell:
                    self.pos_to_cell[(row, col)].draw()
                    continue
                cell = Cell(canvas, row, col)
                self.pos_to_cell[row, col] = cell
                self.rect_to_cell[cell.rect] = cell
        
        self.canvas = canvas

    def on_canvas_down(self, event):
        if self.gui_mode == GUI.FIXED:
            return
        self.dragging.append((event.x, event.y))
        self.dragging.append((event.x, event.y))
    
    def on_canvas_drag(self, event):
        if self.gui_mode == GUI.FIXED:
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
        if self.gui_mode == GUI.FIXED:
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
                if up_pos not in self.pos_to_agent:
                    self.add_agent(up_pos)
                else:
                    self.remove_agent(up_pos)
                    
            elif down_pos in self.pos_to_agent and down_pos != up_pos and up_pos not in self.pos_to_agent:
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
                if cell_pos in self.pos_to_agent and self.gui_mode == GUI.OBSTACLE:
                    self.remove_agent(cell_pos)
                    
        self.dragging.clear()

    def remove_agent(self, pos):
        self.pos_to_agent[pos].erase()
        del self.pos_to_agent[pos]

    def add_agent(self, pos):
        agent = ColoredAgent(self.canvas, *pos)
        self.pos_to_agent[pos] = agent

    def move_agent(self, from_, to):
        selected = self.pos_to_agent[from_]
        del self.pos_to_agent[from_]
        selected.row = to[0]
        selected.col = to[1]
        self.pos_to_agent[to] = selected
        selected.draw()
    
    def move_agent_along(self, agents_info):
        new_agents = {}
        for i, info in agents_info.items():
            agent = self.idx_to_agent[i]
            to = info.get('new_pos', (agent.row, agent.col))
            new_agents[to] = agent
            new_agents[to].row = to[0]
            new_agents[to].col = to[1]
            new_agents[to].draw()
            print('agent', i, 'moved to', to)
        self.pos_to_agent = new_agents
            
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
    
    def start_learn(self):
        if self.running_thread is not None:
            return
        
        if not self.exported_env:
            obstacle_pos = []
            dirty_pos = []
            agent_pos = []
            for pos, cell in self.pos_to_cell.items():
                if pos[0] >= self.n_row or pos[1] >= self.n_col:
                    continue
                if cell.obstacle:
                    obstacle_pos.append((cell.row, cell.col))
                if cell.dirty:
                    dirty_pos.append((cell.row, cell.col))
            for pos, agent in self.pos_to_agent.items():
                if pos[0] >= self.n_row or pos[1] >= self.n_col:
                    continue
                agent_pos.append(pos)
                self.idx_to_agent[len(self.idx_to_agent)] = agent
            
            env = MAACEnv(n_agent=len(agent_pos), n_row=self.n_row, n_col=self.n_col, 
                        obstacle_pos=obstacle_pos, dirty_pos=dirty_pos, agent_pos=agent_pos)
            env.render_callback = self.render
            
            self.main.env = env
            self.main.prepare()
            self.exported_env = True
            self.gui_mode = GUI.FIXED
            self.exported_env = True
        
        self.running_thread = Thread(target=self.main.run, daemon=True)
        self.running_thread.start()
        
    def stop_learn(self):
        self.main.force_stop = True
        self.running_thread.join()
        self.running_thread = None
        
    def render(self, visited_layer, agents_info):
        print(visited_layer)
        print(agents_info)
        for row, col in np.argwhere(visited_layer == -1):
            cell = self.pos_to_cell[row, col]
            cell.visited = False
            cell.draw()
        for row, col in np.argwhere(visited_layer != -1):
            cell = self.pos_to_cell[row, col]
            agent = agents_info[visited_layer[row, col]]
            pos = agent['pos']
            if pos not in self.pos_to_agent:
                continue
            color = self.pos_to_agent[pos[0], pos[1]].color
            cell.visited = True
            cell.visited_color = color
            cell.draw()
        self.move_agent_along(agents_info)

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
            color ='gray90'
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
            self.oval = self.canvas.create_oval(left, top, right, bottom, fill=self.color, outline='black', width=2)
        else:
            self.canvas.coords(self.oval, left, top, right, bottom) 