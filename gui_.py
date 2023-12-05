import random
from enum import Enum, auto
from threading import Thread

import tkinter as tk
from tkinter.simpledialog import askinteger

import numpy as np

from colors import AGENT_COLORS
from environment import MAACEnv
from main import Main

class Grid:
    pass

class GUI(tk.Frame):
    class RunStatus(Enum):
        EDITTING_ENV = auto()
        TRAINING = auto()
        PAUSED = auto()
        TESTING = auto()
    
    _RUN_LITERAL = {
        RunStatus.EDITTING_ENV: '환경 설정',
        RunStatus.TRAINING: '학습 중',
        RunStatus.PAUSED: '일시 정지',
    }
    
    def __init__(self, root, main):
        tk.Frame.__init__(self, root)
        root.title('MAAC')
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor='center')
                
        self.main = main
        self.n_row = main.env.n_row 
        self.n_col = main.env.n_col
        self.grid_ = Grid(self, self.n_row, self.n_col, env=main.env)
        
        self._run_status = GUI.RunStatus.EDITTING_ENV
        
        self._thread = None
        
        self._init_status()
        self._init_menu()
    
    def render(self, episodes=None, steps=None, visual=False, fastest_solve=None,
               visited_layer=None, agents_info=None, reset=False, **kwargs):
        if self.grid_.event_mode != Grid.EventMode.BLOCKED:
            self.grid_.update_(event_mode=Grid.EventMode.BLOCKED)
            
        self.progress_label.config(text=self._get_progress_text(episodes, steps, fastest_solve))
                
        if reset:
            self.grid_.reset()
            self.grid_.render()
            
        if not visual:
            return
        
        for visited in np.argwhere(visited_layer!=-1):
            self.grid_.update_(pos=tuple(visited), 
                             visited=visited_layer[visited[0], visited[1]])
        for idx, agent in agents_info.items():
            self.grid_.update_(agent_idx=idx, agent_new_pos=agent['new_pos'])            

        self.grid_.render()
    
    """ private methods """
    
    def _update_(self, run_status=None):
        if run_status is not None:
            self._run_status = run_status
            self.run_label.config(text=self._get_run_literal())
        self.size_label.config(text=self._get_map_size_text())
        self.event_label.config(text=self._get_event_literal())
        self.progress_label.config(text=self._get_progress_text())
    
    def _init_status(self):
        self.status = tk.Frame(self.master, bd=1, relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.size_label = tk.Label(self.status, text='{}x{}'.format(self.n_row, self.n_col),
                                   padx=5, bd=1, relief=tk.SUNKEN)
        self.size_label.pack(side=tk.LEFT, fill=tk.X)
        
        self.event_label = tk.Label(self.status, text=self._get_event_literal(),
                                   padx=5, bd=1, relief=tk.SUNKEN)
        self.event_label.pack(side=tk.LEFT, fill=tk.X)
        
        self.run_label = tk.Label(self.status, text=self._get_run_literal(),
                                   padx=5, bd=1, relief=tk.SUNKEN)
        self.run_label.pack(side=tk.LEFT, fill=tk.X)
        
        self.progress_label = tk.Label(self.status, text=self._get_progress_text(),
                                       padx=5, bd=1, relief=tk.SUNKEN, anchor='w')
        self.progress_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _init_menu(self):
        menu = tk.Menu(self.master)
        
        map_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='맵 크기', menu=map_menu)
        map_menu.add_command(label='5x5', 
                             command=lambda: self._change_map_size(5, 5))
        map_menu.add_command(label='10x10', 
                             command=lambda: self._change_map_size(10, 10))
        map_menu.add_command(label='15x15', 
                             command=lambda: self._change_map_size(15, 15))
        map_menu.add_separator()
        map_menu.add_command(label='직접 입력', 
                             command=lambda: self._change_map_size(
                                 **self._input_map_size()))
        
        env_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='설정 모드', menu=env_menu)
        env_menu.add_command(label='장애물/벽 위치 설정', 
                             command=lambda: 
                                 self.grid_.update_(event_mode=Grid.EventMode.EDIT_OBSTACLE))
        env_menu.add_command(label='청소할 구역 설정',
                             command=lambda: 
                                 self.grid_.update_(event_mode=Grid.EventMode.EDIT_DIRTY))
        env_menu.add_command(label='청소기 배치',
                             command=lambda: 
                                 self.grid_.update_(event_mode=Grid.EventMode.EDIT_AGENT))
        
        reset_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='초기화', menu=reset_menu)
        reset_menu.add_command(label='장애물/벽 초기화',
                               command=lambda: self._clear_obstacles())
        reset_menu.add_command(label='청소할 구역 초기화',
                               command=lambda: self._clear_dirties())
        reset_menu.add_command(label='청소기 배치 초기화',
                             command=lambda: self._clear_agents())
        
        menu.add_command(label='학습 시작', command=self._start_learn)
        menu.add_command(label='테스트 시작', command=self._start_test)
        menu.add_command(label='실행 중단', command=self._stop_task)
        menu.add_command(label='학습 정보 삭제', command=self._reset_learning_data)
        
        self.master.config(menu=menu)

    def _change_map_size(self, n_row, n_col):
        self.grid_.update_(n_row=n_row, n_col=n_col)
        self.grid_.render()
        self._update_()
    
    def _input_map_size(self):
        row_num = askinteger('', '공간의 높이(행)를 입력하세요. ({} ~ {})'.format(Grid.MIN_MAP, Grid.MAX_MAP))
        col_num = askinteger('', '공간의 넓이(열)를 입력하세요. ({} ~ {})'.format(Grid.MIN_MAP, Grid.MAX_MAP))
        return {'n_row': row_num, 'n_col': col_num}
    
    def _clear_obstacles(self):
        for pos in self.grid_.get_obstacle_pos():
            self.grid_.update_(pos=pos, obstacle=False)
        self.grid_.render()
    
    def _clear_dirties(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                pos = (row, col)
                if not self.grid_._pos_to_cell[pos].dirty:
                    self.grid_.update_(pos=pos, dirty=True)
        self.grid_.render()
    
    def _clear_agents(self):
        for idx, _ in self.grid_.get_idx_agent_pairs():
            self.grid_.update_(agent_idx=idx, agent_new_pos=None)
        self.grid_.render()

    def _start_task(self):
        if self._thread is not None:
            return
        
        self.main.env = self.grid_.export_as_env()
        self.main.env.render_callback = self.render
        self.main.env.export()
        self.main.prepare()
            
        self.main.evaluate = False
        self._thread = Thread(target=self.main.run, daemon=True)
        self._thread.start()
    
    def _start_learn(self):
        self._run_status = GUI.RunStatus.TRAINING
        self.main.evaluate = False
        self._start_task()
    
    def _start_test(self):
        self._run_status = GUI.RunStatus.TESTING
        self.main.evaluate = True
        self._start_task()
    
    def _stop_task(self):
        if self._thread is None:
            return
        self._run_status = GUI.RunStatus.PAUSED
        
        self.main.force_stop = True
        self._thread = None
    
    def _reset_learning_data(self):
        self._stop_task()
        
        self.main.env = self.grid_.export_as_env()
        self.main.env.render_callback = self.render
        self.main.env.export()
        self.main.prepare()
        
        self.main.save_checkpoint()
    
    def _get_map_size_text(self):
        return '{}x{}'.format(self.n_row, self.n_col)
    
    def _get_progress_text(self, episode=None, step=None, fast_solve=None):
        if episode is None:
            return '학습 준비 중'
        text = '에피소드: {}'.format(episode)
        if step is not None:
            text += ' | 스텝: {}'.format(step)
        if fast_solve is not None:
            text += ' | 최단 에피소드: {} 스텝'.format(fast_solve)
        return text
    
    def _get_event_literal(self):
        return Grid._MODE_LITERAL[self.grid_.event_mode]
    
    def _get_run_literal(self):
        return GUI._RUN_LITERAL[self._run_status]
    
# TODO: implement auto save / load environment
class Grid(tk.Frame):
    CELL_SIZE = 50
    MIN_MAP = 5
    MAX_MAP = 20
    
    class EventMode(Enum):
        EDIT_OBSTACLE = auto()
        EDIT_DIRTY = auto()
        EDIT_AGENT = auto()
        BLOCKED = auto()
    
    _CURSOR = {
        EventMode.EDIT_OBSTACLE: 'crosshair', 
        EventMode.EDIT_DIRTY: 'crosshair', 
        EventMode.EDIT_AGENT: 'hand2', 
        EventMode.BLOCKED: 'X_cursor'
    }
    
    _MODE_LITERAL = {
        EventMode.EDIT_OBSTACLE: '장애물 및 벽 설정', 
        EventMode.EDIT_DIRTY: '청소 구역 설정', 
        EventMode.EDIT_AGENT: '청소기 배치', 
        EventMode.BLOCKED: '실행 중'
    }
    
    def __init__(self, master, n_row, n_col, env=None):
        tk.Frame.__init__(self, master)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor='center', 
                          padx=5, pady=5)
        
        self.n_row = min(max(Grid.MIN_MAP, n_row), Grid.MAX_MAP)
        self.n_col = min(max(Grid.MIN_MAP, n_col), Grid.MAX_MAP)

        self.event_mode = Grid.EventMode.EDIT_OBSTACLE
        
        self._canvas = tk.Canvas(self, bg='white', cursor=self._get_cursor(), 
                                 width=self._get_canvas_width(), 
                                 height=self._get_canvas_height())
        self._canvas.pack(side=tk.TOP)
        self._canvas.bind('<Button-1>', self._on_mouse_down)
        self._canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self._canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        
        self._drag = [] # [(start_row, start_col), (dest_row, dest_col)]
        self._drag_indicator = None
        
        self._render_size = (n_row, n_col)
        self._pos_to_cell = {}  # {(row, col): Cell()}
        self._rect_to_cell = {} # {rect_id: Cell()}
        self._pos_to_agent = {} # {(row, col): ColoredAgent()}
        self._idx_to_agent = {} # {idx: ColoredAgent()}
        
        if env is not None:
            self.update_(n_row=env.n_row, n_col=env.n_col)
            for pos in env.obstacle_pos:
                self.update_(pos=tuple(pos), obstacle=True)
            for pos in env.dirty_pos:
                self.update_(pos=tuple(pos), dirty=True)
            for idx, pos in enumerate(env.agent_pos):
                agent = ColoredAgent(self._canvas, *tuple(pos))
                self._pos_to_agent[tuple(pos)] = agent
                self._idx_to_agent[idx] = agent
        
        self.render()
        
    def reset(self):
        for cell in self._pos_to_cell.values():
            cell.reset()
        for agent in self._pos_to_agent.values():
            agent.reset()
    
    def render(self):
        if self._render_size != (self.n_row, self.n_col):
            self._canvas.config(width=self._get_canvas_width(), 
                                height=self._get_canvas_height())
            self._render_size = (self.n_row, self.n_col)\
        
        for row in range(self.n_row):
            for col in range(self.n_col):
                pos = (row, col)
                cell = self._pos_to_cell.get(pos, None)
                if cell is None:
                    cell = Cell(self._canvas, row, col)
                    self._pos_to_cell[pos] = cell
                if not cell.updated:
                    pass
                cell.render()
                if cell.rect not in self._rect_to_cell:
                    self._rect_to_cell[cell.rect] = cell
                if pos in self._pos_to_agent:
                    self._pos_to_agent[pos].render()
    
    def update_(self, event_mode=None,
               n_row=None, n_col=None, 
               pos=None, obstacle=None, dirty=None, visited=None, 
               agent_idx=None, agent_new_pos=None):
        if event_mode is not None:
            self.event_mode = event_mode
            self._canvas.config(cursor=self._get_cursor())
            return
        
        if n_row is not None:
            self.n_row = min(max(Grid.MIN_MAP, n_row), Grid.MAX_MAP)
        if n_col is not None:
            self.n_col = min(max(Grid.MIN_MAP, n_col), Grid.MAX_MAP)
        
        if n_row is not None or n_col is not None:
            for row in range(self.n_row):
                for col in range(self.n_col):
                    _pos = (row, col)
                    if _pos not in self._pos_to_cell:
                        cell = Cell(self._canvas, *_pos)
                        self._pos_to_cell[_pos] = cell

        if pos is not None:
            if obstacle is not None:
                self._pos_to_cell[pos].update_(obstacle=obstacle)
            if dirty is not None:
                self._pos_to_cell[pos].update_(dirty=dirty)
            if visited is not None:
                self._pos_to_cell[pos].update_(visited=self._idx_to_agent[visited])
            if agent_new_pos is not None:
                self._pos_to_agent[pos].update_(new_pos=agent_new_pos)
                self._pos_to_agent[agent_new_pos] = self._pos_to_agent.pop(pos)

        if agent_idx is not None:
            if agent_new_pos is None:
                agent = self._idx_to_agent.pop(agent_idx)
                agent.render(erase=True)
                self._pos_to_agent.pop((agent.row, agent.col))
            else:
                self._idx_to_agent[agent_idx].update_(new_pos=agent_new_pos)
        
        # self.render()
        
    def get_obstacle_pos(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                pos = (row, col)
                if self._pos_to_cell[pos].obstacle:
                    yield pos
    
    def get_dirty_pos(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                pos = (row, col)
                if self._pos_to_cell[pos].dirty:
                    yield pos
    
    def get_idx_agent_pairs(self):
        pairs = []
        for i, agent in self._idx_to_agent.items():
            if agent.row < self.n_row and agent.col < self.n_col:            
                pairs.append((i, (agent.row, agent.col)))
        return pairs
    
    def export_as_env(self):
        obstacle_pos = list(map(tuple, self.get_obstacle_pos()))
        dirty_pos = list(map(tuple, self.get_dirty_pos()))
        agent_pos = [None for _ in range(len(self._idx_to_agent))]
        for idx, agent in self._idx_to_agent.items():
            agent_pos[idx] = (agent.row, agent.col)
        
        return MAACEnv(len(agent_pos), self.n_row, self.n_col, 
                       agent_pos=agent_pos, obstacle_pos=obstacle_pos, dirty_pos=dirty_pos)

    """ private methods """
    
    def _on_mouse_down(self, event):
        if self.event_mode == Grid.EventMode.BLOCKED:
            return
        self._drag = [(event.x, event.y), (event.x, event.y)]
    
    def _on_mouse_drag(self, event):
        if self.event_mode == Grid.EventMode.BLOCKED:
            return
        if not self._drag:
            return
        
        self._drag[-1] = (event.x, event.y)
        
        # visualize indicator of dragging event
        if self._drag_indicator is None:
            if self.event_mode == Grid.EventMode.EDIT_AGENT:
                self._drag_indicator = self._canvas.create_line(
                    *self._drag[0], *self._drag[1], width=2, fill='gray')
            else:
                self._drag_indicator = self._canvas.create_rectangle(
                    *self._drag[0], *self._drag[1], width=2, outline='gray')
        self._canvas.coords(self._drag_indicator, *self._drag[0], *self._drag[1])    
    
    def _on_mouse_up(self, event):
        if self.event_mode == Grid.EventMode.BLOCKED:
            return
        if not self._drag:
            return
        
        # do something according to event_mode
        if self.event_mode == Grid.EventMode.EDIT_AGENT:
            start_shapes = self._canvas.find_overlapping(*self._drag[0], *self._drag[0])
            dest_shapes = self._canvas.find_overlapping(*self._drag[1], *self._drag[1])
            
            start_cell = None
            dest_cell = None
            
            for shape in start_shapes:
                start_cell = self._get_cell_by_rect(shape)
                if start_cell is not None:
                    break
            for shape in dest_shapes:
                dest_cell = self._get_cell_by_rect(shape)
                if dest_cell is not None:
                    break
                
            if start_cell is not None and dest_cell is not None and not dest_cell.obstacle:
                pos = (start_cell.row, start_cell.col)
                new_pos = (dest_cell.row, dest_cell.col)
                updated = False
                
                if pos == new_pos:
                    updated = True
                    if pos in self._pos_to_agent:
                        self._pos_to_agent.pop(pos).render(erase=True)
                    else:
                        agent = ColoredAgent(self._canvas, *pos)
                        self._pos_to_agent[pos] = agent
                elif pos in self._pos_to_agent and new_pos not in self._pos_to_agent:
                    updated = True
                    self._pos_to_agent[pos].update_(new_pos=new_pos)
                    self._pos_to_agent[new_pos] = self._pos_to_agent.pop(pos)

                if updated:
                    self._idx_to_agent = {i: agent 
                                            for i, agent 
                                            in enumerate(self._pos_to_agent.values())}
        else:
            shapes = self._canvas.find_overlapping(*self._drag[0], *self._drag[1])
            for shape in shapes:
                cell = self._get_cell_by_rect(shape)
                if cell is None:
                    continue
                if self.event_mode == Grid.EventMode.EDIT_OBSTACLE \
                    and (cell.row, cell.col) not in self._pos_to_agent:
                    cell.update_(obstacle=not cell.obstacle)
                elif self.event_mode == Grid.EventMode.EDIT_DIRTY:
                    cell.update_(dirty=not cell.dirty)
                    
        self.render()
        
        # remove indicator
        self._drag.clear()
        self._canvas.delete(self._drag_indicator)
        self._drag_indicator = None
    
    def _get_cell_by_pos(self, pos):
        return self._pos_to_cell.get(pos, None)
    
    def _get_cell_by_rect(self, rect):
        cell = self._rect_to_cell.get(rect, None)
        if cell is None:
            for _cell in self._pos_to_cell.values():
                if _cell.rect == rect:
                    self._rect_to_cell[rect] = _cell
                    cell = _cell
                    break
        return cell
    
    def _get_agent_by_pos(self, pos):
        return self._pos_to_agent[pos]

    def _get_agent_by_idx(self, idx):
        return self._idx_to_agent[idx]

    def _get_cursor(self):
        return Grid._CURSOR[self.event_mode]
 
    def _get_canvas_height(self):
        return self.n_row * Grid.CELL_SIZE - 1

    def _get_canvas_width(self):
        return self.n_col * Grid.CELL_SIZE - 1

""" Rule: Each visual component never render itself. """

class ColoredAgent:
    def __init__(self, canvas, row, col):
        self.color = random.choice(AGENT_COLORS)
        self.canvas = canvas
        self.row = row
        self.col = col
        
        self._original_pos = (row, col)
        self._oval = None
    
    def reset(self):
        self.row, self.col = self._original_pos
    
    def render(self, erase=False):
        if erase:
            self.canvas.delete(self._oval)
            self._oval = None
            return
        
        left = self.col * Grid.CELL_SIZE + 2 + 4
        top = self.row * Grid.CELL_SIZE + 2 + 4
        right = (self.col + 1) * Grid.CELL_SIZE - 4
        bottom = (self.row + 1) * Grid.CELL_SIZE - 4
        if self._oval is not None:
            self.canvas.delete(self._oval)
        self._oval = self.canvas.create_oval(left, top, right, bottom, fill=self.color, width=2)
    
    def update_(self, new_pos=None):
        if new_pos is not None:
            self.row, self.col = new_pos
    
class Cell:
    EMPTY_COLOR = 'white'
    DIRTY_COLOR = 'gray90'
    OBSTACLE_COLOR = 'black'
    
    def __init__(self, canvas, row, col):
        self.canvas = canvas
        self.row = row
        self.col = col
        
        self.obstacle = False
        self.dirty = True   # defalut is True
        self.visited = None
        self.rect = None
        
        self.updated = True
        
        self._fill = Cell.DIRTY_COLOR   # defalut is DIRTY_COLOR
    
    def reset(self):
        self.updated = True
        
        self.visited = None
        
        if self.obstacle:
            self._fill = Cell.OBSTACLE_COLOR
        elif self.dirty:
            self._fill = Cell.DIRTY_COLOR
        else:
            self._fill = Cell.EMPTY_COLOR
    
    def render(self, erase=False):
        self.updated = False
        
        if erase:
            self.canvas.delete(self.rect)
            self.rect = None
            return
        
        if self.rect is None:
            left = self.col * Grid.CELL_SIZE + 2
            top = self.row * Grid.CELL_SIZE + 2
            right = (self.col + 1) * Grid.CELL_SIZE
            bottom = (self.row + 1) * Grid.CELL_SIZE
            self.rect = self.canvas.create_rectangle(left, top, right, bottom, fill=self._fill)
        else:
            self.canvas.itemconfig(self.rect, fill=self._fill)
    
    def update_(self, obstacle=None, dirty=None, visited: ColoredAgent=None):
        self.updated = True
        
        if obstacle is not None:
            self.obstacle = obstacle
            self.visited = None
        if dirty is not None:
            self.dirty = dirty
        if visited is not None and not self.obstacle:
            self.visited = visited
        if self.obstacle:
            self._fill = Cell.OBSTACLE_COLOR
        elif self.visited:
            self._fill = self.visited.color
        elif self.dirty:
            self._fill = Cell.DIRTY_COLOR
        else:
            self._fill = Cell.EMPTY_COLOR


if __name__ == '__main__':
    root = tk.Tk()
    main = Main()
    gui = GUI(root, main)
    gui.mainloop()
    