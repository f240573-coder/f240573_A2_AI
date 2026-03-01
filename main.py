import pygame
import heapq
import math
import time
import random
import sys
 
BG          = (15,  17,  26)
GRID_LINE   = (30,  35,  55)
EMPTY       = (22,  26,  40)
WALL        = (55,  60,  80)
START_C     = (0,  200, 120)
GOAL_C      = (255,  80,  80)
FRONTIER_C  = (255, 200,  30)
VISITED_C   = (60,  90, 180)
PATH_C      = (0,  230, 140)
TEXT_C      = (210, 215, 235)
PANEL_BG    = (18,  20,  32)
ACCENT      = (80, 160, 255)
BTN_ON      = (60, 130, 220)
BTN_OFF     = (38,  42,  62)
BTN_HOVER   = (50, 110, 190)
RED_BTN     = (180,  50,  50)
GREEN_BTN   = (40, 160,  80)
ORANGE_BTN  = (200, 120,  30)
 
PANEL_W    = 280
MIN_ROWS   = 5
MAX_ROWS   = 40
MIN_COLS   = 5
MAX_COLS   = 60
DEF_ROWS   = 20
DEF_COLS   = 30
DEF_DENSITY= 0.28
FPS        = 60
STEP_DELAY = 0.04
 
pygame.init()
INFO   = pygame.display.Info()
SCR_W  = min(1280, INFO.current_w - 20)
SCR_H  = min(750,  INFO.current_h - 60)
GRID_W = SCR_W - PANEL_W
 
FONT_SM  = pygame.font.SysFont("consolas", 13)
FONT_MD  = pygame.font.SysFont("consolas", 15, bold=True)
FONT_LG  = pygame.font.SysFont("consolas", 19, bold=True)
FONT_TTL = pygame.font.SysFont("consolas", 22, bold=True)
 
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
 
def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
 
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [[0]*cols for _ in range(rows)]
 
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols
 
    def passable(self, r, c):
        return self.in_bounds(r, c) and self.cells[r][c] == 0
 
    def neighbours(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.passable(nr, nc):
                yield (nr, nc)
 
    def randomise(self, density, start, goal):
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c] = 1 if random.random() < density else 0
        sr, sc = start;  self.cells[sr][sc] = 0
        gr, gc = goal;   self.cells[gr][gc] = 0
 
    def toggle(self, r, c, start, goal):
        if (r, c) in (start, goal):
            return
        self.cells[r][c] ^= 1
 
def greedy_bfs(grid, start, goal, heuristic):
    h = heuristic
    open_set = []
    heapq.heappush(open_set, (h(start, goal), start))
    came_from = {start: None}
    visited   = {start}
    frontier  = {start}
    expanded  = set()
    nodes_visited = 0
 
    while open_set:
        _, current = heapq.heappop(open_set)
        frontier.discard(current)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_visited += 1
 
        if current == goal:
            path = reconstruct(came_from, goal)
            yield {"done": True, "path": path, "frontier": frontier.copy(),
                   "expanded": expanded.copy(), "nodes": nodes_visited}
            return
 
        for nb in grid.neighbours(*current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                heapq.heappush(open_set, (h(nb, goal), nb))
                frontier.add(nb)
 
        yield {"done": False, "path": [], "frontier": frontier.copy(),
               "expanded": expanded.copy(), "nodes": nodes_visited}
 
    yield {"done": True, "path": [], "frontier": set(), "expanded": expanded.copy(),
           "nodes": nodes_visited}
 
 
def astar(grid, start, goal, heuristic):
    h = heuristic
    open_set  = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    came_from = {start: None}
    g_score   = {start: 0}
    expanded  = set()
    frontier  = {start}
    nodes_visited = 0
 
    while open_set:
        f, g, current = heapq.heappop(open_set)
        frontier.discard(current)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_visited += 1
 
        if current == goal:
            path = reconstruct(came_from, goal)
            yield {"done": True, "path": path, "frontier": frontier.copy(),
                   "expanded": expanded.copy(), "nodes": nodes_visited,
                   "cost": g_score[goal]}
            return
 
        for nb in grid.neighbours(*current):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb]   = tentative_g
                came_from[nb] = current
                f_new = tentative_g + h(nb, goal)
                heapq.heappush(open_set, (f_new, tentative_g, nb))
                frontier.add(nb)
 
        yield {"done": False, "path": [], "frontier": frontier.copy(),
               "expanded": expanded.copy(), "nodes": nodes_visited, "cost": 0}
 
    yield {"done": True, "path": [], "frontier": set(), "expanded": expanded.copy(),
           "nodes": nodes_visited, "cost": 0}
 
 
def reconstruct(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path
 
def draw_text(surf, text, x, y, font=None, colour=TEXT_C, center=False):
    font = font or FONT_SM
    img  = font.render(text, True, colour)
    rect = img.get_rect()
    if center:
        rect.centerx = x; rect.y = y
    else:
        rect.x = x; rect.y = y
    surf.blit(img, rect)
    return rect
 
def draw_button(surf, rect, label, active=False, hover=False, colour=None):
    c = colour if colour else (BTN_ON if active else (BTN_HOVER if hover else BTN_OFF))
    pygame.draw.rect(surf, c, rect, border_radius=6)
    pygame.draw.rect(surf, ACCENT, rect, 1, border_radius=6)
    draw_text(surf, label, rect.centerx, rect.centery - 8, FONT_SM, TEXT_C, center=True)
 
def pill(surf, rect, colour, label, font=None):
    pygame.draw.rect(surf, colour, rect, border_radius=4)
    draw_text(surf, label, rect.centerx, rect.centery - 7, font or FONT_SM, (20,20,20), center=True)
 
class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCR_W, SCR_H), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent — AI 2002 A2 Q6")
        self.clock  = pygame.time.Clock()
 
        self.rows    = DEF_ROWS
        self.cols    = DEF_COLS
        self.density = DEF_DENSITY
        self.start   = (self.rows//2, 1)
        self.goal    = (self.rows//2, self.cols-2)
        self.grid    = Grid(self.rows, self.cols)
        self.grid.randomise(self.density, self.start, self.goal)
 
        self.algo    = "A*"
        self.heur    = "Manhattan"
 
        self.state   = "idle"
        self.gen     = None
        self.result  = {"path":[], "frontier":set(), "expanded":set(), "nodes":0, "cost":0}
        self.exec_ms = 0
        self.msg     = "Draw walls or click RUN"
        self.dragging_wall = None
        self.placing_start = False
        self.placing_goal  = False
 
        self.dynamic_mode  = False
        self.dyn_agent_idx = 0
        self.dyn_path      = []
        self.dyn_last_step = 0
        self.dyn_interval  = 0.18
        self.spawn_prob    = 0.04
 
        self.input_rows = str(self.rows)
        self.input_cols = str(self.cols)
        self.input_dens = f"{int(self.density*100)}"
        self.active_inp = None
 
        self.btn = {}
        self.mouse_pos = (0,0)
        self._start_time = 0
 
    @property
    def cell(self):
        cw = GRID_W / self.cols
        ch = SCR_H  / self.rows
        return max(4, int(min(cw, ch)))
 
    def px_to_cell(self, x, y):
        c = int(x // self.cell)
        r = int(y // self.cell)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None
 
    @property
    def hfunc(self):
        return manhattan if self.heur == "Manhattan" else euclidean
 
    def reset_search(self):
        self.state  = "idle"
        self.gen    = None
        self.result = {"path":[], "frontier":set(), "expanded":set(), "nodes":0, "cost":0}
        self.dyn_agent_idx = 0
        self.dyn_path      = []
 
    def apply_settings(self):
        try:
            r = max(MIN_ROWS, min(MAX_ROWS, int(self.input_rows)))
            c = max(MIN_COLS, min(MAX_COLS, int(self.input_cols)))
            d = max(0, min(60, int(self.input_dens))) / 100
        except ValueError:
            return
        self.rows    = r
        self.cols    = c
        self.density = d
        self.start   = (r//2, 1)
        self.goal    = (r//2, c-2)
        self.grid    = Grid(r, c)
        self.grid.randomise(d, self.start, self.goal)
        self.reset_search()
        self.msg = "Grid regenerated!"
 
    def run_search(self):
        self.reset_search()
        if self.algo == "A*":
            self.gen = astar(self.grid, self.start, self.goal, self.hfunc)
        else:
            self.gen = greedy_bfs(self.grid, self.start, self.goal, self.hfunc)
        self.state      = "searching"
        self._start_time = time.time()
        self.msg        = f"Running {self.algo} with {self.heur}..."
 
    def run_dynamic(self):
        self.reset_search()
        if self.algo == "A*":
            gen = astar(self.grid, self.start, self.goal, self.hfunc)
        else:
            gen = greedy_bfs(self.grid, self.start, self.goal, self.hfunc)
        state = None
        for state in gen:
            pass
        if state and state["path"]:
            self.dyn_path      = list(state["path"])
            self.dyn_agent_idx = 0
            self.result        = state
            self.state         = "dynamic"
            self.dyn_last_step = time.time()
            self.msg = "Dynamic mode — obstacles may spawn!"
        else:
            self.msg = "No path found for dynamic mode."
 
    def replan(self):
        current = self.dyn_path[self.dyn_agent_idx]
        if self.algo == "A*":
            gen = astar(self.grid, current, self.goal, self.hfunc)
        else:
            gen = greedy_bfs(self.grid, current, self.goal, self.hfunc)
        state = None
        for state in gen:
            pass
        if state and state["path"]:
            self.dyn_path      = list(state["path"])
            self.dyn_agent_idx = 0
            self.result        = state
        else:
            self.state = "done"
            self.msg   = "No path after replanning!"
 
    def run(self):
        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
 
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
 
            if ev.type == pygame.VIDEORESIZE:
                global SCR_W, SCR_H, GRID_W
                SCR_W  = ev.w; SCR_H = ev.h
                GRID_W = SCR_W - PANEL_W
                self.screen = pygame.display.set_mode((SCR_W, SCR_H), pygame.RESIZABLE)
 
            if ev.type == pygame.KEYDOWN:
                if self.active_inp:
                    if ev.key == pygame.K_BACKSPACE:
                        setattr(self, self.active_inp,
                                getattr(self, self.active_inp)[:-1])
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.apply_settings()
                        self.active_inp = None
                    elif ev.unicode.isdigit():
                        setattr(self, self.active_inp,
                                getattr(self, self.active_inp) + ev.unicode)
                else:
                    if ev.key == pygame.K_r:
                        self.run_search()
                    if ev.key == pygame.K_c:
                        self.reset_search()
                    if ev.key == pygame.K_SPACE:
                        self.run_dynamic()
 
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                self.handle_click(mx, my)
 
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.dragging_wall  = None
                self.placing_start  = False
                self.placing_goal   = False
 
            if ev.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    mx, my = ev.pos
                    if mx < GRID_W and self.dragging_wall is not None:
                        cell = self.px_to_cell(mx, my)
                        if cell:
                            if self.dragging_wall:
                                if cell not in (self.start, self.goal):
                                    self.grid.cells[cell[0]][cell[1]] = 1
                            else:
                                self.grid.cells[cell[0]][cell[1]] = 0
                            self.reset_search()
 
    def handle_click(self, mx, my):
        for name, rect in self.btn.items():
            if rect.collidepoint(mx, my):
                self.handle_btn(name)
                return
 
        if mx < GRID_W:
            cell = self.px_to_cell(mx, my)
            if cell is None:
                return
            if self.placing_start:
                self.start = cell
                self.grid.cells[cell[0]][cell[1]] = 0
                self.placing_start = False
                self.reset_search()
            elif self.placing_goal:
                self.goal = cell
                self.grid.cells[cell[0]][cell[1]] = 0
                self.placing_goal = False
                self.reset_search()
            else:
                current_val = self.grid.cells[cell[0]][cell[1]]
                self.dragging_wall = (current_val == 0)
                if cell not in (self.start, self.goal):
                    self.grid.cells[cell[0]][cell[1]] ^= 1
                self.reset_search()
 
        for name, rect in self.btn.items():
            if name.startswith("inp_") and rect.collidepoint(mx, my):
                self.active_inp = name.replace("inp_", "input_")
                return
 
    def handle_btn(self, name):
        if name == "run":
            self.run_search()
        elif name == "clear":
            self.reset_search()
            for r in range(self.rows):
                for c in range(self.cols):
                    self.grid.cells[r][c] = 0
            self.msg = "Grid cleared"
        elif name == "regen":
            self.grid.randomise(self.density, self.start, self.goal)
            self.reset_search()
            self.msg = "Map regenerated"
        elif name == "apply":
            self.apply_settings()
        elif name == "algo_gbfs":
            self.algo = "GBFS"; self.reset_search()
        elif name == "algo_astar":
            self.algo = "A*"; self.reset_search()
        elif name == "heur_man":
            self.heur = "Manhattan"; self.reset_search()
        elif name == "heur_euc":
            self.heur = "Euclidean"; self.reset_search()
        elif name == "set_start":
            self.placing_start = True
            self.placing_goal  = False
            self.msg = "Click a cell to set START"
        elif name == "set_goal":
            self.placing_goal  = True
            self.placing_start = False
            self.msg = "Click a cell to set GOAL"
        elif name == "dynamic":
            self.run_dynamic()
        elif name == "dyn_faster":
            self.dyn_interval = max(0.05, self.dyn_interval - 0.05)
        elif name == "dyn_slower":
            self.dyn_interval = min(1.0,  self.dyn_interval + 0.05)
 
    def update(self):
        if self.state == "searching" and self.gen:
            try:
                step = next(self.gen)
                self.result = step
                if step["done"]:
                    self.exec_ms = (time.time() - self._start_time) * 1000
                    self.state   = "done"
                    if step["path"]:
                        self.msg = (f"Path found! Cost={step.get('cost', len(step['path'])-1)}"
                                    f"  Nodes={step['nodes']}  {self.exec_ms:.1f}ms")
                    else:
                        self.msg = "No path found!"
            except StopIteration:
                self.state = "done"
 
        if self.state == "dynamic":
            now = time.time()
            if now - self.dyn_last_step >= self.dyn_interval:
                self.dyn_last_step = now
                if self.dyn_agent_idx < len(self.dyn_path) - 1:
                    self.dyn_agent_idx += 1
                    if random.random() < self.spawn_prob:
                        self.spawn_random_obstacle()
                else:
                    self.state = "done"
                    self.msg   = "Agent reached GOAL!"
 
    def spawn_random_obstacle(self):
        for _ in range(30):
            r = random.randint(0, self.rows-1)
            c = random.randint(0, self.cols-1)
            if (r,c) in (self.start, self.goal): continue
            if (r,c) in self.dyn_path:
                self.grid.cells[r][c] = 1
                self.msg = f"Obstacle spawned at ({r},{c}) — Replanning..."
                self.replan()
                return
            if self.grid.cells[r][c] == 0:
                self.grid.cells[r][c] = 1
                return
 
    def draw(self):
        self.screen.fill(BG)
        self.draw_grid()
        self.draw_panel()
        pygame.display.flip()
 
    def draw_grid(self):
        cs = self.cell
        surf = self.screen
        path_set = set(self.result["path"])
 
        agent_cell = None
        if self.state == "dynamic" and self.dyn_path:
            agent_cell = self.dyn_path[self.dyn_agent_idx]
            path_set   = set(self.dyn_path)
 
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cs; y = r * cs
                rect = pygame.Rect(x, y, cs-1, cs-1)
                cell = (r, c)
 
                if self.grid.cells[r][c] == 1:
                    colour = WALL
                elif cell == self.start:
                    colour = START_C
                elif cell == self.goal:
                    colour = GOAL_C
                elif cell in path_set:
                    colour = PATH_C
                elif cell in self.result["frontier"]:
                    colour = FRONTIER_C
                elif cell in self.result["expanded"]:
                    colour = VISITED_C
                else:
                    colour = EMPTY
 
                pygame.draw.rect(surf, colour, rect, border_radius=2 if cs > 8 else 0)
 
        if agent_cell and self.state == "dynamic":
            r, c   = agent_cell
            cx, cy = c*cs + cs//2, r*cs + cs//2
            rad    = max(3, cs//2 - 1)
            pygame.draw.circle(surf, (255,255,80), (cx, cy), rad)
            pygame.draw.circle(surf, (255,220,0),  (cx, cy), rad, 2)
 
        if cs >= 8:
            for r in range(self.rows+1):
                pygame.draw.line(surf, GRID_LINE, (0, r*cs), (GRID_W, r*cs))
            for c in range(self.cols+1):
                pygame.draw.line(surf, GRID_LINE, (c*cs, 0), (c*cs, SCR_H))
 
    def draw_panel(self):
        px = GRID_W
        surf = self.screen
        pygame.draw.rect(surf, PANEL_BG, (px, 0, PANEL_W, SCR_H))
        pygame.draw.line(surf, ACCENT, (px, 0), (px, SCR_H), 2)
 
        x0 = px + 12
        y  = 12
        self.btn = {}
 
        draw_text(surf, "PATHFINDING AGENT", x0+10, y, FONT_TTL, ACCENT)
        y += 30
        draw_text(surf, "AI 2002 — Assignment 2 Q6", x0+4, y, FONT_SM, (120,130,160))
        y += 28
 
        draw_text(surf, "ALGORITHM", x0, y, FONT_MD, ACCENT); y += 20
        bw = (PANEL_W-30)//2
        r1 = pygame.Rect(x0,       y, bw, 28)
        r2 = pygame.Rect(x0+bw+6,  y, bw, 28)
        draw_button(surf, r1, "GBFS",  self.algo=="GBFS",  r1.collidepoint(self.mouse_pos))
        draw_button(surf, r2, "A*",    self.algo=="A*",    r2.collidepoint(self.mouse_pos))
        self.btn["algo_gbfs"] = r1; self.btn["algo_astar"] = r2
        y += 36
 
        draw_text(surf, "HEURISTIC", x0, y, FONT_MD, ACCENT); y += 20
        r3 = pygame.Rect(x0,       y, bw, 28)
        r4 = pygame.Rect(x0+bw+6,  y, bw, 28)
        draw_button(surf, r3, "Manhattan", self.heur=="Manhattan", r3.collidepoint(self.mouse_pos))
        draw_button(surf, r4, "Euclidean", self.heur=="Euclidean", r4.collidepoint(self.mouse_pos))
        self.btn["heur_man"] = r3; self.btn["heur_euc"] = r4
        y += 36
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "GRID SETTINGS", x0, y, FONT_MD, ACCENT); y += 22
 
        lw = 70; iw = 56
        for label, attr in [("Rows", "input_rows"), ("Cols", "input_cols"), ("Wall%", "input_dens")]:
            draw_text(surf, label, x0, y+5, FONT_SM)
            inp_rect = pygame.Rect(x0+lw, y, iw, 24)
            border_c = ACCENT if self.active_inp == attr else GRID_LINE
            pygame.draw.rect(surf, (28,32,50), inp_rect, border_radius=4)
            pygame.draw.rect(surf, border_c, inp_rect, 1, border_radius=4)
            draw_text(surf, getattr(self, attr), inp_rect.x+5, inp_rect.y+4, FONT_SM)
            btn_key = "inp_" + attr.replace("input_", "")
            self.btn[btn_key] = inp_rect
            y += 30
 
        apply_r = pygame.Rect(x0, y, PANEL_W-24, 26)
        draw_button(surf, apply_r, "APPLY & REGENERATE", colour=ORANGE_BTN,
                    hover=apply_r.collidepoint(self.mouse_pos))
        self.btn["apply"] = apply_r; y += 34
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "CONTROLS", x0, y, FONT_MD, ACCENT); y += 22
 
        run_r  = pygame.Rect(x0, y, PANEL_W-24, 30)
        draw_button(surf, run_r, "RUN SEARCH  [R]", colour=GREEN_BTN,
                    hover=run_r.collidepoint(self.mouse_pos))
        self.btn["run"] = run_r; y += 36
 
        dyn_r  = pygame.Rect(x0, y, PANEL_W-24, 30)
        draw_button(surf, dyn_r, "DYNAMIC MODE  [SPACE]", colour=BTN_ON,
                    hover=dyn_r.collidepoint(self.mouse_pos))
        self.btn["dynamic"] = dyn_r; y += 36
 
        clr_r  = pygame.Rect(x0,       y, bw, 26)
        reg_r  = pygame.Rect(x0+bw+6,  y, bw, 26)
        draw_button(surf, clr_r, "Clear Grid", colour=RED_BTN,
                    hover=clr_r.collidepoint(self.mouse_pos))
        draw_button(surf, reg_r, "Regen Map",
                    hover=reg_r.collidepoint(self.mouse_pos))
        self.btn["clear"] = clr_r; self.btn["regen"] = reg_r; y += 34
 
        ss_r = pygame.Rect(x0,       y, bw, 26)
        sg_r = pygame.Rect(x0+bw+6,  y, bw, 26)
        draw_button(surf, ss_r, "Set Start", colour=(30,120,60),
                    hover=ss_r.collidepoint(self.mouse_pos))
        draw_button(surf, sg_r, "Set Goal",  colour=(150,40,40),
                    hover=sg_r.collidepoint(self.mouse_pos))
        self.btn["set_start"] = ss_r; self.btn["set_goal"] = sg_r; y += 34
 
        if self.state == "dynamic":
            draw_text(surf, f"Speed: {1/self.dyn_interval:.1f} steps/s", x0, y, FONT_SM)
            y += 18
            sf_r = pygame.Rect(x0,       y, bw, 24)
            sl_r = pygame.Rect(x0+bw+6,  y, bw, 24)
            draw_button(surf, sf_r, "Faster", hover=sf_r.collidepoint(self.mouse_pos))
            draw_button(surf, sl_r, "Slower", hover=sl_r.collidepoint(self.mouse_pos))
            self.btn["dyn_faster"] = sf_r; self.btn["dyn_slower"] = sl_r
            y += 32
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "LEGEND", x0, y, FONT_MD, ACCENT); y += 20
        legends = [
            (START_C,    "Start node"),
            (GOAL_C,     "Goal node"),
            (FRONTIER_C, "Frontier (open list)"),
            (VISITED_C,  "Expanded (closed)"),
            (PATH_C,     "Final path"),
            (WALL,       "Wall / obstacle"),
        ]
        for col, lbl in legends:
            pygame.draw.rect(surf, col, (x0, y+2, 14, 14), border_radius=2)
            draw_text(surf, lbl, x0+20, y+1, FONT_SM)
            y += 20
        y += 4
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "METRICS", x0, y, FONT_MD, ACCENT); y += 20
        metrics = [
            ("Algorithm",     self.algo),
            ("Heuristic",     self.heur),
            ("Nodes visited", str(self.result.get("nodes", 0))),
            ("Path cost",     str(self.result.get("cost", len(self.result["path"])-1
                                                  if self.result["path"] else 0))),
            ("Path length",   str(len(self.result["path"]))),
            ("Exec time",     f"{self.exec_ms:.1f} ms"),
        ]
        for k, v in metrics:
            draw_text(surf, f"{k}:", x0, y, FONT_SM, (140,150,180))
            draw_text(surf, v, x0+120, y, FONT_SM, TEXT_C)
            y += 17
 
        y = SCR_H - 36
        pygame.draw.rect(surf, (25,28,45), (px, y, PANEL_W, 36))
        pygame.draw.line(surf, ACCENT, (px, y), (px+PANEL_W, y))
        words = self.msg.split()
        line1 = ""; line2 = ""
        for w in words:
            if FONT_SM.size(line1+" "+w)[0] < PANEL_W-20:
                line1 += (" " if line1 else "") + w
            else:
                line2 += (" " if line2 else "") + w
        draw_text(surf, line1, x0, y+4,  FONT_SM, (200,220,255))
        if line2:
            draw_text(surf, line2, x0, y+18, FONT_SM, (200,220,255))
 
        if self.placing_start:
            draw_text(surf, "Click to place START", px+10, 6, FONT_SM, START_C)
        if self.placing_goal:
            draw_text(surf, "Click to place GOAL",  px+10, 6, FONT_SM, GOAL_C)
 
        pygame.draw.rect(surf, (20,22,36), (px, SCR_H-60, PANEL_W, 24))
        draw_text(surf, "R=Run  SPACE=Dynamic  C=Clear",
                  px+8, SCR_H-56, FONT_SM, (80,90,110))
 
if __name__ == "__main__":
    app = App()
    app.run()
 
import pygame
import heapq
import math
import time
import random
import sys
 
BG          = (15,  17,  26)
GRID_LINE   = (30,  35,  55)
EMPTY       = (22,  26,  40)
WALL        = (55,  60,  80)
START_C     = (0,  200, 120)
GOAL_C      = (255,  80,  80)
FRONTIER_C  = (255, 200,  30)
VISITED_C   = (60,  90, 180)
PATH_C      = (0,  230, 140)
TEXT_C      = (210, 215, 235)
PANEL_BG    = (18,  20,  32)
ACCENT      = (80, 160, 255)
BTN_ON      = (60, 130, 220)
BTN_OFF     = (38,  42,  62)
BTN_HOVER   = (50, 110, 190)
RED_BTN     = (180,  50,  50)
GREEN_BTN   = (40, 160,  80)
ORANGE_BTN  = (200, 120,  30)
 
PANEL_W    = 280
MIN_ROWS   = 5
MAX_ROWS   = 40
MIN_COLS   = 5
MAX_COLS   = 60
DEF_ROWS   = 20
DEF_COLS   = 30
DEF_DENSITY= 0.28
FPS        = 60
STEP_DELAY = 0.04
 
pygame.init()
INFO   = pygame.display.Info()
SCR_W  = min(1280, INFO.current_w - 20)
SCR_H  = min(750,  INFO.current_h - 60)
GRID_W = SCR_W - PANEL_W
 
FONT_SM  = pygame.font.SysFont("consolas", 13)
FONT_MD  = pygame.font.SysFont("consolas", 15, bold=True)
FONT_LG  = pygame.font.SysFont("consolas", 19, bold=True)
FONT_TTL = pygame.font.SysFont("consolas", 22, bold=True)
 
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
 
def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
 
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [[0]*cols for _ in range(rows)]
 
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols
 
    def passable(self, r, c):
        return self.in_bounds(r, c) and self.cells[r][c] == 0
 
    def neighbours(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.passable(nr, nc):
                yield (nr, nc)
 
    def randomise(self, density, start, goal):
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c] = 1 if random.random() < density else 0
        sr, sc = start;  self.cells[sr][sc] = 0
        gr, gc = goal;   self.cells[gr][gc] = 0
 
    def toggle(self, r, c, start, goal):
        if (r, c) in (start, goal):
            return
        self.cells[r][c] ^= 1
 
def greedy_bfs(grid, start, goal, heuristic):
    h = heuristic
    open_set = []
    heapq.heappush(open_set, (h(start, goal), start))
    came_from = {start: None}
    visited   = {start}
    frontier  = {start}
    expanded  = set()
    nodes_visited = 0
 
    while open_set:
        _, current = heapq.heappop(open_set)
        frontier.discard(current)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_visited += 1
 
        if current == goal:
            path = reconstruct(came_from, goal)
            yield {"done": True, "path": path, "frontier": frontier.copy(),
                   "expanded": expanded.copy(), "nodes": nodes_visited}
            return
 
        for nb in grid.neighbours(*current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                heapq.heappush(open_set, (h(nb, goal), nb))
                frontier.add(nb)
 
        yield {"done": False, "path": [], "frontier": frontier.copy(),
               "expanded": expanded.copy(), "nodes": nodes_visited}
 
    yield {"done": True, "path": [], "frontier": set(), "expanded": expanded.copy(),
           "nodes": nodes_visited}
 
 
def astar(grid, start, goal, heuristic):
    h = heuristic
    open_set  = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    came_from = {start: None}
    g_score   = {start: 0}
    expanded  = set()
    frontier  = {start}
    nodes_visited = 0
 
    while open_set:
        f, g, current = heapq.heappop(open_set)
        frontier.discard(current)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_visited += 1
 
        if current == goal:
            path = reconstruct(came_from, goal)
            yield {"done": True, "path": path, "frontier": frontier.copy(),
                   "expanded": expanded.copy(), "nodes": nodes_visited,
                   "cost": g_score[goal]}
            return
 
        for nb in grid.neighbours(*current):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb]   = tentative_g
                came_from[nb] = current
                f_new = tentative_g + h(nb, goal)
                heapq.heappush(open_set, (f_new, tentative_g, nb))
                frontier.add(nb)
 
        yield {"done": False, "path": [], "frontier": frontier.copy(),
               "expanded": expanded.copy(), "nodes": nodes_visited, "cost": 0}
 
    yield {"done": True, "path": [], "frontier": set(), "expanded": expanded.copy(),
           "nodes": nodes_visited, "cost": 0}
 
 
def reconstruct(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path
 
def draw_text(surf, text, x, y, font=None, colour=TEXT_C, center=False):
    font = font or FONT_SM
    img  = font.render(text, True, colour)
    rect = img.get_rect()
    if center:
        rect.centerx = x; rect.y = y
    else:
        rect.x = x; rect.y = y
    surf.blit(img, rect)
    return rect
 
def draw_button(surf, rect, label, active=False, hover=False, colour=None):
    c = colour if colour else (BTN_ON if active else (BTN_HOVER if hover else BTN_OFF))
    pygame.draw.rect(surf, c, rect, border_radius=6)
    pygame.draw.rect(surf, ACCENT, rect, 1, border_radius=6)
    draw_text(surf, label, rect.centerx, rect.centery - 8, FONT_SM, TEXT_C, center=True)
 
def pill(surf, rect, colour, label, font=None):
    pygame.draw.rect(surf, colour, rect, border_radius=4)
    draw_text(surf, label, rect.centerx, rect.centery - 7, font or FONT_SM, (20,20,20), center=True)
 
class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCR_W, SCR_H), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent — AI 2002 A2 Q6")
        self.clock  = pygame.time.Clock()
 
        self.rows    = DEF_ROWS
        self.cols    = DEF_COLS
        self.density = DEF_DENSITY
        self.start   = (self.rows//2, 1)
        self.goal    = (self.rows//2, self.cols-2)
        self.grid    = Grid(self.rows, self.cols)
        self.grid.randomise(self.density, self.start, self.goal)
 
        self.algo    = "A*"
        self.heur    = "Manhattan"
 
        self.state   = "idle"
        self.gen     = None
        self.result  = {"path":[], "frontier":set(), "expanded":set(), "nodes":0, "cost":0}
        self.exec_ms = 0
        self.msg     = "Draw walls or click RUN"
        self.dragging_wall = None
        self.placing_start = False
        self.placing_goal  = False
 
        self.dynamic_mode  = False
        self.dyn_agent_idx = 0
        self.dyn_path      = []
        self.dyn_last_step = 0
        self.dyn_interval  = 0.18
        self.spawn_prob    = 0.04
 
        self.input_rows = str(self.rows)
        self.input_cols = str(self.cols)
        self.input_dens = f"{int(self.density*100)}"
        self.active_inp = None
 
        self.btn = {}
        self.mouse_pos = (0,0)
        self._start_time = 0
 
    @property
    def cell(self):
        cw = GRID_W / self.cols
        ch = SCR_H  / self.rows
        return max(4, int(min(cw, ch)))
 
    def px_to_cell(self, x, y):
        c = int(x // self.cell)
        r = int(y // self.cell)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None
 
    @property
    def hfunc(self):
        return manhattan if self.heur == "Manhattan" else euclidean
 
    def reset_search(self):
        self.state  = "idle"
        self.gen    = None
        self.result = {"path":[], "frontier":set(), "expanded":set(), "nodes":0, "cost":0}
        self.dyn_agent_idx = 0
        self.dyn_path      = []
 
    def apply_settings(self):
        try:
            r = max(MIN_ROWS, min(MAX_ROWS, int(self.input_rows)))
            c = max(MIN_COLS, min(MAX_COLS, int(self.input_cols)))
            d = max(0, min(60, int(self.input_dens))) / 100
        except ValueError:
            return
        self.rows    = r
        self.cols    = c
        self.density = d
        self.start   = (r//2, 1)
        self.goal    = (r//2, c-2)
        self.grid    = Grid(r, c)
        self.grid.randomise(d, self.start, self.goal)
        self.reset_search()
        self.msg = "Grid regenerated!"
 
    def run_search(self):
        self.reset_search()
        if self.algo == "A*":
            self.gen = astar(self.grid, self.start, self.goal, self.hfunc)
        else:
            self.gen = greedy_bfs(self.grid, self.start, self.goal, self.hfunc)
        self.state      = "searching"
        self._start_time = time.time()
        self.msg        = f"Running {self.algo} with {self.heur}..."
 
    def run_dynamic(self):
        self.reset_search()
        if self.algo == "A*":
            gen = astar(self.grid, self.start, self.goal, self.hfunc)
        else:
            gen = greedy_bfs(self.grid, self.start, self.goal, self.hfunc)
        state = None
        for state in gen:
            pass
        if state and state["path"]:
            self.dyn_path      = list(state["path"])
            self.dyn_agent_idx = 0
            self.result        = state
            self.state         = "dynamic"
            self.dyn_last_step = time.time()
            self.msg = "Dynamic mode — obstacles may spawn!"
        else:
            self.msg = "No path found for dynamic mode."
 
    def replan(self):
        current = self.dyn_path[self.dyn_agent_idx]
        if self.algo == "A*":
            gen = astar(self.grid, current, self.goal, self.hfunc)
        else:
            gen = greedy_bfs(self.grid, current, self.goal, self.hfunc)
        state = None
        for state in gen:
            pass
        if state and state["path"]:
            self.dyn_path      = list(state["path"])
            self.dyn_agent_idx = 0
            self.result        = state
        else:
            self.state = "done"
            self.msg   = "No path after replanning!"
 
    def run(self):
        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
 
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
 
            if ev.type == pygame.VIDEORESIZE:
                global SCR_W, SCR_H, GRID_W
                SCR_W  = ev.w; SCR_H = ev.h
                GRID_W = SCR_W - PANEL_W
                self.screen = pygame.display.set_mode((SCR_W, SCR_H), pygame.RESIZABLE)
 
            if ev.type == pygame.KEYDOWN:
                if self.active_inp:
                    if ev.key == pygame.K_BACKSPACE:
                        setattr(self, self.active_inp,
                                getattr(self, self.active_inp)[:-1])
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.apply_settings()
                        self.active_inp = None
                    elif ev.unicode.isdigit():
                        setattr(self, self.active_inp,
                                getattr(self, self.active_inp) + ev.unicode)
                else:
                    if ev.key == pygame.K_r:
                        self.run_search()
                    if ev.key == pygame.K_c:
                        self.reset_search()
                    if ev.key == pygame.K_SPACE:
                        self.run_dynamic()
 
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                self.handle_click(mx, my)
 
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.dragging_wall  = None
                self.placing_start  = False
                self.placing_goal   = False
 
            if ev.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    mx, my = ev.pos
                    if mx < GRID_W and self.dragging_wall is not None:
                        cell = self.px_to_cell(mx, my)
                        if cell:
                            if self.dragging_wall:
                                if cell not in (self.start, self.goal):
                                    self.grid.cells[cell[0]][cell[1]] = 1
                            else:
                                self.grid.cells[cell[0]][cell[1]] = 0
                            self.reset_search()
 
    def handle_click(self, mx, my):
        for name, rect in self.btn.items():
            if rect.collidepoint(mx, my):
                self.handle_btn(name)
                return
 
        if mx < GRID_W:
            cell = self.px_to_cell(mx, my)
            if cell is None:
                return
            if self.placing_start:
                self.start = cell
                self.grid.cells[cell[0]][cell[1]] = 0
                self.placing_start = False
                self.reset_search()
            elif self.placing_goal:
                self.goal = cell
                self.grid.cells[cell[0]][cell[1]] = 0
                self.placing_goal = False
                self.reset_search()
            else:
                current_val = self.grid.cells[cell[0]][cell[1]]
                self.dragging_wall = (current_val == 0)
                if cell not in (self.start, self.goal):
                    self.grid.cells[cell[0]][cell[1]] ^= 1
                self.reset_search()
 
        for name, rect in self.btn.items():
            if name.startswith("inp_") and rect.collidepoint(mx, my):
                self.active_inp = name.replace("inp_", "input_")
                return
 
    def handle_btn(self, name):
        if name == "run":
            self.run_search()
        elif name == "clear":
            self.reset_search()
            for r in range(self.rows):
                for c in range(self.cols):
                    self.grid.cells[r][c] = 0
            self.msg = "Grid cleared"
        elif name == "regen":
            self.grid.randomise(self.density, self.start, self.goal)
            self.reset_search()
            self.msg = "Map regenerated"
        elif name == "apply":
            self.apply_settings()
        elif name == "algo_gbfs":
            self.algo = "GBFS"; self.reset_search()
        elif name == "algo_astar":
            self.algo = "A*"; self.reset_search()
        elif name == "heur_man":
            self.heur = "Manhattan"; self.reset_search()
        elif name == "heur_euc":
            self.heur = "Euclidean"; self.reset_search()
        elif name == "set_start":
            self.placing_start = True
            self.placing_goal  = False
            self.msg = "Click a cell to set START"
        elif name == "set_goal":
            self.placing_goal  = True
            self.placing_start = False
            self.msg = "Click a cell to set GOAL"
        elif name == "dynamic":
            self.run_dynamic()
        elif name == "dyn_faster":
            self.dyn_interval = max(0.05, self.dyn_interval - 0.05)
        elif name == "dyn_slower":
            self.dyn_interval = min(1.0,  self.dyn_interval + 0.05)
 
    def update(self):
        if self.state == "searching" and self.gen:
            try:
                step = next(self.gen)
                self.result = step
                if step["done"]:
                    self.exec_ms = (time.time() - self._start_time) * 1000
                    self.state   = "done"
                    if step["path"]:
                        self.msg = (f"Path found! Cost={step.get('cost', len(step['path'])-1)}"
                                    f"  Nodes={step['nodes']}  {self.exec_ms:.1f}ms")
                    else:
                        self.msg = "No path found!"
            except StopIteration:
                self.state = "done"
 
        if self.state == "dynamic":
            now = time.time()
            if now - self.dyn_last_step >= self.dyn_interval:
                self.dyn_last_step = now
                if self.dyn_agent_idx < len(self.dyn_path) - 1:
                    self.dyn_agent_idx += 1
                    if random.random() < self.spawn_prob:
                        self.spawn_random_obstacle()
                else:
                    self.state = "done"
                    self.msg   = "Agent reached GOAL!"
 
    def spawn_random_obstacle(self):
        for _ in range(30):
            r = random.randint(0, self.rows-1)
            c = random.randint(0, self.cols-1)
            if (r,c) in (self.start, self.goal): continue
            if (r,c) in self.dyn_path:
                self.grid.cells[r][c] = 1
                self.msg = f"Obstacle spawned at ({r},{c}) — Replanning..."
                self.replan()
                return
            if self.grid.cells[r][c] == 0:
                self.grid.cells[r][c] = 1
                return
 
    def draw(self):
        self.screen.fill(BG)
        self.draw_grid()
        self.draw_panel()
        pygame.display.flip()
 
    def draw_grid(self):
        cs = self.cell
        surf = self.screen
        path_set = set(self.result["path"])
 
        agent_cell = None
        if self.state == "dynamic" and self.dyn_path:
            agent_cell = self.dyn_path[self.dyn_agent_idx]
            path_set   = set(self.dyn_path)
 
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cs; y = r * cs
                rect = pygame.Rect(x, y, cs-1, cs-1)
                cell = (r, c)
 
                if self.grid.cells[r][c] == 1:
                    colour = WALL
                elif cell == self.start:
                    colour = START_C
                elif cell == self.goal:
                    colour = GOAL_C
                elif cell in path_set:
                    colour = PATH_C
                elif cell in self.result["frontier"]:
                    colour = FRONTIER_C
                elif cell in self.result["expanded"]:
                    colour = VISITED_C
                else:
                    colour = EMPTY
 
                pygame.draw.rect(surf, colour, rect, border_radius=2 if cs > 8 else 0)
 
        if agent_cell and self.state == "dynamic":
            r, c   = agent_cell
            cx, cy = c*cs + cs//2, r*cs + cs//2
            rad    = max(3, cs//2 - 1)
            pygame.draw.circle(surf, (255,255,80), (cx, cy), rad)
            pygame.draw.circle(surf, (255,220,0),  (cx, cy), rad, 2)
 
        if cs >= 8:
            for r in range(self.rows+1):
                pygame.draw.line(surf, GRID_LINE, (0, r*cs), (GRID_W, r*cs))
            for c in range(self.cols+1):
                pygame.draw.line(surf, GRID_LINE, (c*cs, 0), (c*cs, SCR_H))
 
    def draw_panel(self):
        px = GRID_W
        surf = self.screen
        pygame.draw.rect(surf, PANEL_BG, (px, 0, PANEL_W, SCR_H))
        pygame.draw.line(surf, ACCENT, (px, 0), (px, SCR_H), 2)
 
        x0 = px + 12
        y  = 12
        self.btn = {}
 
        draw_text(surf, "PATHFINDING AGENT", x0+10, y, FONT_TTL, ACCENT)
        y += 30
        draw_text(surf, "AI 2002 — Assignment 2 Q6", x0+4, y, FONT_SM, (120,130,160))
        y += 28
 
        draw_text(surf, "ALGORITHM", x0, y, FONT_MD, ACCENT); y += 20
        bw = (PANEL_W-30)//2
        r1 = pygame.Rect(x0,       y, bw, 28)
        r2 = pygame.Rect(x0+bw+6,  y, bw, 28)
        draw_button(surf, r1, "GBFS",  self.algo=="GBFS",  r1.collidepoint(self.mouse_pos))
        draw_button(surf, r2, "A*",    self.algo=="A*",    r2.collidepoint(self.mouse_pos))
        self.btn["algo_gbfs"] = r1; self.btn["algo_astar"] = r2
        y += 36
 
        draw_text(surf, "HEURISTIC", x0, y, FONT_MD, ACCENT); y += 20
        r3 = pygame.Rect(x0,       y, bw, 28)
        r4 = pygame.Rect(x0+bw+6,  y, bw, 28)
        draw_button(surf, r3, "Manhattan", self.heur=="Manhattan", r3.collidepoint(self.mouse_pos))
        draw_button(surf, r4, "Euclidean", self.heur=="Euclidean", r4.collidepoint(self.mouse_pos))
        self.btn["heur_man"] = r3; self.btn["heur_euc"] = r4
        y += 36
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "GRID SETTINGS", x0, y, FONT_MD, ACCENT); y += 22
 
        lw = 70; iw = 56
        for label, attr in [("Rows", "input_rows"), ("Cols", "input_cols"), ("Wall%", "input_dens")]:
            draw_text(surf, label, x0, y+5, FONT_SM)
            inp_rect = pygame.Rect(x0+lw, y, iw, 24)
            border_c = ACCENT if self.active_inp == attr else GRID_LINE
            pygame.draw.rect(surf, (28,32,50), inp_rect, border_radius=4)
            pygame.draw.rect(surf, border_c, inp_rect, 1, border_radius=4)
            draw_text(surf, getattr(self, attr), inp_rect.x+5, inp_rect.y+4, FONT_SM)
            btn_key = "inp_" + attr.replace("input_", "")
            self.btn[btn_key] = inp_rect
            y += 30
 
        apply_r = pygame.Rect(x0, y, PANEL_W-24, 26)
        draw_button(surf, apply_r, "APPLY & REGENERATE", colour=ORANGE_BTN,
                    hover=apply_r.collidepoint(self.mouse_pos))
        self.btn["apply"] = apply_r; y += 34
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "CONTROLS", x0, y, FONT_MD, ACCENT); y += 22
 
        run_r  = pygame.Rect(x0, y, PANEL_W-24, 30)
        draw_button(surf, run_r, "RUN SEARCH  [R]", colour=GREEN_BTN,
                    hover=run_r.collidepoint(self.mouse_pos))
        self.btn["run"] = run_r; y += 36
 
        dyn_r  = pygame.Rect(x0, y, PANEL_W-24, 30)
        draw_button(surf, dyn_r, "DYNAMIC MODE  [SPACE]", colour=BTN_ON,
                    hover=dyn_r.collidepoint(self.mouse_pos))
        self.btn["dynamic"] = dyn_r; y += 36
 
        clr_r  = pygame.Rect(x0,       y, bw, 26)
        reg_r  = pygame.Rect(x0+bw+6,  y, bw, 26)
        draw_button(surf, clr_r, "Clear Grid", colour=RED_BTN,
                    hover=clr_r.collidepoint(self.mouse_pos))
        draw_button(surf, reg_r, "Regen Map",
                    hover=reg_r.collidepoint(self.mouse_pos))
        self.btn["clear"] = clr_r; self.btn["regen"] = reg_r; y += 34
 
        ss_r = pygame.Rect(x0,       y, bw, 26)
        sg_r = pygame.Rect(x0+bw+6,  y, bw, 26)
        draw_button(surf, ss_r, "Set Start", colour=(30,120,60),
                    hover=ss_r.collidepoint(self.mouse_pos))
        draw_button(surf, sg_r, "Set Goal",  colour=(150,40,40),
                    hover=sg_r.collidepoint(self.mouse_pos))
        self.btn["set_start"] = ss_r; self.btn["set_goal"] = sg_r; y += 34
 
        if self.state == "dynamic":
            draw_text(surf, f"Speed: {1/self.dyn_interval:.1f} steps/s", x0, y, FONT_SM)
            y += 18
            sf_r = pygame.Rect(x0,       y, bw, 24)
            sl_r = pygame.Rect(x0+bw+6,  y, bw, 24)
            draw_button(surf, sf_r, "Faster", hover=sf_r.collidepoint(self.mouse_pos))
            draw_button(surf, sl_r, "Slower", hover=sl_r.collidepoint(self.mouse_pos))
            self.btn["dyn_faster"] = sf_r; self.btn["dyn_slower"] = sl_r
            y += 32
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "LEGEND", x0, y, FONT_MD, ACCENT); y += 20
        legends = [
            (START_C,    "Start node"),
            (GOAL_C,     "Goal node"),
            (FRONTIER_C, "Frontier (open list)"),
            (VISITED_C,  "Expanded (closed)"),
            (PATH_C,     "Final path"),
            (WALL,       "Wall / obstacle"),
        ]
        for col, lbl in legends:
            pygame.draw.rect(surf, col, (x0, y+2, 14, 14), border_radius=2)
            draw_text(surf, lbl, x0+20, y+1, FONT_SM)
            y += 20
        y += 4
 
        pygame.draw.line(surf, GRID_LINE, (px+8, y), (px+PANEL_W-8, y)); y += 10
        draw_text(surf, "METRICS", x0, y, FONT_MD, ACCENT); y += 20
        metrics = [
            ("Algorithm",     self.algo),
            ("Heuristic",     self.heur),
            ("Nodes visited", str(self.result.get("nodes", 0))),
            ("Path cost",     str(self.result.get("cost", len(self.result["path"])-1
                                                  if self.result["path"] else 0))),
            ("Path length",   str(len(self.result["path"]))),
            ("Exec time",     f"{self.exec_ms:.1f} ms"),
        ]
        for k, v in metrics:
            draw_text(surf, f"{k}:", x0, y, FONT_SM, (140,150,180))
            draw_text(surf, v, x0+120, y, FONT_SM, TEXT_C)
            y += 17
 
        y = SCR_H - 36
        pygame.draw.rect(surf, (25,28,45), (px, y, PANEL_W, 36))
        pygame.draw.line(surf, ACCENT, (px, y), (px+PANEL_W, y))
        words = self.msg.split()
        line1 = ""; line2 = ""
        for w in words:
            if FONT_SM.size(line1+" "+w)[0] < PANEL_W-20:
                line1 += (" " if line1 else "") + w
            else:
                line2 += (" " if line2 else "") + w
        draw_text(surf, line1, x0, y+4,  FONT_SM, (200,220,255))
        if line2:
            draw_text(surf, line2, x0, y+18, FONT_SM, (200,220,255))
 
        if self.placing_start:
            draw_text(surf, "Click to place START", px+10, 6, FONT_SM, START_C)
        if self.placing_goal:
            draw_text(surf, "Click to place GOAL",  px+10, 6, FONT_SM, GOAL_C)
 
        pygame.draw.rect(surf, (20,22,36), (px, SCR_H-60, PANEL_W, 24))
        draw_text(surf, "R=Run  SPACE=Dynamic  C=Clear",
                  px+8, SCR_H-56, FONT_SM, (80,90,110))
 
if __name__ == "__main__":
    app = App()
    app.run()
 
