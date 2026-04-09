import math
import random
import time
import os
from collections import deque

import numpy as np
from controller import Robot
from stable_baselines3 import DQN # Same library as 2D environment 

# ----- Utility helpers ------

# Keep a value inside a safe numeric range.
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# Normalize an angle to [-pi, pi] so yaw errors stay consistent.
def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

# Four discrete grid moves used by the DQN policy.
ACTIONS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class MavicDQNController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        # Devices
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)

        self.lidar = self.robot.getDevice("Hokuyo URG-04LX")
        self.lidar.enable(self.timestep)

        self.front_left_motor = self.robot.getDevice("front left propeller")
        self.front_right_motor = self.robot.getDevice("front right propeller")
        self.rear_left_motor = self.robot.getDevice("rear left propeller")
        self.rear_right_motor = self.robot.getDevice("rear right propeller")

        for m in (
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor,
        ):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

     
        self.minimap = None
        self.minimap_w = 0
        self.minimap_h = 0
        try:
            self.minimap = self.robot.getDevice("minimap")
            self.minimap_w = self.minimap.getWidth()
            self.minimap_h = self.minimap.getHeight()
        except Exception:
            self.minimap = None

        # ---------------- Control constants ----------------
        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6

        self.k_alt_p = 7.0
        self.k_alt_d = 3.0
        self.max_vertical_cmd = 14.0

        self.k_roll_p_full = 50.0
        self.k_pitch_p_full = 30.0
        self.k_roll_p_takeoff = 22.0
        self.k_pitch_p_takeoff = 13.0

        self.k_yaw_p = 6.0
        self.k_yaw_d = 1.5
        self.max_yaw_cmd = 2.0

        self.max_motor_vel = 110.0
        self.tilt_min_cos = 0.70
        self.tilt_comp_cap = 1.50
        self.target_altitude = 8.0

        # Altitude dither
        self.alt_dither_enabled = True
        self.alt_dither_amp = 2.0
        self.alt_dither_period_s = 8.0
        self.alt_dither_ramp_s = 2.0
        self.alt_ceiling = 14.0

        self._alt_dither_rng = random.Random(123)
        self._alt_dither_next_t = None
        self._alt_dither_prev = 0.0
        self._alt_dither_goal = 0.0
        self._alt_dither_t0 = 0.0

        # Altitude safety
        self.airborne_floor_alt = 4.0
        self.panic_alt = 3.6
        self.panic_boost = 14.0
        self.panic_tilt_cap = 0.90

        self.min_safe_alt = 3.5
        self.near_ground_alt = 4.5

        self.floor_guard_margin = 0.8
        self.floor_tilt_suppress = 0.25
        self.floor_tilt_cap = 0.18

        # State machine
        self.state = "GROUND_INIT"
        self.ground_init_time = 2.5
        self.ground_init_t0 = None
        self.spool = 0.0

        self.coverage_gain_ramp_s = 2.0
        self.coverage_t0 = None
        self.yaw_target = None

        # Coverage / planning
        self.cell_size = 4.0
        self.grid_H = 20
        self.grid_W = 45

        self.region_half_width = (self.grid_W * self.cell_size) / 2.0
        self.region_half_height = (self.grid_H * self.cell_size) / 2.0

        self.origin_xy = None

        self.visited = set()
        self.visit_counts = {}
        self.entered = set()

        self.goal_world = None
        self.goal_hold_steps = 0
        self.max_goal_hold_steps = int(14.0 / self.dt)

        # Goal persistence / replanning
        self.goal_blocked_time = 0.0
        self.goal_blocked_timeout = 4.0

        self.goal_prev_dist = None
        self.goal_progress_timer = 0.0
        self.goal_stall_timeout = 5.0
        self.goal_progress_epsilon_m = 0.8

        # Candidate scoring weights kept for fallback / parity
        self.score_dist_w = 1.0
        self.score_gain_w = 3.8
        self.score_revisit_w = 0.55
        self.score_edge_w = 5.5
        self.frontier_neighbor_bonus = 0.35
        self.max_candidate_eval = 220

        # Constant-speed navigation
        self.cruise_speed = 2.0
        self.slowdown_radius = 5.0
        self.k_vel_forward = 0.22
        self.k_vel_lateral = 0.26
        self.max_forward_cmd = 0.50
        self.max_lateral_cmd = 0.50

        self.max_roll_cmd = 0.60
        self.max_pitch_cmd = 0.65
        self.goal_reached_radius = 4.0

        self.cmd_filter = 0.90
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0

        # Boundary guard / recovery
        self.boundary_soft_margin_m = 6.0
        self.boundary_hard_margin_cells = 3
        self.boundary_k_push = 0.09
        self.boundary_max_roll_cmd = 0.42
        self.boundary_max_pitch_cmd = 0.42
        self.boundary_resume_margin_m = 3.0
        self.boundary_recovery = False

        # Limits
        self.time_limit_s = 900.0
        self.energy_limit = 500.0

        # Metrics
        self.metrics_started = False
        self.wall_t0 = None
        self.sim_t0 = None
        self.prev_xy = None
        self.prev_cell = None
        self.path_len_m = 0.0
        self.path_len_cells = 0
        self.overlap_count = 0

        self.energy_proxy = 0.0
        self.energy_cell_step = 1.0
        self.energy_idle_per_s = 0.02

        self.steps = 0
        self._last_debug_t = -999.0

        # Altitude filtering
        self.alt_f = None
        self.prev_alt_f = None
        self.vz_f = 0.0
        self.alt_alpha = 0.12
        self.vz_alpha = 0.30

        # Coverage output
        self.reveal_radius_m = 6.0
        self.cover_scale = 12
        self.coverage_pgm_filename = "coverage_map.pgm"
        self.use_binary_pgm = True

        self.stop_reason = None

        # Yaw face travel
        self.face_travel_enabled = True
        self.prev_xy_for_vel = None
        self.vx_f = 0.0
        self.vy_f = 0.0
        self.vel_alpha = 0.25
        self.min_speed_for_yaw = 0.20

        self.face_goal_when_slow = True
        self.min_goal_dist_for_yaw = 1.0

        # Body velocity estimate for drift cancellation
        self._prev_vel_x = None
        self._prev_vel_y = None

        # Object avoidance
        self.oa_enabled = True
        self.oa_enable_altitude = 2.0
        self.oa_reverse_lidar = False

        self.oa_front_obstacle_dist = 22.0
        self.oa_emergency_dist = 6.0
        self.oa_side_switch_margin = 0.20

        self.oa_avoid_side_speed = 1.20
        self.oa_max_yaw_rate = 1.0
        self.oa_k_yaw_p = 3.5

        self.oa_side_danger_dist = 7.0
        self.oa_side_emergency_dist = 3.5

        self.oa_avoid_direction = 0
        self.oa_avoid_hold_time = 1.0
        self.oa_avoid_hold_timer = 0.0

        self.oa_detour_mode = False
        self.oa_resume_clear_time = 0.5
        self.oa_clear_timer = 0.0

        # ---------------- DQN planner ----------------
        # Number of cells observed around the current cell in each direction.
        self.obs_radius = 3
        self.n_stack = 4 # Number of recent observations stacked to provide short-term motion context.
        self.obs_hist = deque(maxlen=self.n_stack) # Rolling observation history used as input to the trained DQN.
        self.last_dqn_action = None

        model_path = os.path.join(os.path.dirname(__file__), "dqn_static_model.zip")
        self.dqn_model = DQN.load(model_path) # Load the pre-trained DQN policy used for grid-level action selection.
        print(f"[DQN] Loaded model: {model_path}")

        # ---------- Live minimap / fog-of-war ----------
        self.minimap_enabled = self.minimap is not None
        self.minimap_update_period = 3

        self.minimap_bg = 0x555555
        self.minimap_border = 0x00FFFF
        self.minimap_axis_x = 0xFF2020
        self.minimap_axis_y = 0x2040FF
        self.minimap_trail = 0x1FA31F
        self.minimap_current = 0x00FFFF
        self.minimap_goal = 0xFF2020
        self.minimap_text = 0xFFFFFF

        self.trail_world = []
        self.trail_decimate_dist = 0.8

    # ---------- Coverage helpers ----------
    def region_dims(self):
        return self.grid_H, self.grid_W
    
    # Convert world coordinates into grid indices without clipping to map bounds.
    def world_to_cell_unclamped(self, x, y):
        x0, y0 = self.origin_xy
        rx = x - x0
        ry = y - y0
        c = math.floor((rx + self.region_half_width) / self.cell_size)
        r = math.floor((ry + self.region_half_height) / self.cell_size)
        return int(r), int(c)

    def in_coverage_bounds(self, x, y):
        x0, y0 = self.origin_xy
        rx = x - x0
        ry = y - y0
        return (
            -self.region_half_width <= rx < self.region_half_width
            and -self.region_half_height <= ry < self.region_half_height
        )

    def world_to_cell(self, x, y):
        r, c = self.world_to_cell_unclamped(x, y)
        r = int(clamp(r, 0, self.grid_H - 1))
        c = int(clamp(c, 0, self.grid_W - 1))
        return (r, c)

    def bounds_relative(self, x, y):
        x0, y0 = self.origin_xy
        return x - x0, y - y0

    def near_coverage_edge(self, x, y, margin_m):
        rx, ry = self.bounds_relative(x, y)
        return (
            abs(rx) >= (self.region_half_width - margin_m)
            or abs(ry) >= (self.region_half_height - margin_m)
        )

    def nearest_interior_goal(self, x, y, margin_cells=None):
        if margin_cells is None:
            margin_cells = self.boundary_hard_margin_cells
        r, c = self.world_to_cell_unclamped(x, y)
        r = int(clamp(r, margin_cells, self.grid_H - 1 - margin_cells))
        c = int(clamp(c, margin_cells, self.grid_W - 1 - margin_cells))
        return self.cell_to_world_center(r, c)

    def cell_to_world_center(self, r, c):
        x0, y0 = self.origin_xy
        rx = (c + 0.5) * self.cell_size - self.region_half_width
        ry = (r + 0.5) * self.cell_size - self.region_half_height
        return (x0 + rx, y0 + ry)

    def cell_in_region(self, r, c):
        H, W = self.region_dims()
        return 0 <= r < H and 0 <= c < W

    def _mark_cell_visit(self, cell):
        if not self.cell_in_region(*cell):
            return
        self.visited.add(cell)
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1
        
    # Reveal all cells within the sensing radius around the current drone position.
    def _reveal_at_position(self, x, y):
        if self.origin_xy is None:
            return
        if not self.in_coverage_bounds(x, y):
            return

        r0, c0 = self.world_to_cell(x, y)
        if not self.cell_in_region(r0, c0):
            return

        rad_cells = int(math.ceil(self.reveal_radius_m / max(1e-6, self.cell_size)))
        rad2 = self.reveal_radius_m * self.reveal_radius_m

        for dr in range(-rad_cells, rad_cells + 1):
            for dc in range(-rad_cells, rad_cells + 1):
                r = r0 + dr
                c = c0 + dc
                if not self.cell_in_region(r, c):
                    continue
                cx, cy = self.cell_to_world_center(r, c)
                if (cx - x) ** 2 + (cy - y) ** 2 <= rad2:
                    self._mark_cell_visit((r, c))

    # ---------- DQN helpers ----------
    
    # Return the discrete actions that keep the agent inside the grid.
    def valid_actions_from_cell(self, cell):
        r, c = cell
        valid = []
        for idx, (dr, dc) in enumerate(ACTIONS_4):
            nr, nc = r + dr, c + dc
            if self.cell_in_region(nr, nc):
                valid.append(idx)
        return valid
    
    # Build the local DQN observation from nearby cells and coarse lidar obstacle cues.
    def local_observation_webots(self, cur_cell, radius=3):
        side = 2 * radius + 1
        occ = np.ones((side, side), dtype=np.float32) # Occupancy channel marks blocked or out-of-bounds cells.
        vis = np.zeros((side, side), dtype=np.float32) # Visitation channel marks which nearby cells have already been revealed.

        cr, cc = cur_cell

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r = cr + dr
                c = cc + dc
                if self.cell_in_region(r, c):
                    occ[dr + radius, dc + radius] = 0.0
                    vis[dr + radius, dc + radius] = 1.0 if (r, c) in self.visited else 0.0
        
        # Project a few simple obstacle warnings from lidar into the local grid observation.
        ranges = self.lidar.getRangeImage()
        if ranges:
            if self.oa_reverse_lidar:
                ranges = list(reversed(ranges))

            left_min = self.sector_min(ranges, 0.08, 0.30)
            front_min = self.sector_min(ranges, 0.44, 0.56)
            right_min = self.sector_min(ranges, 0.70, 0.92)

            if front_min < 6.0 and radius - 1 >= 0:
                occ[radius - 1, radius] = 1.0
            if left_min < 5.0 and radius - 1 >= 0:
                occ[radius, radius - 1] = 1.0
            if right_min < 5.0 and radius + 1 < side:
                occ[radius, radius + 1] = 1.0

        return np.concatenate([occ.flatten(), vis.flatten()], axis=0)
    
    # Stack recent observations so the policy can infer short-term motion trends.
    def get_stacked_obs(self, cur_cell):
        obs = self.local_observation_webots(cur_cell, radius=self.obs_radius).astype(np.float32)
        if len(self.obs_hist) == 0:
            for _ in range(self.n_stack):
                self.obs_hist.append(obs)
        else:
            self.obs_hist.append(obs)
        return np.concatenate(list(self.obs_hist), axis=0)
    
    # Query the trained DQN for the next grid move, then convert it into a world-space goal.
    def choose_new_goal(self, x, y):
        cur_cell = self.world_to_cell(x, y)
        self._mark_cell_visit(cur_cell)

        obs = self.get_stacked_obs(cur_cell)
        action, _ = self.dqn_model.predict(obs, deterministic=True)
        action = int(action)

        valid = self.valid_actions_from_cell(cur_cell)
        if valid and action not in valid: # If the policy proposes an invalid border move, fall back to a random valid action.
            action = random.choice(valid)

        self.last_dqn_action = action

        dr, dc = ACTIONS_4[action]
        nr, nc = cur_cell[0] + dr, cur_cell[1] + dc

        if not self.cell_in_region(nr, nc):
            return self.cell_to_world_center(*cur_cell)

        return self.cell_to_world_center(nr, nc)
  
  
    # Convert the current goal direction into desired roll/pitch motion in body coordinates.
    def compute_nav_disturbances(self, x, y, yaw, gx, gy, vx_body, vy_body):
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return 0.0, 0.0, dist

        speed_ref = self.cruise_speed
        if dist < self.slowdown_radius:
            speed_ref *= clamp(dist / max(1e-6, self.slowdown_radius), 0.15, 1.0)

        ux = dx / dist
        uy = dy / dist

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        v_forward_des = speed_ref * (ux * cy + uy * sy)
        v_left_des = speed_ref * (-ux * sy + uy * cy)

        e_forward = v_forward_des - vx_body
        e_left = v_left_des - vy_body

        pitch_cmd = -self.k_vel_forward * e_forward
        roll_cmd = self.k_vel_lateral * e_left

        pitch_cmd = clamp(pitch_cmd, -self.max_forward_cmd, self.max_forward_cmd)
        roll_cmd = clamp(roll_cmd, -self.max_lateral_cmd, self.max_lateral_cmd)
        return roll_cmd, pitch_cmd, dist
    
    # Push the drone back toward the interior when it gets too close to the map edge.
    def boundary_recovery_command(self, x, y, yaw):
        x0, y0 = self.origin_xy
        rx = x - x0
        ry = y - y0

        safe_hw = self.region_half_width - self.boundary_resume_margin_m
        safe_hh = self.region_half_height - self.boundary_resume_margin_m

        tx = clamp(rx, -safe_hw, safe_hw)
        ty = clamp(ry, -safe_hh, safe_hh)

        ex = tx - rx
        ey = ty - ry
        dist = math.hypot(ex, ey)

        outside = (
            abs(rx) > self.region_half_width
            or abs(ry) > self.region_half_height
        )

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        forward = ex * cy + ey * sy
        left = -ex * sy + ey * cy

        pitch_cmd = -self.boundary_k_push * forward
        roll_cmd = self.boundary_k_push * left

        roll_cmd = clamp(roll_cmd, -self.boundary_max_roll_cmd, self.boundary_max_roll_cmd)
        pitch_cmd = clamp(pitch_cmd, -self.boundary_max_pitch_cmd, self.boundary_max_pitch_cmd)

        return roll_cmd, pitch_cmd, dist, outside

    # ---------- Live minimap helpers ----------
    
    # Convert world coordinates into minimap pixel coordinates.
    def world_to_minimap_px(self, x, y):
        if self.origin_xy is None or self.minimap is None:
            return 0, 0

        x0, y0 = self.origin_xy
        rx = x - x0
        ry = y - y0

        u = (rx + self.region_half_width) / (2.0 * self.region_half_width)
        v = (ry + self.region_half_height) / (2.0 * self.region_half_height)

        u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)

        px = int(u * (self.minimap_w - 1))
        py = int((1.0 - v) * (self.minimap_h - 1))
        return px, py

    def update_trail(self, x, y):
        if not self.minimap_enabled:
            return

        if not self.trail_world:
            self.trail_world.append((x, y))
            return

        lx, ly = self.trail_world[-1]
        if math.hypot(x - lx, y - ly) >= self.trail_decimate_dist:
            self.trail_world.append((x, y))

    def visit_count_to_color(self, cnt):
        if cnt <= 0:
            return None
        if cnt == 1:
            return 0x1B6F1B
        if cnt == 2:
            return 0x208A20
        if cnt == 3:
            return 0x24A124
        return 0x29B929
    
    # Draw explored cells, path trail, current position, goal, and latest DQN action.
    def draw_minimap(self, x, y):
        if (not self.minimap_enabled) or (self.minimap is None) or (self.origin_xy is None):
            return

        d = self.minimap
        W = self.minimap_w
        H = self.minimap_h

        d.setColor(self.minimap_bg)
        d.fillRectangle(0, 0, W, H)

        # Revealed cells / fog of war
        cell_w = W / self.grid_W
        cell_h = H / self.grid_H

        for r in range(self.grid_H):
            for c in range(self.grid_W):
                cnt = self.visit_counts.get((r, c), 0)
                color = self.visit_count_to_color(cnt)
                if color is None:
                    continue

                px0 = int(c * cell_w)
                py0 = int((self.grid_H - 1 - r) * cell_h)
                pw = max(1, int(math.ceil(cell_w)))
                ph = max(1, int(math.ceil(cell_h)))

                d.setColor(color)
                d.fillRectangle(px0, py0, pw, ph)

        # Draw path trail
        if len(self.trail_world) >= 2:
            d.setColor(self.minimap_trail)
            for i in range(1, len(self.trail_world)):
                x1, y1 = self.trail_world[i - 1]
                x2, y2 = self.trail_world[i]
                p1x, p1y = self.world_to_minimap_px(x1, y1)
                p2x, p2y = self.world_to_minimap_px(x2, y2)
                d.drawLine(p1x, p1y, p2x, p2y)

        # Axes through origin
        ox, oy = self.world_to_minimap_px(self.origin_xy[0], self.origin_xy[1])

        d.setColor(self.minimap_axis_x)
        d.drawLine(0, oy, W - 1, oy)

        d.setColor(self.minimap_axis_y)
        d.drawLine(ox, 0, ox, H - 1)

        # Goal marker
        if self.goal_world is not None:
            gx, gy = self.goal_world
            pgx, pgy = self.world_to_minimap_px(gx, gy)
            d.setColor(self.minimap_goal)
            s = 10
            d.drawLine(pgx - s, pgy - s, pgx + s, pgy + s)
            d.drawLine(pgx - s, pgy + s, pgx + s, pgy - s)

        # Drone/current position marker
        px, py = self.world_to_minimap_px(x, y)
        d.setColor(self.minimap_current)
        d.fillOval(px - 4, py - 4, 8, 8)

        # Border
        d.setColor(self.minimap_border)
        d.drawRectangle(0, 0, W - 1, H - 1)

        # Tiny HUD text
        total_cells = self.grid_H * self.grid_W
        cov = 100.0 * len(self.visited) / max(1, total_cells)
        d.setColor(self.minimap_text)
        try:
            d.drawText(f"cov {cov:.1f}%", 8, 8)
            d.drawText(f"cell {self.world_to_cell(x, y)}", 8, 24)
            d.drawText(f"a {self.last_dqn_action}", 8, 40)
        except Exception:
            pass

    # ---------- Altitude filtering ----------
    
    # Low-pass filter altitude and estimate vertical speed from altitude changes.
    def _update_altitude_filters(self, altitude_raw):
        if self.alt_f is None:
            self.alt_f = float(altitude_raw)
            self.prev_alt_f = float(altitude_raw)
            self.vz_f = 0.0
            return self.alt_f, self.vz_f

        a = self.alt_alpha
        self.alt_f = (1.0 - a) * self.alt_f + a * float(altitude_raw)

        dz = (self.alt_f - self.prev_alt_f) / max(1e-6, self.dt)
        self.prev_alt_f = self.alt_f

        b = self.vz_alpha
        self.vz_f = (1.0 - b) * self.vz_f + b * dz
        return self.alt_f, self.vz_f

    # ---------- Altitude dither ----------
    
    # Add slow altitude variation to avoid flying at one exact height throughout the run.
    def altitude_dither_offset(self, t, has_airborne, panic_mode, near_floor):
        if (not self.alt_dither_enabled) or (not has_airborne) or panic_mode or near_floor:
            self._alt_dither_prev = 0.0
            self._alt_dither_goal = 0.0
            self._alt_dither_next_t = None
            return 0.0

        if self._alt_dither_next_t is None:
            self._alt_dither_next_t = t + self.alt_dither_period_s
            self._alt_dither_prev = 0.0
            self._alt_dither_goal = 0.0
            self._alt_dither_t0 = t

        if t >= self._alt_dither_next_t:
            self._alt_dither_prev = self._alt_dither_goal
            self._alt_dither_goal = self._alt_dither_rng.uniform(-self.alt_dither_amp, self.alt_dither_amp)
            self._alt_dither_t0 = t
            jitter = self._alt_dither_rng.uniform(
                -0.35 * self.alt_dither_period_s, 0.35 * self.alt_dither_period_s
            )
            self._alt_dither_next_t = t + max(4.0, self.alt_dither_period_s + jitter)

        u = (t - self._alt_dither_t0) / max(1e-6, self.alt_dither_ramp_s)
        u = clamp(u, 0.0, 1.0)
        return (1.0 - u) * self._alt_dither_prev + u * self._alt_dither_goal

    # ---------- Yaw target from travel ----------
    
    # Prefer facing the travel direction; if moving slowly, face the active goal instead.
    def _update_yaw_target_from_travel(self, x, y):
        if not self.face_travel_enabled:
            return

        if self.prev_xy_for_vel is None:
            self.prev_xy_for_vel = (x, y)
            return

        dx = x - self.prev_xy_for_vel[0]
        dy = y - self.prev_xy_for_vel[1]
        self.prev_xy_for_vel = (x, y)

        vx = dx / max(1e-6, self.dt)
        vy = dy / max(1e-6, self.dt)

        a = self.vel_alpha
        self.vx_f = (1.0 - a) * self.vx_f + a * vx
        self.vy_f = (1.0 - a) * self.vy_f + a * vy

        speed = math.hypot(self.vx_f, self.vy_f)

        if speed >= self.min_speed_for_yaw:
            self.yaw_target = math.atan2(self.vy_f, self.vx_f)
            return

        if self.face_goal_when_slow and (self.goal_world is not None):
            gx, gy = self.goal_world
            gdx = gx - x
            gdy = gy - y
            if math.hypot(gdx, gdy) >= self.min_goal_dist_for_yaw:
                self.yaw_target = math.atan2(gdy, gdx)

    # ---------- Body velocity ----------
    
    # Estimate body-frame velocity from GPS position differences and current yaw.
    def body_velocity_from_gps(self, x, y, yaw):
        if self._prev_vel_x is None:
            self._prev_vel_x = x
            self._prev_vel_y = y
            return 0.0, 0.0

        vx_world = (x - self._prev_vel_x) / max(1e-6, self.dt)
        vy_world = (y - self._prev_vel_y) / max(1e-6, self.dt)

        self._prev_vel_x = x
        self._prev_vel_y = y

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        vx_body = cy * vx_world + sy * vy_world
        vy_body = -sy * vx_world + cy * vy_world
        return vx_body, vy_body

    # ---------- Metrics ----------
    def _metrics_start_if_needed(self):
        if self.metrics_started:
            return
        self.metrics_started = True
        self.wall_t0 = time.perf_counter()
        self.sim_t0 = self.robot.getTime()

    def _sim_elapsed(self):
        if not self.metrics_started or self.sim_t0 is None:
            return 0.0
        return self.robot.getTime() - self.sim_t0

    def _limits_reached(self):
        if not self.metrics_started:
            return False
        if self._sim_elapsed() >= self.time_limit_s:
            self.stop_reason = f"time limit reached ({self.time_limit_s:.0f}s)"
            return True
        if self.energy_proxy >= self.energy_limit:
            self.stop_reason = f"energy limit reached ({self.energy_limit:.0f})"
            return True
        return False
    
    # Track path length, revisit count, and simple energy usage during the run.
    def _metrics_update(self, x, y, cur_cell, fl, fr, rl, rr):
        if self.prev_xy is not None:
            dx = x - self.prev_xy[0]
            dy = y - self.prev_xy[1]
            d = math.hypot(dx, dy)
            if d > 1e-3:
                self.path_len_m += d
        self.prev_xy = (x, y)

        moved_cell = False
        if self.prev_cell is not None and cur_cell != self.prev_cell:
            moved_cell = True
            self.path_len_cells += 1

            if self.cell_in_region(*cur_cell):
                if cur_cell in self.entered:
                    self.overlap_count += 1
                else:
                    self.entered.add(cur_cell)
                self._mark_cell_visit(cur_cell)

        if self.prev_cell is None:
            if self.cell_in_region(*cur_cell):
                self.entered.add(cur_cell)
            self._mark_cell_visit(cur_cell)

        if moved_cell:
            self.energy_proxy += self.energy_cell_step
        else:
            self.energy_proxy += self.energy_idle_per_s * self.dt

        self.prev_cell = cur_cell
        self.steps += 1
    
    # PGM MAP WAS NOT USED DURING TESTING OR EVALUATION
    def _write_coverage_pgm(self, filename):
        H, W = self.region_dims()
        scale = max(1, int(self.cover_scale))
        outH, outW = H * scale, W * scale
        maxval = 255

        max_count = max(self.visit_counts.values()) if self.visit_counts else 1

        def count_to_pix(cnt):
            if cnt <= 0:
                return 0
            if max_count <= 1:
                return 200
            u = math.log(1.0 + cnt) / math.log(1.0 + max_count)
            return int(120 + u * 135)

        img = bytearray(outW * outH)
        for r in range(H):
            for c in range(W):
                cnt = self.visit_counts.get((r, c), 0)
                pix = count_to_pix(cnt)
                for yy in range(r * scale, (r + 1) * scale):
                    row_off = yy * outW
                    base = c * scale
                    for xx in range(base, base + scale):
                        img[row_off + xx] = pix

        if self.use_binary_pgm:
            with open(filename, "wb") as f:
                header = (
                    f"P5\n# fog-of-war: reveal_radius_m={self.reveal_radius_m}, scale={scale}\n"
                    f"{outW} {outH}\n{maxval}\n"
                )
                f.write(header.encode("ascii"))
                f.write(img)
        else:
            with open(filename, "w", encoding="ascii") as f:
                f.write("P2\n")
                f.write(f"# fog-of-war: reveal_radius_m={self.reveal_radius_m}, scale={scale}\n")
                f.write(f"{outW} {outH}\n")
                f.write(f"{maxval}\n")
                for yy in range(outH):
                    start = yy * outW
                    f.write(" ".join(str(b) for b in img[start:start + outW]) + "\n")

    def _final_print(self): # Print final run metrics and write the optional coverage image.
        H, W = self.region_dims()
        total_cells = H * W
        cov = 100.0 * len(self.visited) / max(1, total_cells)
        sim_time_s = self.robot.getTime() - (self.sim_t0 if self.sim_t0 is not None else 0.0)
        wall_time_s = time.perf_counter() - (self.wall_t0 if self.wall_t0 is not None else time.perf_counter())

        print("\n================ DQN RUN SUMMARY ================")
        if self.stop_reason is not None:
            print(f"Stop reason:         {self.stop_reason}")
        print(f"Coverage (revealed): {cov:.2f}%  ({len(self.visited)}/{total_cells} cells)")
        print(f"Entered cells:       {len(self.entered)}")
        print(f"Path length:         {self.path_len_cells:d} cell-steps")
        print(f"Path length:         {self.path_len_m:.2f} m (XY distance)")
        print(f"Overlap count:       {self.overlap_count:d}   (re-entered center cell)")
        print(f"Sim time:            {sim_time_s:.2f} s")
        print(f"Wall time:           {wall_time_s:.2f} s")
        print(f"Energy (proxy):      {self.energy_proxy:.2f}  (limit={self.energy_limit:.0f})")
        print(f"Control steps:       {self.steps:d}")
        print("======================================================\n")

        try:
            out_path = os.path.abspath(self.coverage_pgm_filename)
            self._write_coverage_pgm(out_path)
            print(f"[coverage] Wrote fog-of-war PGM: {out_path}")
        except Exception as e:
            print(f"[coverage] Failed to write PGM: {e}")

    # ---------- Mixer priority ----------
    def mix_motors_with_priority(self, collective, roll_input, pitch_input, yaw_input):
        a_fl = (-roll_input + pitch_input - yaw_input)
        a_fr = (roll_input + pitch_input + yaw_input)
        a_rl = (-roll_input - pitch_input + yaw_input)
        a_rr = (roll_input - pitch_input - yaw_input)

        maxv = self.max_motor_vel
        s = 1.0

        for a in (a_fl, a_fr, a_rl, a_rr):
            if abs(a) < 1e-9:
                continue
            if a > 0:
                s = min(s, (maxv - collective) / a)
            else:
                s = min(s, (-maxv - collective) / a)

        s = clamp(s, 0.0, 1.0)

        fl = clamp(collective + s * a_fl, -maxv, maxv)
        fr = clamp(collective + s * a_fr, -maxv, maxv)
        rl = clamp(collective + s * a_rl, -maxv, maxv)
        rr = clamp(collective + s * a_rr, -maxv, maxv)
        return fl, fr, rl, rr, s

    def _coverage_gain_blend(self, t):
        if self.coverage_t0 is None:
            return self.k_roll_p_takeoff, self.k_pitch_p_takeoff
        u = (t - self.coverage_t0) / max(1e-6, self.coverage_gain_ramp_s)
        u = clamp(u, 0.0, 1.0)
        k_roll = (1.0 - u) * self.k_roll_p_takeoff + u * self.k_roll_p_full
        k_pitch = (1.0 - u) * self.k_pitch_p_takeoff + u * self.k_pitch_p_full
        return k_roll, k_pitch

    # ---------- OA helpers ----------
    
    # Return the minimum valid lidar range inside a normalized angular sector.
    def sector_min(self, ranges, start_ratio, end_ratio):
        n = len(ranges)
        i0 = int(start_ratio * n)
        i1 = int(end_ratio * n)
        sector = ranges[i0:i1]
        vals = [r for r in sector if math.isfinite(r) and r > 0.05]
        return min(vals) if vals else float("inf")
    
    # Generate lateral detour behavior when an obstacle blocks forward motion.
    def avoidance_command(self, altitude):
        if (not self.oa_enabled) or altitude < self.oa_enable_altitude:
            return False, True, 0.0, 0.0, 0.0, 0.0, float("inf"), float("inf"), float("inf")

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return False, True, 0.0, 0.0, 0.0, 0.0, float("inf"), float("inf"), float("inf")

        if self.oa_reverse_lidar:
            ranges = list(reversed(ranges))

        left_min = self.sector_min(ranges, 0.08, 0.30)
        front_left_min = self.sector_min(ranges, 0.30, 0.44)
        front_min = self.sector_min(ranges, 0.44, 0.56)
        front_right_min = self.sector_min(ranges, 0.56, 0.70)
        right_min = self.sector_min(ranges, 0.70, 0.92)

        clear_ahead = (
            front_min > self.oa_front_obstacle_dist
            and front_left_min > self.oa_front_obstacle_dist * 0.75
            and front_right_min > self.oa_front_obstacle_dist * 0.75
        )

        if clear_ahead:
            self.oa_avoid_direction = 0
            self.oa_avoid_hold_timer = 0.0
            return False, True, 0.0, 0.0, 0.0, 0.0, front_min, left_min, right_min

        left_score = min(left_min, front_left_min)
        right_score = min(right_min, front_right_min)

        if self.oa_avoid_direction == 0:
            self.oa_avoid_direction = 1 if left_score >= right_score else -1
            self.oa_avoid_hold_timer = self.oa_avoid_hold_time

        if self.oa_avoid_hold_timer <= 0.0:
            if self.oa_avoid_direction == 1 and right_score > left_score + self.oa_side_switch_margin:
                self.oa_avoid_direction = -1
                self.oa_avoid_hold_timer = self.oa_avoid_hold_time
            elif self.oa_avoid_direction == -1 and left_score > right_score + self.oa_side_switch_margin:
                self.oa_avoid_direction = 1
                self.oa_avoid_hold_timer = self.oa_avoid_hold_time

        denom = max(1e-6, self.oa_front_obstacle_dist - self.oa_emergency_dist)
        danger = 1.0 - clamp((front_min - self.oa_emergency_dist) / denom, 0.0, 1.0)

        if front_min < self.oa_emergency_dist:
            return (
                True,
                False,
                0.0,
                1.6 * self.oa_avoid_side_speed * self.oa_avoid_direction,
                0.9 * self.oa_avoid_direction,
                1.0,
                front_min,
                left_min,
                right_min,
            )

        return (
            True,
            False,
            0.0,
            (1.0 + 0.8 * danger) * self.oa_avoid_side_speed * self.oa_avoid_direction,
            (0.35 + 0.35 * danger) * self.oa_avoid_direction,
            danger,
            front_min,
            left_min,
            right_min,
        )

    # ---------- Main ----------
    def run(self):
        print("Starting DQN.")
        print(
            f"Limits: time={self.time_limit_s:.0f}s, energy={self.energy_limit:.0f}, "
            f"grid={self.grid_H}x{self.grid_W} ({self.grid_H * self.grid_W} cells)"
        )
        if not self.minimap_enabled:
            print("[minimap] No Display device named 'minimap' found. Live map disabled.")

        while self.robot.step(self.timestep) != -1:
            if self.robot.getTime() > 0.5:
                break

        try:
            while self.robot.step(self.timestep) != -1: # Let the simulator settle briefly before starting active control.
                t = self.robot.getTime()

                roll, pitch, yaw = self.imu.getRollPitchYaw()
                x, y, altitude_raw = self.gps.getValues()
                roll_vel, pitch_vel, yaw_rate = self.gyro.getValues()

                altitude, vz = self._update_altitude_filters(altitude_raw)
                vx_body_dbg, vy_body_dbg = self.body_velocity_from_gps(x, y, yaw)

                if self.origin_xy is None: # Use the initial drone position as the local origin of the coverage grid.
                    self.origin_xy = (x, y)
                    self._metrics_start_if_needed()
                    self.ground_init_t0 = t
                    self.update_trail(x, y)
                    print(f"Origin set to x={x:.2f}, y={y:.2f}")

                raw_r, raw_c = self.world_to_cell_unclamped(x, y)
                out_of_bounds = not self.in_coverage_bounds(x, y)

                if self.oa_avoid_hold_timer > 0.0:
                    self.oa_avoid_hold_timer -= self.dt
                    if self.oa_avoid_hold_timer < 0.0:
                        self.oa_avoid_hold_timer = 0.0

                cur_cell = self.world_to_cell(x, y)

                self._reveal_at_position(x, y)
                self.update_trail(x, y)
                
                # Gradually spool up the motors before entering takeoff mode.
                if self.state == "GROUND_INIT":
                    frac = (t - self.ground_init_t0) / max(1e-6, self.ground_init_time)
                    self.spool = clamp(frac, 0.0, 1.0)
                    if self.spool >= 1.0:
                        self.state = "TAKEOFF"
                
                # Switch to coverage mode once a safe airborne altitude has been reached.
                if self.state == "TAKEOFF" and altitude >= self.airborne_floor_alt:
                    self.state = "COVERAGE"
                    self.coverage_t0 = t
                    self.goal_world = None
                    self.goal_prev_dist = None
                    self.goal_progress_timer = 0.0
                    self.yaw_target = yaw
                    self.prev_xy_for_vel = None
                    self.vx_f = 0.0
                    self.vy_f = 0.0
                    print("Takeoff complete -> COVERAGE enabled.")

                has_airborne = (self.state == "COVERAGE")
                floor_alt = self.airborne_floor_alt if has_airborne else self.min_safe_alt
                
                # Panic mode suppresses aggressive motion and prioritizes altitude recovery.
                panic_mode = has_airborne and (
                    (altitude < self.panic_alt) or
                    (altitude < (floor_alt + 0.8) and vz < -0.8)
                )

                if self.state in ("GROUND_INIT", "TAKEOFF") or panic_mode:
                    k_roll_p = self.k_roll_p_takeoff
                    k_pitch_p = self.k_pitch_p_takeoff
                else:
                    k_roll_p, k_pitch_p = self._coverage_gain_blend(t)

                enable_lateral = (
                    (self.state == "COVERAGE")
                    and (not panic_mode)
                    and (altitude >= self.near_ground_alt)
                )

                near_edge = False
                if has_airborne and self.origin_xy is not None:
                    near_edge = self.near_coverage_edge(x, y, self.boundary_soft_margin_m)

                    if out_of_bounds:
                        self.boundary_recovery = True
                        self.goal_world = self.nearest_interior_goal(x, y)
                        self.goal_hold_steps = 0
                        self.goal_blocked_time = 0.0
                        self.goal_prev_dist = None
                        self.goal_progress_timer = 0.0
                    elif self.boundary_recovery:
                        rx, ry = self.bounds_relative(x, y)
                        comfortably_inside = (
                            abs(rx) <= self.region_half_width - self.boundary_resume_margin_m
                            and abs(ry) <= self.region_half_height - self.boundary_resume_margin_m
                        )
                        if comfortably_inside:
                            self.boundary_recovery = False

                roll_dist = 0.0
                pitch_dist = 0.0
                avoid_active = False
                avoid_yaw_input = 0.0
                oa_danger = 0.0
                oa_front_min = float("inf")
                oa_left_min = float("inf")
                oa_right_min = float("inf")
                
                # Query the DQN for a new local goal only when normal lateral motion is allowed.
                if enable_lateral and (not self.oa_detour_mode):
                    if self.goal_world is None or self.goal_hold_steps > self.max_goal_hold_steps:
                        self.goal_world = self.choose_new_goal(x, y)
                        self.goal_hold_steps = 0
                        self.goal_blocked_time = 0.0
                        self.goal_prev_dist = None
                        self.goal_progress_timer = 0.0

                    if self.goal_world is not None:
                        gx, gy = self.goal_world
                        rcmd, pcmd, dist = self.compute_nav_disturbances(
                            x, y, yaw, gx, gy, vx_body_dbg, vy_body_dbg
                        )

                        if self.goal_prev_dist is None:
                            self.goal_prev_dist = dist
                            self.goal_progress_timer = 0.0
                        else:
                            if dist < self.goal_prev_dist - self.goal_progress_epsilon_m:
                                self.goal_prev_dist = dist
                                self.goal_progress_timer = 0.0
                            else:
                                self.goal_progress_timer += self.dt
                                # If progress toward the current DQN-selected goal stalls, discard it and replan.
                                if self.goal_progress_timer >= self.goal_stall_timeout:
                                    self.goal_world = None
                                    self.goal_hold_steps = 0
                                    self.goal_blocked_time = 0.0
                                    self.goal_prev_dist = None
                                    self.goal_progress_timer = 0.0

                        if self.goal_world is not None:
                            self._roll_cmd = self.cmd_filter * self._roll_cmd + (1.0 - self.cmd_filter) * rcmd
                            self._pitch_cmd = self.cmd_filter * self._pitch_cmd + (1.0 - self.cmd_filter) * pcmd

                            roll_dist = self._roll_cmd
                            pitch_dist = self._pitch_cmd

                            if dist < self.goal_reached_radius:
                                self.goal_world = None
                                self.goal_hold_steps = 0
                                self.goal_blocked_time = 0.0
                                self.goal_prev_dist = None
                                self.goal_progress_timer = 0.0
                            else:
                                self.goal_hold_steps += 1

                    roll_dist = clamp(roll_dist, -self.max_roll_cmd, self.max_roll_cmd)
                    pitch_dist = clamp(pitch_dist, -self.max_pitch_cmd, self.max_pitch_cmd)
                else:
                    self._roll_cmd *= 0.95
                    self._pitch_cmd *= 0.95

                boundary_roll_bias = 0.0
                boundary_pitch_bias = 0.0
                
                
                # Boundary recovery overrides normal goal-following near or beyond the map edge.
                if enable_lateral and self.origin_xy is not None and (near_edge or self.boundary_recovery):
                    b_roll, b_pitch, _, bout = self.boundary_recovery_command(x, y, yaw)

                    if bout:
                        self.boundary_recovery = True
                        self.goal_world = self.nearest_interior_goal(x, y)
                        self.goal_hold_steps = 0
                        self.goal_blocked_time = 0.0
                        self.goal_prev_dist = None
                        self.goal_progress_timer = 0.0
                    else:
                        rx, ry = self.bounds_relative(x, y)
                        comfortably_inside = (
                            abs(rx) <= self.region_half_width - self.boundary_resume_margin_m
                            and abs(ry) <= self.region_half_height - self.boundary_resume_margin_m
                        )
                        if comfortably_inside:
                            self.boundary_recovery = False

                    rx, ry = self.bounds_relative(x, y)
                    dx_edge = self.region_half_width - abs(rx)
                    dy_edge = self.region_half_height - abs(ry)
                    min_edge_dist = min(dx_edge, dy_edge)

                    edge_u = 1.0 - clamp(
                        min_edge_dist / max(1e-6, self.boundary_soft_margin_m),
                        0.0,
                        1.0,
                    )
                    if bout:
                        edge_u = 1.0

                    boundary_roll_bias = edge_u * b_roll
                    boundary_pitch_bias = edge_u * b_pitch

                near_floor = has_airborne and (altitude < (floor_alt + self.floor_guard_margin))
                if near_floor: # Near the floor, reduce tilt authority to prioritize stable altitude recovery.
                    roll_dist *= self.floor_tilt_suppress
                    pitch_dist *= self.floor_tilt_suppress
                    roll_dist = clamp(roll_dist, -self.floor_tilt_cap, self.floor_tilt_cap)
                    pitch_dist = clamp(pitch_dist, -self.floor_tilt_cap, self.floor_tilt_cap)

                if panic_mode:
                    self.goal_world = None
                    self.goal_hold_steps = 0
                    self.goal_blocked_time = 0.0
                    self.goal_prev_dist = None
                    self.goal_progress_timer = 0.0
                    self.oa_detour_mode = False
                    self.oa_clear_timer = 0.0
                    self.boundary_recovery = False
                    roll_dist = 0.0
                    pitch_dist = 0.0
                
                # Obstacle avoidance temporarily overrides the DQN goal-following command.
                if enable_lateral and (not near_floor) and (not panic_mode):
                    (
                        avoid_active,
                        clear_ahead,
                        _,
                        _,
                        oa_yaw_rate_cmd,
                        oa_danger,
                        oa_front_min,
                        oa_left_min,
                        oa_right_min,
                    ) = self.avoidance_command(altitude)

                    if avoid_active:
                        self.oa_detour_mode = True
                        self.oa_clear_timer = 0.0
                    else:
                        if self.oa_detour_mode and clear_ahead:
                            self.oa_clear_timer += self.dt
                            if self.oa_clear_timer >= self.oa_resume_clear_time:
                                self.oa_detour_mode = False
                                self.oa_clear_timer = 0.0
                        else:
                            self.oa_clear_timer = 0.0

                    if avoid_active and self.goal_world is not None:
                        self.goal_blocked_time += self.dt
                    else:
                        self.goal_blocked_time = max(0.0, self.goal_blocked_time - 2.0 * self.dt)

                    if self.goal_world is not None and self.goal_blocked_time >= self.goal_blocked_timeout:
                        self.goal_world = None
                        self.goal_hold_steps = 0
                        self.goal_blocked_time = 0.0
                        self.goal_prev_dist = None
                        self.goal_progress_timer = 0.0

                    if self.oa_detour_mode:
                        base_roll = 0.38 + 0.16 * oa_danger
                        roll_dist = base_roll * self.oa_avoid_direction

                        pitch_dist *= 0.35

                        if self.oa_avoid_direction > 0:
                            if vy_body_dbg > 0.05:
                                roll_dist -= 0.55 * min(vy_body_dbg, 1.0)

                            if oa_left_min < self.oa_side_danger_dist:
                                side_u = 1.0 - clamp(
                                    (oa_left_min - self.oa_side_emergency_dist) /
                                    max(1e-6, self.oa_side_danger_dist - self.oa_side_emergency_dist),
                                    0.0,
                                    1.0
                                )
                                roll_dist *= (1.0 - 0.90 * side_u)
                                pitch_dist = max(pitch_dist, 0.10 * side_u)

                                if oa_left_min < self.oa_side_emergency_dist:
                                    roll_dist = -0.35
                                    pitch_dist = max(pitch_dist, 0.12)

                        else:
                            if vy_body_dbg < -0.05:
                                roll_dist += 0.55 * min(-vy_body_dbg, 1.0)

                            if oa_right_min < self.oa_side_danger_dist:
                                side_u = 1.0 - clamp(
                                    (oa_right_min - self.oa_side_emergency_dist) /
                                    max(1e-6, self.oa_side_danger_dist - self.oa_side_emergency_dist),
                                    0.0,
                                    1.0
                                )
                                roll_dist *= (1.0 - 0.90 * side_u)
                                pitch_dist = max(pitch_dist, 0.10 * side_u)

                                if oa_right_min < self.oa_side_emergency_dist:
                                    roll_dist = 0.35
                                    pitch_dist = max(pitch_dist, 0.12)

                        if oa_danger > 0.4 and vx_body_dbg > 0.10:
                            pitch_dist = max(pitch_dist, 0.10 + 0.12 * min(vx_body_dbg, 1.0))

                        if oa_danger > 0.75:
                            pitch_dist = max(pitch_dist, 0.12)

                        roll_dist = clamp(roll_dist, -self.max_roll_cmd, self.max_roll_cmd)
                        pitch_dist = clamp(pitch_dist, -self.max_pitch_cmd, self.max_pitch_cmd)

                        avoid_yaw_input = self.oa_k_yaw_p * clamp(
                            oa_yaw_rate_cmd - yaw_rate,
                            -self.oa_max_yaw_rate,
                            self.oa_max_yaw_rate,
                        )

                roll_dist += boundary_roll_bias
                pitch_dist += boundary_pitch_bias

                roll_dist = clamp(roll_dist, -self.max_roll_cmd, self.max_roll_cmd)
                pitch_dist = clamp(pitch_dist, -self.max_pitch_cmd, self.max_pitch_cmd)

                yaw_input = 0.0
                if has_airborne:
                    if self.yaw_target is None:
                        self.yaw_target = yaw

                    if (not panic_mode) and (not near_floor):
                        self._update_yaw_target_from_travel(x, y)

                    yaw_err = wrap_pi(self.yaw_target - yaw)
                    yaw_input = self.k_yaw_p * yaw_err - self.k_yaw_d * yaw_rate
                    yaw_input = clamp(yaw_input, -self.max_yaw_cmd, self.max_yaw_cmd)

                    yaw_input += avoid_yaw_input
                    yaw_input = clamp(yaw_input, -self.max_yaw_cmd, self.max_yaw_cmd)

                roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_vel + roll_dist
                pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_dist

                dither = self.altitude_dither_offset(t, has_airborne, panic_mode, near_floor)
                target_with_dither = min(self.target_altitude + dither, self.alt_ceiling)
                safe_target = max(target_with_dither, floor_alt)

                diff = safe_target - altitude + self.k_vertical_offset
                diff_bounded = clamp(diff, -1.2, 1.2)

                if has_airborne and altitude < (floor_alt + self.floor_guard_margin):
                    diff_bounded = max(0.0, diff_bounded)

                vertical_input = self.k_alt_p * diff_bounded - self.k_alt_d * vz
                vertical_input = clamp(vertical_input, -self.max_vertical_cmd, self.max_vertical_cmd)

                if self.state == "GROUND_INIT":
                    tilt_comp = 1.0
                    base_thrust = self.k_vertical_thrust * (0.25 + 0.75 * self.spool)
                    vertical_input *= self.spool
                    roll_input *= 0.2
                    pitch_input *= 0.2
                    yaw_input *= 0.2
                elif self.state == "TAKEOFF":
                    tilt_comp = 1.0
                    base_thrust = self.k_vertical_thrust
                else:
                    if panic_mode:
                        cosprod = math.cos(roll) * math.cos(pitch)
                        tilt_comp = 1.0 / max(self.panic_tilt_cap, cosprod)
                        vertical_input += self.panic_boost
                        roll_input *= 0.25
                        pitch_input *= 0.25
                        yaw_input *= 0.35
                    else:
                        cosprod = math.cos(roll) * math.cos(pitch)
                        tilt_comp = 1.0 / max(self.tilt_min_cos, cosprod)

                    tilt_comp = clamp(tilt_comp, 1.0, self.tilt_comp_cap)
                    if near_floor:
                        tilt_comp = min(tilt_comp, 1.25)

                    base_thrust = self.k_vertical_thrust * tilt_comp
                
                # Combine stabilization, navigation, altitude, and yaw commands into motor outputs.
                collective = base_thrust + vertical_input
                fl, fr, rl, rr, _ = self.mix_motors_with_priority(
                    collective, roll_input, pitch_input, yaw_input
                )

                self.front_left_motor.setVelocity(fl)
                self.front_right_motor.setVelocity(-fr)
                self.rear_left_motor.setVelocity(-rl)
                self.rear_right_motor.setVelocity(rr)

                self._metrics_update(x, y, cur_cell, fl, fr, rl, rr)

                if self.minimap_enabled and (self.steps % self.minimap_update_period == 0):
                    self.draw_minimap(x, y)

                total_cells = self.grid_H * self.grid_W
                cov = 100.0 * len(self.visited) / max(1, total_cells)

                if self._limits_reached():
                    break

                if t - self._last_debug_t > 1.0:
                    self._last_debug_t = t
                    gtxt = "None" if self.goal_world is None else f"({self.goal_world[0]:.1f},{self.goal_world[1]:.1f})"
                    front_txt = "inf" if math.isinf(oa_front_min) else f"{oa_front_min:.2f}"
                    left_txt = "inf" if math.isinf(oa_left_min) else f"{oa_left_min:.2f}"
                    right_txt = "inf" if math.isinf(oa_right_min) else f"{oa_right_min:.2f}"
                    speed_xy = math.hypot(vx_body_dbg, vy_body_dbg)
                    print(
                        f"state={self.state} panic={int(panic_mode)} avoid={int(avoid_active)} "
                        f"detour={int(self.oa_detour_mode)} brec={int(self.boundary_recovery)} "
                        f"oob={int(out_of_bounds)} raw_cell=({raw_r},{raw_c}) "
                        f"hold={self.oa_avoid_hold_timer:.2f} clear_t={self.oa_clear_timer:.2f} "
                        f"stall_t={self.goal_progress_timer:.2f} blocked_t={self.goal_blocked_time:.2f} "
                        f"front={front_txt} left={left_txt} right={right_txt} danger={oa_danger:.2f} "
                        f"vx={vx_body_dbg:.2f} vy={vy_body_dbg:.2f} speed={speed_xy:.2f} "
                        f"alt={altitude:.2f} cov={cov:.2f}% ({len(self.visited)}/{total_cells} cells) "
                        f"path_cells={self.path_len_cells} path_m={self.path_len_m:.1f} "
                        f"overlap={self.overlap_count} energy={self.energy_proxy:.2f} "
                        f"t={self._sim_elapsed():.0f}s goal={gtxt} dqn_a={self.last_dqn_action}"
                    )

        finally:
            if self.metrics_started:
                self._final_print()


if __name__ == "__main__":
    MavicDQNController().run()