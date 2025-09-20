import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from ultralytics import YOLO
from collections import defaultdict
import math
import optuna
import numpy as np
from datetime import datetime
import pickle
import os
from PIL import Image, ImageTk

import vehicleData
import speedData
import accident
import gps


class SpeedCalculator:
    def __init__(self, scale_factor_zone1=0.143, scale_factor_zone2=0.165,
                 smoothing_factor=0.3, min_distance_threshold=2.0):
        self.scale_factor_zone1 = scale_factor_zone1
        self.scale_factor_zone2 = scale_factor_zone2
        self.smoothing_factor = smoothing_factor
        self.min_distance_threshold = min_distance_threshold
        self.speed_history = defaultdict(list)

    def calculate_speed(self, track_id, curr_center, prev_center, frame_time, zone):
        distance_pixels = math.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                    (curr_center[1] - prev_center[1]) ** 2)

        if distance_pixels < self.min_distance_threshold:
            return 0.0

        scale_factor = self.scale_factor_zone1 if zone == 1 else self.scale_factor_zone2
        distance_meters = distance_pixels * scale_factor
        speed_mps = distance_meters / frame_time
        speed_kmph = speed_mps * 3.6

        if track_id in self.speed_history and self.speed_history[track_id]:
            last_speed = self.speed_history[track_id][-1]
            smoothed_speed = (self.smoothing_factor * speed_kmph +
                              (1 - self.smoothing_factor) * last_speed)
        else:
            smoothed_speed = speed_kmph

        self.speed_history[track_id].append(smoothed_speed)
        if len(self.speed_history[track_id]) > 10:
            self.speed_history[track_id].pop(0)

        return max(0, smoothed_speed)


class TrafficSpeedDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Speed Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')

        # Initialize variables
        self.video_path = None
        self.cap = None
        self.model = None
        self.is_running = False
        self.is_paused = False
        self.speed_calc = None
        self.optimized_params = None

        # Tracking variables
        self.reset_tracking_variables()

        # Create GUI elements
        self.create_widgets()

        # Load model
        self.load_model()

    def reset_tracking_variables(self):
        """Reset all tracking variables"""
        self.counted_ids_red_to_blue = set()
        self.counted_ids_blue_to_red = set()
        self.count_red_to_blue = defaultdict(int)
        self.count_blue_to_red = defaultdict(int)
        self.crossed_red_first = {}
        self.crossed_blue_first = {}
        self.prev_positions = {}
        self.line_y_red = 298
        self.line_y_blue = self.line_y_red + 150

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top control panel
        control_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # File selection
        tk.Label(control_frame, text="Video File:", bg='#3b3b3b', fg='white', font=('Arial', 10, 'bold')).pack(
            side=tk.LEFT, padx=5, pady=5)

        self.file_label = tk.Label(control_frame, text="No file selected", bg='#3b3b3b', fg='#cccccc',
                                   font=('Arial', 9))
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="Browse", command=self.browse_file, bg='#4CAF50', fg='white',
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)

        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#3b3b3b')
        button_frame.pack(side=tk.RIGHT, padx=5, pady=5)

        self.start_btn = tk.Button(button_frame, text="Start", command=self.start_detection, bg='#2196F3', fg='white',
                                   font=('Arial', 10, 'bold'), width=8)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = tk.Button(button_frame, text="Pause", command=self.pause_detection, bg='#FF9800', fg='white',
                                   font=('Arial', 10, 'bold'), width=8)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_detection, bg='#F44336', fg='white',
                                  font=('Arial', 10, 'bold'), width=8)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.optimize_btn = tk.Button(button_frame, text="Optimize", command=self.run_optimization, bg='#9C27B0',
                                      fg='white', font=('Arial', 10, 'bold'), width=8)
        self.optimize_btn.pack(side=tk.LEFT, padx=2)

        # Main content area
        content_frame = tk.Frame(main_frame, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Video display area
        video_frame = tk.Frame(content_frame, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_frame, text="Video Feed", bg='#1e1e1e', fg='white', font=('Arial', 16))
        self.video_label.pack(expand=True)

        # Right panel for statistics and controls
        right_panel = tk.Frame(content_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Parameters section
        params_frame = tk.LabelFrame(right_panel, text="Speed Parameters", bg='#3b3b3b', fg='white',
                                     font=('Arial', 11, 'bold'))
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        # Scale factor controls
        tk.Label(params_frame, text="Zone 1 Scale:", bg='#3b3b3b', fg='white').pack(anchor='w', padx=5, pady=2)
        self.scale1_var = tk.DoubleVar(value=0.143)
        self.scale1_scale = tk.Scale(params_frame, from_=0.05, to=0.3, resolution=0.001, orient=tk.HORIZONTAL,
                                     variable=self.scale1_var, bg='#3b3b3b', fg='white', highlightbackground='#3b3b3b')
        self.scale1_scale.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(params_frame, text="Zone 2 Scale:", bg='#3b3b3b', fg='white').pack(anchor='w', padx=5, pady=2)
        self.scale2_var = tk.DoubleVar(value=0.165)
        self.scale2_scale = tk.Scale(params_frame, from_=0.05, to=0.3, resolution=0.001, orient=tk.HORIZONTAL,
                                     variable=self.scale2_var, bg='#3b3b3b', fg='white', highlightbackground='#3b3b3b')
        self.scale2_scale.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(params_frame, text="Smoothing:", bg='#3b3b3b', fg='white').pack(anchor='w', padx=5, pady=2)
        self.smooth_var = tk.DoubleVar(value=0.3)
        self.smooth_scale = tk.Scale(params_frame, from_=0.1, to=0.8, resolution=0.1, orient=tk.HORIZONTAL,
                                     variable=self.smooth_var, bg='#3b3b3b', fg='white', highlightbackground='#3b3b3b')
        self.smooth_scale.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(params_frame, text="Apply Parameters", command=self.apply_parameters,
                  bg='#4CAF50', fg='white', font=('Arial', 9, 'bold')).pack(pady=5)

        # Statistics section
        stats_frame = tk.LabelFrame(right_panel, text="Vehicle Counts", bg='#3b3b3b', fg='white',
                                    font=('Arial', 11, 'bold'))
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create scrollable text widget for statistics
        self.stats_text = tk.Text(stats_frame, height=10, bg='#2b2b2b', fg='white', font=('Courier', 9))
        scrollbar = tk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Speed monitoring section
        speed_frame = tk.LabelFrame(right_panel, text="Speed Monitoring", bg='#3b3b3b', fg='white',
                                    font=('Arial', 11, 'bold'))
        speed_frame.pack(fill=tk.X, padx=10, pady=10)

        self.avg_speed_label = tk.Label(speed_frame, text="Avg Speed: -- km/h", bg='#3b3b3b', fg='#00ff00',
                                        font=('Arial', 12, 'bold'))
        self.avg_speed_label.pack(pady=5)

        self.max_speed_label = tk.Label(speed_frame, text="Max Speed: -- km/h", bg='#3b3b3b', fg='#ff6600',
                                        font=('Arial', 12, 'bold'))
        self.max_speed_label.pack(pady=5)

        self.violations_label = tk.Label(speed_frame, text="Speed Violations: 0", bg='#3b3b3b', fg='#ff0000',
                                         font=('Arial', 12, 'bold'))
        self.violations_label.pack(pady=5)

        # Status bar
        status_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(status_frame, text="Ready", bg='#1e1e1e', fg='white', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.fps_label = tk.Label(status_frame, text="FPS: --", bg='#1e1e1e', fg='white', font=('Arial', 10))
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=2)

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO("yolo11n.pt")
            self.class_list = self.model.names
            self.status_label.config(text="Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")
            self.status_label.config(text="Model loading failed")

    def browse_file(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.status_label.config(text=f"Video selected: {os.path.basename(file_path)}")

    def apply_parameters(self):
        """Apply manually adjusted parameters"""
        if self.speed_calc:
            self.speed_calc.scale_factor_zone1 = self.scale1_var.get()
            self.speed_calc.scale_factor_zone2 = self.scale2_var.get()
            self.speed_calc.smoothing_factor = self.smooth_var.get()
            self.status_label.config(text="Parameters updated")

    def run_optimization(self):
        """Run Optuna optimization in background thread"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        self.status_label.config(text="Running optimization...")
        self.optimize_btn.config(state='disabled')

        def optimize_thread():
            try:
                optimizer = OptimizedTracker(self.video_path)
                best_params = optimizer.optimize(n_trials=20)

                # Update GUI with optimized parameters
                self.root.after(0, self.update_optimized_parameters, best_params)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Optimization failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.optimize_btn.config(state='normal'))

        threading.Thread(target=optimize_thread, daemon=True).start()

    def update_optimized_parameters(self, params):
        """Update GUI with optimized parameters"""
        self.optimized_params = params
        self.scale1_var.set(params.get('scale_factor_zone1', 0.143))
        self.scale2_var.set(params.get('scale_factor_zone2', 0.165))
        self.smooth_var.set(params.get('smoothing_factor', 0.3))
        self.apply_parameters()
        self.status_label.config(text="Optimization completed")
        messagebox.showinfo("Success", "Parameters optimized successfully!")

    def start_detection(self):
        """Start video detection"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        if not self.model:
            messagebox.showerror("Error", "YOLO model not loaded")
            return

        self.is_running = True
        self.is_paused = False
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')

        # Initialize speed calculator
        self.speed_calc = SpeedCalculator(
            scale_factor_zone1=self.scale1_var.get(),
            scale_factor_zone2=self.scale2_var.get(),
            smoothing_factor=self.smooth_var.get()
        )

        # Reset tracking variables
        self.reset_tracking_variables()

        # Start detection thread
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def pause_detection(self):
        """Pause/resume detection"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="Resume")
            self.status_label.config(text="Detection paused")
        else:
            self.pause_btn.config(text="Pause")
            self.status_label.config(text="Detection running")

    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        self.is_paused = False
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text="Pause")
        self.stop_btn.config(state='disabled')

        if self.cap:
            self.cap.release()
            self.cap = None

        self.status_label.config(text="Detection stopped")

        # Clear video display
        self.video_label.config(image='', text="Video Feed")

    def detection_loop(self):
        """Main detection loop"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video file"))
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            frame_time = 1 / fps

            frame_count = 0
            speeds_for_monitoring = []
            violations_count = 0
            speed_limit = 60  # km/h

            # while self.is_running and self.cap.isOpened():
            #     if self.is_paused:
            #         continue

                # ret, frame = self.cap.read()
                # if not ret:
                #     break


                # frame_count += 1
                # frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
                #
                # # Run YOLO tracking
                # results = self.model.track(frame, persist=True)

            frame_skip = 0  # process 1 frame, skip next 2 (adjust as needed)

            while self.is_running and self.cap.isOpened():
                if self.is_paused:
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames
                if frame_count % (frame_skip + 1) != 0:
                    continue

                frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)

                # Run YOLO tracking
                results = self.model.track(frame, persist=True)

                if results and results[0].boxes is not None and results[0].boxes.data is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
                    class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []

                    # Draw detection lines
                    cv2.line(frame, (50, self.line_y_red), (750, self.line_y_red), (0, 0, 255), 3)
                    cv2.putText(frame, 'Red Line', (50, self.line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    cv2.line(frame, (50, self.line_y_blue), (750, self.line_y_blue), (255, 0, 0), 3)
                    cv2.putText(frame, 'Blue Line', (50, self.line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    # Process each detected object
                    for i in range(len(track_ids)):
                        if i >= len(boxes) or i >= len(class_indices):
                            continue

                        x1, y1, x2, y2 = map(int, boxes[i])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        track_id = track_ids[i]
                        class_name = self.class_list[class_indices[i]]
                        curr_center = (cx, cy)

                        # Draw bounding box and tracking info
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                        # Vehicle tracking
                        threading.Thread(target=vehicleData.track_vehicle, args=(track_id, cx, cy, class_name),
                                         daemon=True).start()

                        # Speed calculation
                        speed_kmph = 0
                        if track_id in self.prev_positions:
                            prev_center = self.prev_positions[track_id]

                            # Determine zone and calculate speed
                            if cy < self.line_y_red:
                                zone = 1
                            elif self.line_y_red < cy < self.line_y_blue:
                                zone = 2
                            else:
                                zone = 0  # Below blue line, no speed calculation

                            if zone > 0:
                                speed_kmph = self.speed_calc.calculate_speed(
                                    track_id, curr_center, prev_center, frame_time, zone
                                )

                                if speed_kmph > 0:
                                    speeds_for_monitoring.append(speed_kmph)

                                    # Check for speed violations
                                    if speed_kmph > speed_limit:
                                        violations_count += 1
                                        cv2.putText(frame, "VIOLATION!", (x1, y1 - 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                                    # Log speed
                                    threading.Thread(target=speedData.log_speed, args=(track_id, speed_kmph),
                                                     daemon=True).start()

                        # Display tracking ID, class, and speed
                        label_text = f"ID:{track_id} {class_name}"
                        if speed_kmph > 0:
                            label_text += f" {speed_kmph:.1f}km/h"

                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Update previous position
                        self.prev_positions[track_id] = curr_center

                        # Line crossing detection
                        if abs(cy - self.line_y_red) <= 5:
                            self.crossed_red_first[track_id] = True
                            threading.Thread(target=vehicleData.track_vehicle, args=(track_id, cx, cy, class_name),
                                             daemon=True).start()

                        if abs(cy - self.line_y_blue) <= 5:
                            self.crossed_blue_first[track_id] = True
                            threading.Thread(target=vehicleData.track_vehicle, args=(track_id, cx, cy, class_name),
                                             daemon=True).start()

                        # Counting logic
                        if track_id in self.crossed_red_first and track_id not in self.counted_ids_red_to_blue:
                            if abs(cy - self.line_y_blue) <= 5:
                                self.counted_ids_red_to_blue.add(track_id)
                                self.count_red_to_blue[class_name] += 1

                        if track_id in self.crossed_blue_first and track_id not in self.counted_ids_blue_to_red:
                            if abs(cy - self.line_y_red) <= 5:
                                self.counted_ids_blue_to_red.add(track_id)
                                self.count_blue_to_red[class_name] += 1

                # Update GUI elements
                if frame_count % 10 == 0:  # Update every 10 frames
                    self.root.after(0, self.update_display, frame, speeds_for_monitoring, violations_count, fps)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection error: {str(e)}"))
        finally:
            if self.cap:
                self.cap.release()
            self.root.after(0, self.stop_detection)

    def update_display(self, frame, speeds, violations_count, fps):
        """Update GUI display elements"""
        try:
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update video display
            self.video_label.config(image=frame_tk, text="")
            self.video_label.image = frame_tk  # Keep a reference

            # Update statistics
            self.update_statistics()

            # Update speed monitoring
            if speeds:
                avg_speed = np.mean(speeds[-50:])  # Last 50 speeds
                max_speed = max(speeds[-50:]) if speeds else 0
                self.avg_speed_label.config(text=f"Avg Speed: {avg_speed:.1f} km/h")
                self.max_speed_label.config(text=f"Max Speed: {max_speed:.1f} km/h")

            self.violations_label.config(text=f"Speed Violations: {violations_count}")
            self.fps_label.config(text=f"FPS: {fps:.1f}")

        except Exception as e:
            print(f"Display update error: {e}")

    def update_statistics(self):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)

        # Downward traffic
        self.stats_text.insert(tk.END, "=== DOWNWARD TRAFFIC ===\n", 'header')
        total_down = 0
        for class_name, count in self.count_red_to_blue.items():
            self.stats_text.insert(tk.END, f"{class_name}: {count}\n")
            total_down += count
        self.stats_text.insert(tk.END, f"Total Down: {total_down}\n\n")

        # Upward traffic
        self.stats_text.insert(tk.END, "=== UPWARD TRAFFIC ===\n", 'header')
        total_up = 0
        for class_name, count in self.count_blue_to_red.items():
            self.stats_text.insert(tk.END, f"{class_name}: {count}\n")
            total_up += count
        self.stats_text.insert(tk.END, f"Total Up: {total_up}\n\n")

        # Overall statistics
        self.stats_text.insert(tk.END, f"=== OVERALL ===\n")
        self.stats_text.insert(tk.END, f"Total Vehicles: {total_down + total_up}\n")
        self.stats_text.insert(tk.END, f"Active Tracks: {len(self.prev_positions)}\n")


class OptimizedTracker:
    def __init__(self, video_path, ground_truth_speeds=None):
        self.video_path = video_path
        self.ground_truth_speeds = ground_truth_speeds or {}
        self.model = YOLO("yolo11n.pt")
        self.model.to('cuda')
        self.class_list = self.model.names

    def objective(self, trial):
        """Optuna objective function"""
        scale_factor_zone1 = trial.suggest_float('scale_factor_zone1', 0.05, 0.3)
        scale_factor_zone2 = trial.suggest_float('scale_factor_zone2', 0.05, 0.3)
        smoothing_factor = trial.suggest_float('smoothing_factor', 0.1, 0.8)
        min_distance_threshold = trial.suggest_float('min_distance_threshold', 0.5, 5.0)

        speed_calc = SpeedCalculator(
            scale_factor_zone1=scale_factor_zone1,
            scale_factor_zone2=scale_factor_zone2,
            smoothing_factor=smoothing_factor,
            min_distance_threshold=min_distance_threshold
        )

        return self._evaluate_parameters(speed_calc)

    def _evaluate_parameters(self, speed_calc):
        """Evaluate parameters and return loss"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return float('inf')

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1 / fps
        prev_positions = {}
        all_speeds = []

        frame_count = 0
        max_frames = 500  # Limit for optimization speed

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = self.model.track(frame, persist=True)

            if results and results[0].boxes is not None and results[0].boxes.data is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

                for i in range(len(track_ids)):
                    if i >= len(boxes):
                        continue

                    x1, y1, x2, y2 = map(int, boxes[i])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    track_id = track_ids[i]
                    curr_center = (cx, cy)

                    if track_id in prev_positions:
                        zone = 1 if cy < 298 else 2
                        prev_center = prev_positions[track_id]

                        speed_kmph = speed_calc.calculate_speed(
                            track_id, curr_center, prev_center, frame_time, zone
                        )

                        if speed_kmph > 0:
                            all_speeds.append(speed_kmph)

                    prev_positions[track_id] = curr_center

        cap.release()

        # Calculate loss based on speed consistency and realism
        if not all_speeds:
            return float('inf')

        # Penalize unrealistic speeds (>150 km/h for highway traffic)
        extreme_speed_penalty = sum(max(0, speed - 150) ** 2 for speed in all_speeds)

        # Penalize high variance (inconsistent speeds)
        variance_penalty = np.var(all_speeds) if len(all_speeds) > 1 else 0

        # Penalize too many zero speeds (tracking issues)
        zero_speed_ratio = sum(1 for speed in all_speeds if speed < 1.0) / len(all_speeds)
        zero_penalty = zero_speed_ratio * 1000

        # Penalize unrealistically low average speeds
        avg_speed = np.mean(all_speeds)
        low_speed_penalty = max(0, 20 - avg_speed) ** 2 if avg_speed < 20 else 0

        total_loss = extreme_speed_penalty + variance_penalty * 0.1 + zero_penalty + low_speed_penalty
        return total_loss

    def optimize(self, n_trials=30):
        """Run Optuna optimization"""
        print("Starting hyperparameter optimization...")

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        print("Optimization completed!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best value: {study.best_value}")

        return study.best_params


class AdvancedControlPanel:
    def __init__(self, parent, gui_instance):
        self.parent = parent
        self.gui = gui_instance
        self.create_advanced_controls()

    def create_advanced_controls(self):
        """Create advanced control panel"""
        # Advanced settings frame
        advanced_frame = tk.LabelFrame(self.parent, text="Advanced Settings", bg='#3b3b3b', fg='white',
                                       font=('Arial', 11, 'bold'))
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)

        # Speed limit setting
        speed_limit_frame = tk.Frame(advanced_frame, bg='#3b3b3b')
        speed_limit_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(speed_limit_frame, text="Speed Limit (km/h):", bg='#3b3b3b', fg='white').pack(side=tk.LEFT)
        self.speed_limit_var = tk.IntVar(value=60)
        speed_limit_spinbox = tk.Spinbox(speed_limit_frame, from_=20, to=120, textvariable=self.speed_limit_var,
                                         width=8, bg='#2b2b2b', fg='white')
        speed_limit_spinbox.pack(side=tk.RIGHT)

        # Line position adjustments
        line_frame = tk.Frame(advanced_frame, bg='#3b3b3b')
        line_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(line_frame, text="Red Line Y:", bg='#3b3b3b', fg='white').pack(side=tk.LEFT)
        self.red_line_var = tk.IntVar(value=298)
        red_line_spinbox = tk.Spinbox(line_frame, from_=100, to=500, textvariable=self.red_line_var,
                                      width=8, bg='#2b2b2b', fg='white')
        red_line_spinbox.pack(side=tk.RIGHT)

        # Export/Import settings
        settings_frame = tk.Frame(advanced_frame, bg='#3b3b3b')
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(settings_frame, text="Export Settings", command=self.export_settings,
                  bg='#607D8B', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=2)

        tk.Button(settings_frame, text="Import Settings", command=self.import_settings,
                  bg='#795548', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=2)

        # Real-time parameter adjustment
        realtime_frame = tk.LabelFrame(self.parent, text="Real-time Adjustments", bg='#3b3b3b', fg='white',
                                       font=('Arial', 11, 'bold'))
        realtime_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(realtime_frame, text="Min Distance Threshold:", bg='#3b3b3b', fg='white').pack(anchor='w', padx=5,
                                                                                                pady=2)
        self.min_dist_var = tk.DoubleVar(value=2.0)
        min_dist_scale = tk.Scale(realtime_frame, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                                  variable=self.min_dist_var, bg='#3b3b3b', fg='white', highlightbackground='#3b3b3b')
        min_dist_scale.pack(fill=tk.X, padx=5, pady=2)

        # Detection confidence threshold
        tk.Label(realtime_frame, text="Detection Confidence:", bg='#3b3b3b', fg='white').pack(anchor='w', padx=5,
                                                                                              pady=2)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(realtime_frame, from_=0.1, to=0.9, resolution=0.1, orient=tk.HORIZONTAL,
                                    variable=self.confidence_var, bg='#3b3b3b', fg='white',
                                    highlightbackground='#3b3b3b')
        confidence_scale.pack(fill=tk.X, padx=5, pady=2)

    def export_settings(self):
        """Export current settings to file"""
        settings = {
            'scale_factor_zone1': self.gui.scale1_var.get(),
            'scale_factor_zone2': self.gui.scale2_var.get(),
            'smoothing_factor': self.gui.smooth_var.get(),
            'speed_limit': self.speed_limit_var.get(),
            'red_line_y': self.red_line_var.get(),
            'min_distance_threshold': self.min_dist_var.get(),
            'confidence_threshold': self.confidence_var.get()
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            import json
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "Settings exported successfully!")

    def import_settings(self):
        """Import settings from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    settings = json.load(f)

                # Apply settings
                self.gui.scale1_var.set(settings.get('scale_factor_zone1', 0.143))
                self.gui.scale2_var.set(settings.get('scale_factor_zone2', 0.165))
                self.gui.smooth_var.set(settings.get('smoothing_factor', 0.3))
                self.speed_limit_var.set(settings.get('speed_limit', 60))
                self.red_line_var.set(settings.get('red_line_y', 298))
                self.min_dist_var.set(settings.get('min_distance_threshold', 2.0))
                self.confidence_var.set(settings.get('confidence_threshold', 0.5))

                messagebox.showinfo("Success", "Settings imported successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to import settings: {str(e)}")


def create_menu_bar(root, gui_instance):
    """Create menu bar for the application"""
    menubar = tk.Menu(root, bg='#3b3b3b', fg='white')
    root.config(menu=menubar)

    # File menu
    file_menu = tk.Menu(menubar, tearoff=0, bg='#3b3b3b', fg='white')
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Video", command=gui_instance.browse_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    # Tools menu
    tools_menu = tk.Menu(menubar, tearoff=0, bg='#3b3b3b', fg='white')
    menubar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Optimize Parameters", command=gui_instance.run_optimization)
    tools_menu.add_command(label="Manual Calibration", command=show_calibration_dialog)

    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0, bg='#3b3b3b', fg='white')
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=show_about_dialog)


def show_calibration_dialog():
    """Show manual calibration dialog"""
    dialog = tk.Toplevel()
    dialog.title("Manual Calibration")
    dialog.geometry("400x300")
    dialog.configure(bg='#2b2b2b')

    tk.Label(dialog, text="Manual Calibration Helper", bg='#2b2b2b', fg='white', font=('Arial', 14, 'bold')).pack(
        pady=10)

    instructions = """
To improve speed accuracy:

1. Measure a known distance in your video
   (e.g., lane width ≈ 3.7m, vehicle length ≈ 4.5m)

2. Count the pixels for that same distance
   (use image editing software or measure on screen)

3. Calculate: scale_factor = real_distance_meters / pixel_distance

4. Enter the result in the Zone Scale parameters

Common reference measurements:
• Standard lane width: 3.7 meters
• Typical car length: 4.2-4.7 meters
• Highway lane markers: 3 meters long, 9 meters apart
"""

    text_widget = tk.Text(dialog, bg='#1e1e1e', fg='white', font=('Courier', 10), wrap=tk.WORD)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text_widget.insert(tk.END, instructions)
    text_widget.config(state='disabled')

    tk.Button(dialog, text="Close", command=dialog.destroy, bg='#607D8B', fg='white').pack(pady=10)


def show_about_dialog():
    """Show about dialog"""
    messagebox.showinfo("About",
                        "Traffic Speed Detection v2.0\n\n"
                        "Features:\n"
                        "• YOLO-based vehicle tracking\n"
                        "• Optuna hyperparameter optimization\n"
                        "• Real-time speed calculation\n"
                        "• Dual-zone speed monitoring\n"
                        "• Speed violation detection\n\n"
                        "Optimized for accuracy and performance.")


def main():
    """Main application entry point"""
    # Create main window
    root = tk.Tk()

    # Set application icon and styling
    root.configure(bg='#2b2b2b')

    # Create main GUI instance
    gui = TrafficSpeedDetectionGUI(root)

    # Create advanced control panel
    advanced_controls = AdvancedControlPanel(gui.root.winfo_children()[0].winfo_children()[2], gui)

    # Create menu bar
    create_menu_bar(root, gui)

    # Handle window closing
    def on_closing():
        if gui.is_running:
            gui.stop_detection()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the GUI event loop
    root.mainloop()


# Additional utility classes for enhanced functionality
class SpeedViolationDetector:
    def __init__(self, speed_limit=60):
        self.speed_limit = speed_limit
        self.violations = []

    def check_violation(self, track_id, speed, timestamp, location):
        """Check for speed violations and log them"""
        if speed > self.speed_limit:
            violation = {
                'track_id': track_id,
                'speed': speed,
                'speed_limit': self.speed_limit,
                'excess_speed': speed - self.speed_limit,
                'timestamp': timestamp,
                'location': location
            }
            self.violations.append(violation)
            return True
        return False

    def get_violation_report(self):
        """Generate violation report"""
        if not self.violations:
            return "No speed violations detected."

        report = f"Speed Violation Report\n"
        report += f"Total Violations: {len(self.violations)}\n\n"

        for i, v in enumerate(self.violations[-10:], 1):  # Show last 10
            report += f"{i}. Vehicle ID: {v['track_id']}\n"
            report += f"   Speed: {v['speed']:.1f} km/h (Limit: {v['speed_limit']} km/h)\n"
            report += f"   Excess: {v['excess_speed']:.1f} km/h\n"
            report += f"   Time: {v['timestamp']}\n\n"

        return report


class TrafficAnalyzer:
    def __init__(self):
        self.hourly_counts = defaultdict(int)
        self.speed_data = []
        self.peak_hours = []

    def analyze_traffic_patterns(self, counts_data, speed_data):
        """Analyze traffic patterns and generate insights"""
        total_vehicles = sum(counts_data.values())

        if not speed_data:
            return "Insufficient data for analysis"

        avg_speed = np.mean(speed_data)
        speed_std = np.std(speed_data)

        analysis = f"""
Traffic Analysis Report
======================

Vehicle Count: {total_vehicles}
Average Speed: {avg_speed:.1f} ± {speed_std:.1f} km/h

Speed Distribution:
• < 30 km/h: {sum(1 for s in speed_data if s < 30)} vehicles
• 30-60 km/h: {sum(1 for s in speed_data if 30 <= s <= 60)} vehicles  
• 60-90 km/h: {sum(1 for s in speed_data if 60 < s <= 90)} vehicles
• > 90 km/h: {sum(1 for s in speed_data if s > 90)} vehicles

Traffic Condition: {'Heavy' if total_vehicles > 50 else 'Moderate' if total_vehicles > 20 else 'Light'}
"""
        return analysis


# Performance monitoring with GUI integration
class GUIPerformanceMonitor:
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.frame_times = []
        self.processing_times = []

    def log_frame_time(self, frame_time):
        """Log frame processing time"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:  # Keep last 100 frames
            self.frame_times.pop(0)

    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.frame_times:
            return "No performance data available"

        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        return f"Avg FPS: {fps:.1f} | Frame Time: {avg_frame_time * 1000:.1f}ms"


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import optuna
        import cv2
        from ultralytics import YOLO

        print("All dependencies available. Starting application...")
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install optuna opencv-python ultralytics pillow numpy")