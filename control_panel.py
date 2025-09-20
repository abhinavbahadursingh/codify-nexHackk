import tkinter as tk
from tkinter import font
import subprocess
import os
import sys

class ControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Control Panel")
        self.root.geometry("450x550")
        self.root.configure(bg="#1A1A1A")
        self.root.resizable(False, False)

        # --- Color Scheme & Fonts ---
        self.bg_color = "#1A1A1A"
        self.title_color = "#00FFD1"
        self.button_bg = "#2A2A2A"
        self.button_fg = "#FFFFFF"
        self.button_hover_bg = "#00FFD1"
        self.button_hover_fg = "#1A1A1A"

        self.title_font = font.Font(family="Consolas", size=20, weight="bold")
        self.button_font = font.Font(family="Consolas", size=14, weight="bold")

        # --- Title ---
        self.title_label = tk.Label(root, text="SYSTEM CONTROL PANEL",
                                    font=self.title_font, bg=self.bg_color, fg=self.title_color)
        self.title_label.pack(pady=(30, 20))
        self.title_animation_step = 0
        self.animate_title()

        # --- Button Frame ---
        button_frame = tk.Frame(root, bg=self.bg_color)
        button_frame.pack(pady=20, padx=30, fill="x")

        # --- Buttons ---
        self.buttons = []
        button_defs = [
            ("Run Real-time Tracking", self.run_realtime_tracking),
            ("Run Crash Detection", self.run_crash_detection),
            ("Run SUMO Sim (Before Model)", self.run_sumo_before),
            ("Run SUMO Sim (After Model)", self.run_sumo_after)
        ]

        for text, command in button_defs:
            btn = self.create_button(button_frame, text, command)
            btn.pack(pady=12, fill="x")
            self.buttons.append(btn)

    def create_button(self, parent, text, command):
        btn = tk.Button(
            parent,
            text=text,
            font=self.button_font,
            bg=self.button_bg,
            fg=self.button_fg,
            activebackground=self.button_hover_bg,
            activeforeground=self.button_hover_fg,
            relief="flat",
            pady=15,
            cursor="hand2",
            command=lambda: self.run_script_with_feedback(command, text)
        )
        btn.bind("<Enter>", self.on_enter)
        btn.bind("<Leave>", self.on_leave)
        return btn

    def on_enter(self, e):
        e.widget['background'] = self.button_hover_bg
        e.widget['foreground'] = self.button_hover_fg

    def on_leave(self, e):
        e.widget['background'] = self.button_bg
        e.widget['foreground'] = self.button_fg

    def animate_title(self):
        # Create a pulsating effect
        brightness = int(150 + 105 * (0.5 * (1 + abs(self.title_animation_step / 10 - 1))))
        color = f'#00{brightness:02x}{brightness-20:02x}'
        self.title_label.config(fg=color)
        self.title_animation_step = (self.title_animation_step + 1) % 20
        self.root.after(100, self.animate_title)

    def run_script_with_feedback(self, script_runner, original_text):
        clicked_button = None
        for btn in self.buttons:
            if btn.cget("text") in [original_text, "Launching..."]:
                clicked_button = btn
                break

        if not clicked_button:
            return

        # Feedback
        clicked_button.config(text="Launching...", state="disabled", bg="#555555")
        self.root.update()

        # Run the script
        script_runner()

        # Revert state after delay
        self.root.after(1000, lambda: clicked_button.config(
            text=original_text, state="normal", bg=self.button_bg))

    def run_script_with_cwd(self, script_path, cwd=None):
        try:
            subprocess.Popen(
                [sys.executable, script_path],
                cwd=cwd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        except FileNotFoundError:
            print(f"Error: Script not found at {script_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # --- Script Launchers ---
    def run_realtime_tracking(self):
        script_path = r"E:\codify_hackquanta\realTimeTracking\demo_working\mainRunFile.py"
        self.run_script_with_cwd(script_path, cwd=os.path.dirname(script_path))

    def run_crash_detection(self):
        script_path = r"E:\codify_hackquanta\crashDetectionSystem\Testing\detectoin.py"
        self.run_script_with_cwd(script_path, cwd=os.path.dirname(script_path))

    def run_sumo_before(self):
        script_path = r"E:\codify_hackquanta\rlSUmo-main\traci5.FT.py"
        self.run_script_with_cwd(script_path, cwd=os.path.dirname(script_path))

    def run_sumo_after(self):
        script_path = r"E:\codify_hackquanta\rlSUmo-main\traci6.QL.py"
        self.run_script_with_cwd(script_path, cwd=os.path.dirname(script_path))


if __name__ == "__main__":
    root = tk.Tk()
    app = ControlPanel(root)
    root.mainloop()
