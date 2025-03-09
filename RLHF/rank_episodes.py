import os
import json
import argparse
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pickle
import subprocess
import platform
import sys

class RankingApp:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.video_dir = None
        self.trajectories = []
        self.video_paths = []
        self.rankings = {}
        
        self.root.title("Episode Ranking Tool")
        self.root.geometry("600x800")
        
        self._load_data()
        self._create_ui()
    
    def _load_data(self):
        # Load trajectories
        traj_path = self.data_dir / "trajectories.pkl"
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        
        with open(traj_path, 'rb') as f:
            data = pickle.load(f)
            self.trajectories = data["trajectories"]
            self.video_paths = data["video_paths"]
        
        # Use video dir from first video path
        if self.video_paths and self.video_paths[0]:
            self.video_dir = Path(self.video_paths[0]).parent
        else:
            self.video_dir = Path("preference_videos")  # Default
        
        # Load existing rankings if available
        rankings_path = self.data_dir / "rankings.json"
        if rankings_path.exists():
            with open(rankings_path, 'r') as f:
                self.rankings = json.load(f)
    
    def _create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        ttk.Label(main_frame, text="Rank the episodes from 1 (best) to N (worst)", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create episode list frame
        episode_frame = ttk.Frame(main_frame)
        episode_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(episode_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas and inner frame for scrolling
        canvas = tk.Canvas(episode_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas
        inner_frame = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
        
        # Episode entries
        self.rank_vars = {}
        for i, trajectory in enumerate(self.trajectories):
            episode_frame = ttk.Frame(inner_frame)
            episode_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Episode info
            stats = trajectory["info"]
            metrics = trajectory["musical_metrics"]
            
            info_text = f"Episode {i}: Return={trajectory['return']:.2f}, "
            if "pressed_keys" in stats:
                info_text += f"Keys={stats['pressed_keys']:.0f}, "
            if "correct_keys" in stats:
                info_text += f"Correct={stats['correct_keys']:.0f}, "
            if "note_precision" in metrics:
                info_text += f"Precision={metrics['note_precision']*100:.1f}%, "
            if "note_recall" in metrics:
                info_text += f"Recall={metrics['note_recall']*100:.1f}%"
                
            ttk.Label(episode_frame, text=info_text).pack(side=tk.LEFT, padx=5)
            
            # Play button
            play_btn = ttk.Button(episode_frame, text="Play", 
                                  command=lambda path=self.video_paths[i]: self._play_video(path))
            play_btn.pack(side=tk.LEFT, padx=5)
            
            # Rank dropdown
            rank_var = tk.StringVar()
            if str(i) in self.rankings:
                rank_var.set(str(self.rankings[str(i)]))
            
            rank_combo = ttk.Combobox(episode_frame, textvariable=rank_var, 
                                     values=[str(j) for j in range(1, len(self.trajectories) + 1)],
                                     width=5)
            rank_combo.pack(side=tk.RIGHT, padx=5)
            ttk.Label(episode_frame, text="Rank:").pack(side=tk.RIGHT)
            
            self.rank_vars[i] = rank_var
        
        # Update scroll region
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Save button
        save_btn = ttk.Button(button_frame, text="Save Rankings", command=self._save_rankings)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        # Generate preferences button
        gen_btn = ttk.Button(button_frame, text="Generate Preferences", command=self._generate_preferences)
        gen_btn.pack(side=tk.RIGHT, padx=5)
        
        # Open folder button
        folder_btn = ttk.Button(button_frame, text="Open Video Folder", 
                               command=lambda: self._open_folder(self.video_dir))
        folder_btn.pack(side=tk.LEFT, padx=5)
    
    def _play_video(self, video_path):
        """Play a video file using the default system player."""
        if not video_path or not Path(video_path).exists():
            print(f"Video file not found: {video_path}")
            return
        
        try:
            if platform.system() == 'Windows':
                os.startfile(video_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', video_path], check=True)
            else:  # Linux
                subprocess.run(['xdg-open', video_path], check=True)
        except Exception as e:
            print(f"Error playing video: {e}")
    
    def _open_folder(self, folder_path):
        """Open a folder in the file explorer."""
        try:
            if platform.system() == 'Windows':
                os.startfile(folder_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', folder_path], check=True)
            else:  # Linux
                subprocess.run(['xdg-open', folder_path], check=True)
        except Exception as e:
            print(f"Error opening folder: {e}")
    
    def _save_rankings(self):
        """Save the rankings to a JSON file."""
        # Collect rankings
        rankings = {}
        for idx, var in self.rank_vars.items():
            if var.get():  # Only save if a value is selected
                rankings[str(idx)] = int(var.get())
        
        # Save to file
        rankings_path = self.data_dir / "rankings.json"
        with open(rankings_path, 'w') as f:
            json.dump(rankings, f)
        
        print(f"Saved rankings to {rankings_path}")
        tk.messagebox.showinfo("Success", f"Rankings saved to {rankings_path}")
    
    def _generate_preferences(self):
        """Generate pairwise preferences and save them."""
        # First save the rankings
        self._save_rankings()
        
        # Execute the preference generation script
        cmd = [
            sys.executable, 
            "RLHF/generate_preference_data.py",
            "--checkpoint", "dummy",  # Not actually used when loading from disk
            "--data_dir", str(self.data_dir),
            "--rankings_file", str(self.data_dir / "rankings.json")
        ]
        
        # Run in subprocess to avoid importing in the GUI thread
        subprocess.run(cmd, check=True)
        
        tk.messagebox.showinfo("Success", "Pairwise preferences generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank episodes for preference data")
    parser.add_argument("--data_dir", type=str, default="preference_data", help="Directory with preference data")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = RankingApp(root, args.data_dir)
    root.mainloop() 