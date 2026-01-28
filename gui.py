"""Simple Tkinter GUI that runs a sample analysis and plots results."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from chemometrics import sample_dataset, pca_transform


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CM Studio")
        self.geometry("800x600")
        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True)

        btn = ttk.Button(frm, text="Run sample PCA", command=self.run_pca)
        btn.pack(pady=10)

        self.canvas_frame = ttk.Frame(frm)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def run_pca(self):
        X = sample_dataset(200, 6)
        Xr, pca = pca_transform(X, n_components=2)
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.scatter(Xr[:, 0], Xr[:, 1], c=Xr[:, 0], cmap="viridis", s=20)
        ax.set_title("Sample PCA (2 components)")

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
