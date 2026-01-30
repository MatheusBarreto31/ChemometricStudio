import tkinter as tk
from main_gui import ChemometricsGUI

print("Starting Chemometric Studio GUI...")
root = tk.Tk()
root.iconbitmap("Icon.ico")
app = ChemometricsGUI(root)
root.mainloop() 
