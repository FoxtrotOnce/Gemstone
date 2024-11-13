import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.title('JQFileManager')
# root.withdraw()


file_path_output = ''


def file_selection():
    global file_path_output
    file_path_input = filedialog.askopenfilenames(
        title="Select all files that you want to modify.")
    if not len(file_path_input):
        return

    file_path_output = filedialog.askdirectory(
        title="Select the destination for the new files.")
    if not len(file_path_output):
        return

    for file_path in file_path_input:
        _, file_extension = os.path.splitext(file_path)
        match file_extension:
            case '.cfg' | '.dat' | '.plr':
                config_files(file_path, 1)
            case '.cfge' | '.date' | '.plre':
                config_files(file_path, -1)
            case '.pnge' | '.jpge' | '.gife':
                image_files(file_path, False)
            case '.png' | '.jpg' | '.gif':
                image_files(file_path, True)


def config_files(current_file, encrypt):
    with open(current_file, "rb") as f:
        file_array = bytearray(f.read())

        start_offset = 13
        line = rule24 = 0
        for i, byte in enumerate(file_array):
            if file_array[i] == 10:
                line += 1
                rule24 = 0
                start_offset = 13 + line % 3
            elif byte not in [0, 13, 254, 255]:
                file_array[i] = (file_array[i] + encrypt * (start_offset + rule24 % 24)) % 256
                rule24 += 1

        file_array = bytearray([x for x in file_array if x not in [0, 254, 255]])

    if encrypt == 1:
        current_file += 'e'
    else:
        current_file = current_file[:-1]

    current_file = os.path.basename(current_file)
    with open(f"{file_path_output}/{current_file}", "wb") as f:
        f.write(file_array)


def image_files(current_file, encrypt):
    with open(current_file, "rb") as f:
        file_array = bytearray(a ^ 1 for a in f.read())

    if encrypt:
        current_file += 'e'
    else:
        current_file = current_file[:-1]

    current_file = os.path.basename(current_file)
    with open(f"{file_path_output}/{current_file}", "wb") as f:
        f.write(file_array)


greeting = tk.Label(
    text="Supports: .cfg(e), .dat(e), .plr(e), .png(e), .jpg(e), .gif(e).",
    pady=20, padx=10)
button = tk.Button(
    text="Modify Files", width=10, padx=10, command=file_selection)
greeting.grid(row=0)
button.grid(row=1, pady=10)


root.mainloop()