import sys
sys.path.insert(1, 'C:/Users/M M Amjad/Documents/[=] A Level/Sixth Form/CS Project/Code')

from tkinter import *
import Network
from Network import Network

#net = Network ()


def btn_clicked():
    data = []

    try:
        Z = int (entry0.get())
    except ValueError:
        print ("Please enter an integer number of Protons")
        return

    try:
        N = int (entry1.get())
    except ValueError:
        print ("Please enter an integer number of Neutrons")
        return

    try:
        Q = float (entry2.get())
    except ValueError:
        print ("Please enter a number for Energy Release")
        return

    A = Z+N
    Zdist = min ([abs(Z-28), abs(Z-50), abs(Z-82), abs(Z-126)])
    Ndist = min ([abs(N-28), abs(N-50), abs(N-82), abs(N-84), abs(N-126)])
    data.append(Z)
    data.append(N)
    data.append(A)
    data.append(Q)
    data.append(Zdist)
    data.append(Ndist)

    print (data)
    #net.feedforward (data)


window = Tk()

window.geometry("502x526")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 526,
    width = 502,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    251.0, 268.0,
    image=background_img)

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    247.0, 153.5,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#c4c4c4",
    highlightthickness = 0)

entry0.place(
    x = 94, y = 124,
    width = 306,
    height = 57)

entry1_img = PhotoImage(file = f"img_textBox1.png")
entry1_bg = canvas.create_image(
    247.0, 263.5,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#c4c4c4",
    highlightthickness = 0)

entry1.place(
    x = 94, y = 234,
    width = 306,
    height = 57)

entry2_img = PhotoImage(file = f"img_textBox2.png")
entry2_bg = canvas.create_image(
    247.0, 375.5,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#c4c4c4",
    highlightthickness = 0)

entry2.place(
    x = 94, y = 346,
    width = 306,
    height = 57)

img0 = PhotoImage(file = f"img0.png")

b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 199, y = 426,
    width = 106,
    height = 34)

window.resizable(False, False)
window.mainloop()
