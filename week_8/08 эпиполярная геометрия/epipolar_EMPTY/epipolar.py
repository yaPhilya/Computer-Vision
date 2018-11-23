from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import argparse


if __name__ == "__main__":

    w = 0
    curr_state = 1
    r = 4
    color_set = 'aqua'
    color_active = 'yellow'
    image1_point = np.zeros(3)

    parser = argparse.ArgumentParser()

    parser.add_argument('-image1', help="name of first image", required=True)
    parser.add_argument('-image2', help="name of second image", required=True)
    parser.add_argument('-out', help="name output .npy", required=True)

    args = parser.parse_args()

    points = np.zeros((1,2,3))
    np.save(args.out, points)

    root = Tk()
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(2, weight=1)


    #setting up a tkinter canvas with scrollbars
    
    xscroll_image1 = Scrollbar(frame, orient=HORIZONTAL)
    xscroll_image1.grid(row=1, column=0, sticky=E+W)
    yscroll_image1 = Scrollbar(frame)
    yscroll_image1.grid(row=0, column=1, sticky=N+S)

    canvas_image1 = Canvas(frame, bd=0, xscrollcommand=xscroll_image1.set, yscrollcommand=yscroll_image1.set)
    canvas_image1.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll_image1.config(command=canvas_image1.xview)
    yscroll_image1.config(command=canvas_image1.yview)

    img1 = ImageTk.PhotoImage(Image.open(args.image1))
    canvas_image1.create_image(0,0,image=img1,anchor="nw")
    canvas_image1.config(scrollregion=canvas_image1.bbox(ALL))
    frame.pack(fill=BOTH,expand=1)



    xscroll_image2 = Scrollbar(frame, orient=HORIZONTAL)
    xscroll_image2.grid(row=1, column=2, sticky=E+W)
    yscroll_image2 = Scrollbar(frame)
    yscroll_image2.grid(row=0, column=3, sticky=N+S)

    canvas_image2 = Canvas(frame, bd=0, xscrollcommand=xscroll_image2.set, yscrollcommand=yscroll_image2.set)
    canvas_image2.grid(row=0, column=2, sticky=N+S+E+W)
    xscroll_image2.config(command=canvas_image2.xview)
    yscroll_image2.config(command=canvas_image2.yview)

    img2 = ImageTk.PhotoImage(Image.open(args.image2))
    canvas_image2.create_image(0,0,image=img2,anchor="nw")
    canvas_image2.config(scrollregion=canvas_image2.bbox(ALL))
    frame.pack(fill=BOTH,expand=1)
   
    def printcoords(event):
        global w
        global curr_state
        global image1_point

        x = canvas_image1.canvasx(event.x)
        y = canvas_image1.canvasy(event.y)

        if curr_state == 2:
            
            canvas_image2.itemconfigure(w, fill=color_set)
            w = canvas_image1.create_oval(x-r, y-r, x+r, y+r, fill=color_active)
        else:

            canvas_image1.delete(w)
            w = canvas_image1.create_oval(x-r, y-r, x+r, y+r, fill=color_active)

        image1_point = [x,y,1]

        curr_state = 1


    def printcoords1(event):
        global w
        global curr_state
        global image1_point

        if curr_state == 0:
            print ('Image1 first')
            return

        points = np.load(args.out)

        x = canvas_image2.canvasx(event.x)
        y = canvas_image2.canvasy(event.y)

        if curr_state == 1:
            canvas_image1.itemconfigure(w, fill=color_set)
            w = canvas_image2.create_oval(x-r, y-r, x+r, y+r, fill=color_active)

            point = np.asarray([[image1_point[0],image1_point[1],1],[x, y, 1]]).reshape(1,2,3)

            if np.all(np.zeros((1,2,3)) == points):
                points = point
            else:
                points = np.vstack((points, point))

            np.save(args.out, points)
            print (points.shape[0], 'points saved')

        else:
            canvas_image2.delete(w)
            w = canvas_image2.create_oval(x-r, y-r, x+r, y+r, fill=color_active)

            points[-1,1] = (x,y,1)
            np.save(args.out, points)
        
        curr_state = 2

    canvas_image1.bind('<Button-1>',lambda event: printcoords(event))
    canvas_image2.bind('<Button-1>',lambda event: printcoords1(event))

    root.mainloop()
