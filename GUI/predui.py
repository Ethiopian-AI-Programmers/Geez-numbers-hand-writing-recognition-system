from tkinter import *
from tkinter.colorchooser import askcolor
from keras.models import load_model
import cv2
import numpy as np
import os
import time


class Paint(object):

    DEFAULT_PEN_SIZE = 50
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='predict', command=self.predictt)
        self.pen_button.grid(row=0, column=1)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.clear_button = Button(self.root, text='clear', command=self.use_clear)
        self.clear_button.grid(row=0, column=4)

        self.choose_size_button = Scale(self.root, from_=1, to=80, orient=HORIZONTAL)


        self.label = Label(self.root,text='The predicted value will be here ===>  ',font='Helvetica -16 bold')
        self.label.grid(row=3, column=0)



        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=2, columnspan=5)

        self.model = load_model('myclass_num.h5')
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


        self.setup()
        self.root.mainloop()




    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def predictt(self):
        #self.activate_button(self.pen_button)
        s = 'predim.ps'
        self.c.postscript(file=s,colormode='color')
        os.system('mogrify -format jpg *.ps')
        os.system('rm *.ps')


        im = cv2.imread('predim.jpg')
        im = cv2.resize(im,(28,28))
        im1 = np.reshape(im,[1,28,28,3])
        class1 = self.model.predict_classes(im1)
        print(class1[0])
        pt = 'The predicted value is ===>  ' + str(class1[0] + 1)
        self.label.config(text=pt)
        os.system('rm predim.jpg')
        #self.c.delete('all')



    def use_brush(self):
        self.activate_button(self.brush_button)

    def use_clear(self):
        self.c.delete('all')

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = 50
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
