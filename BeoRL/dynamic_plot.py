import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.Qt import Qt
import beogym
import data_helper as dh
import numpy as np
import math

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, prev, curr, o):

        self.bo = beogym.BeoGym()
        self.agents_pos_prev = prev
        self.agents_pos_curr = curr
        self.o = o
        self.curr_image = self.o.image_name(self.agents_pos_curr)
        self.curr_angle = self.bo.get_angle(curr, prev) # Current view of the agent.
        self.turning_range = 45 # This parameter denotes how many degrees the agent turn everytime and turn left or turn right key is pressed.
        super(MainWindow, self).__init__()

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        self.xdata = []
        self.ydata = []
        self.update_plot(self.agents_pos_prev[0], self.agents_pos_prev[1])
        self.update_plot(self.agents_pos_curr[0], self.agents_pos_curr[1])

        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.left_key()
        if event.key() == Qt.Key_Right:
            self.right_key()
        if event.key() == Qt.Key_Up:
            self.up_key()
        if event.key() == Qt.Key_Down:
            self.down_key()

    def left_key(self):

        agents_pos_next, image_name, self.curr_angle = self.bo.go_left(self.agents_pos_curr, self.agents_pos_prev, self.o, self.curr_angle)

        self.update_plot(self.agents_pos_curr[0], self.agents_pos_curr[1])

    def right_key(self):

        agents_pos_next, image_name, self.curr_angle = self.bo.go_right(self.agents_pos_curr, self.agents_pos_prev, self.o, self.curr_angle)

        self.update_plot(self.agents_pos_curr[0], self.agents_pos_curr[1])

    def up_key(self):
        #print("the arguments are", self.agents_pos_curr, self.agents_pos_prev,)
        agents_pos_next, self.curr_image, self.curr_angle = self.bo.go_straight(self.agents_pos_curr, self.agents_pos_prev, self.o, self.curr_angle)

        self.agents_pos_prev = self.agents_pos_curr
        self.agents_pos_curr = agents_pos_next            
        
        self.update_plot(self.agents_pos_curr[0], self.agents_pos_curr[1])

    def down_key(self):

        agents_pos_next, self.curr_image, self.curr_angle = self.bo.go_back(self.agents_pos_curr, self.agents_pos_prev, self.o, self.curr_angle)

        self.agents_pos_prev = self.agents_pos_curr
        self.agents_pos_curr  = agents_pos_next
        
        self.update_plot(self.agents_pos_curr[0], self.agents_pos_curr[1])
    
    def draw_angle_cone(self, curr_pos, angle, color = 'm'):

        x = curr_pos[0]
        y = curr_pos[1]

        angle_range = [self.bo.fix_angle(angle - 45), self.bo.fix_angle(angle + 45)]
        line_length = 50

        for angle in angle_range:

            end_y = y + line_length * math.sin(math.radians(angle))
            end_x = x + line_length * math.cos(math.radians(angle))

            self.canvas.axes.plot([x, end_x], [y, end_y], ':' + color )

        self.canvas.draw()


    def update_plot(self, x, y):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata + [y]
        self.xdata = self.xdata + [x]

        #print("ydata: ", self.ydata)
        #print("xdata: ", self.xdata)

        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.xdata, self.ydata, '-ob')

        current_pos = (x,y)
        #print("Current node: \n", current_pos)

        adj_nodes_list = [keys for keys, values in self.o.G.adj[current_pos].items()]
        #print("Adj_nodes_list: \n", adj_nodes_list)
        num_adj_nodes = len(adj_nodes_list)
        adj_nodes_list = np.array( [[x_coor, y_coor] for x_coor, y_coor in adj_nodes_list])

        #print("Adj_nodes_list: \n", adj_nodes_list)

        x_pos_list = np.array([x] * num_adj_nodes)
        y_pos_list = np.array([y] * num_adj_nodes)

        #print("X_pos_list: \n", x_pos_list)

        #print("Adj_nodes_list[:,0]: \n", adj_nodes_list[:,0])
        self.canvas.axes.plot([x_pos_list,adj_nodes_list[:,0]], [y_pos_list, adj_nodes_list[:,1]], '--or')
        self.canvas.axes.plot(x, y, color = 'green', marker = 'o')
        self.canvas.axes.text(x, y, '({}, {})'.format(x, y))
        self.canvas.axes.plot(self.agents_pos_prev[0], self.agents_pos_prev[1], color = 'purple', marker = 'o')

        # View of the agent when it was at the previous node.
        #self.draw_angle_cone(self.agents_pos_prev, self.prev_angle )
        # Current view of the agent.
        self.draw_angle_cone(self.agents_pos_curr, self.curr_angle, color = 'g')
        # The turn right cone is red. The turn left cone is black.
        #self.draw_angle_cone(self.agents_pos_curr, self.curr_angle + 90, color = 'k')
        #self.draw_angle_cone(self.agents_pos_curr, self.curr_angle - 90, color = 'r')
        self.canvas.axes.set_xlim([-100, 100])
        self.canvas.axes.set_ylim([-100, 100])

        self.canvas.draw()

def main():
    o = dh.dataHelper()
    o.read_routes()
    prev = o.reset()
    print("image_name prev", o.image_name(prev))
    curr = o.find_adjacent(prev)[0]
    print("image_name curr", o.image_name(curr))
    print("agents_pos", curr)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(prev, curr, o)
    app.exec_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()