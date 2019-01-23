##
# Ransalu Senanayake
# This can generate simulated paths
##

import sys
import numpy as np
import matplotlib.pylab as pl
import matplotlib.patches as patches

class Obstacle():
    """
    Dynamic or static rectangular obstacle. It is assumed that dynamic objects are under constant acceleration.
    E.g. moving vehicle, parked vehicle, wall
    """
    def __init__(self, centroid, dx, dy, angle=0, vel=[1, 0], acc=[0, 0]):
        """
        :param centroid: centroid of the obstacle
        :param dx: length of the vehicle >=0
        :param dy: width of the vegicle >= 0
        :param angle: anti-clockwise rotation from the x-axis
        :param vel: [x-velocity, y-velocity], put [0,0] for static objects
        :param acc: [x-acceleration, y-acceleration], put [0,0] for static objects/constant velocity
        """
        self.centroid = centroid
        self.dx = dx
        self.dy = dy
        self.angle = angle
        self.vel = vel #moving up/right is positive
        self.acc = acc
        self.time = 0 #time is incremented for every self.update() call

    def __get_points(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        dx_cos = self.dx*np.cos(self.angle)
        dx_sin = self.dx*np.sin(self.angle)
        dy_sin = self.dy*np.sin(self.angle)
        dy_cos = self.dy*np.cos(self.angle)

        BR_x = centroid[0] + 0.5*(dx_cos + dy_sin) #BR=Bottom-right
        BR_y = centroid[1] + 0.5*(dx_sin - dy_cos)
        BL_x = centroid[0] - 0.5*(dx_cos - dy_sin)
        BL_y = centroid[1] - 0.5*(dx_sin + dy_cos)
        TL_x = centroid[0] - 0.5*(dx_cos + dy_sin)
        TL_y = centroid[1] - 0.5*(dx_sin - dy_cos)
        TR_x = centroid[0] + 0.5*(dx_cos - dy_sin)
        TR_y = centroid[1] + 0.5*(dx_sin + dy_cos)

        seg_bottom = (BL_x, BL_y, BR_x, BR_y)
        seg_left = (BL_x, BL_y, TL_x, TL_y)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (TL_x, TL_y, TR_x, TR_y)
            seg_right = (BR_x, BR_y, TR_x, TR_y)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def __get_points_old(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        seg_bottom = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] - self.dy/2)
        seg_left = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] - self.dx/2, centroid[1] + self.dy/2)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (centroid[0] - self.dx/2, centroid[1] + self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            seg_right = (centroid[0] + self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def update(self, pos=None, recycle_pos=True):
        """
        :param pos: manually give a position. If None, update based on time.
        :return: updated centroid
        """
        if pos is None:
            disp_x = self.centroid[0] + self.vel[0]*self.time + 0.5*self.acc[0]*(self.time**2) #s_x = ut + 0.5at^2
            disp_y = self.centroid[1] + self.vel[1]*self.time + 0.5*self.acc[1]*(self.time**2) #s_y = ut + 0.5at^2
        else:
            if recycle_pos is True:
                if self.time >= pos.shape[0]:
                    t = self.time%pos.shape[0]
                else:
                    t = self.time
            else: #stay at where it is when t > t_max
                if self.time > pos.shape[0]:
                    t = pos.shape[0]
                else:
                    t = self.time
            disp_x = pos[t, 0]
            disp_y = pos[t, 1]
        self.time += 1 #time is incremented for every self.update() call
        return self.__get_points(centroid=[disp_x, disp_y])

def connect_segments(segments, resolution = 0.01):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           step_size : resolution for plotting
    :return: stack of all connected line segments as (X, Y)
    """

    for i, seg_i in enumerate(segments):
        if seg_i[1] == seg_i[3]: #horizontal segment
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = seg_i[1]*np.ones(len(x))
        elif seg_i[0] == seg_i[2]: #vertical segment
            y = np.arange(min(seg_i[1],seg_i[3]), max(seg_i[1],seg_i[3]), resolution)
            x = seg_i[0]*np.ones(len(y))
        else: # gradient exists
            m = (seg_i[3] - seg_i[1])/(seg_i[2] - seg_i[0])
            c = seg_i[1] - m*seg_i[0]
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = m*x + c

        obs = np.vstack((x, y)).T
        if i == 0:
            connected_segments = obs
        else:
            connected_segments = np.vstack((connected_segments, obs))

    return connected_segments

def load_obstacles(environment):
    if environment == 'crosswalk1': #dynamic robot in a dynamic environment
        obs1 = Obstacle(centroid=[10, 12.5], dx=1000, dy=3005, angle=0, vel=[0, 0], acc=[0, 0])  # a wall
        all_obstacles = (obs1,)
        area = (0, 20, 0, 25)
    else:
        print(environment + ' not specified!')

    return all_obstacles, area

def get_way_points(environment, vehicle_poses=None):
    class mouse_events:
        def __init__(self, fig, line):
            self.path_start = False #If true, capture data
            self.fig = fig
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.orientation = []
            self.path_no = -1
            self.path_no_list = []

        def connect(self):
            self.a = self.fig.canvas.mpl_connect('button_press_event', self.__on_press)
            self.b = self.fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)

        def __on_press(self, event):
            print('You pressed', event.button, event.xdata, event.ydata)
            self.path_start = not self.path_start
            if self.path_start is True:
                self.path_no += 1

        def __on_motion(self, event):
            if self.path_start is True:
                if len(self.orientation) == 0:
                    self.orientation.append(0)
                else:
                    self.orientation.append( np.pi/2 + np.arctan2( (self.ys[-1] - event.ydata), (self.xs[-1] - event.xdata) ) )

                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.path_no_list.append(self.path_no)
                self.line.set_data(self.xs, self.ys)
                #self.line.figure.canvas.draw()

    # set up the environment
    all_obstacles, area = load_obstacles(environment=environment)

    # update obstacles
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update()

    connected_components = connect_segments(all_obstacle_segments)

    # plot
    pl.close('all')
    fig = pl.figure()  # (9,5)
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(connected_components[:, 0], connected_components[:, 1], marker='.', c='k', edgecolor='', alpha=0.2)  # obstacles
    if vehicle_poses is not None:
        pl.plot(vehicle_poses[:, 0], vehicle_poses[:, 1], 'o--', c='m')
    #pl.xlim(area[:2]); pl.ylim(area[2:])

    for dx in range(5,15,2):
        ax.add_patch(patches.Rectangle(
            (dx+0.5, 12.5-5/2),   # (x,y)
            1,          # width
            5,          # height
            alpha=0.3,
            facecolor="#000000"
        ))
    ax.add_patch(patches.Rectangle(
        (10-5, 12.5-35/2), 10, 35, fill=False,
        linewidth=3
    ))

    #pl.axis('equal')
    pl.gca().set_ylim(0-5, 25+5)
    pl.gca().set_xlim(0, 20)

    line, = ax.plot([], [])
    mouse = mouse_events(fig, line)
    mouse.connect()
    pl.show()

    return np.hstack( (np.array(mouse.xs)[:, None], np.array(mouse.ys)[:, None], np.array(mouse.orientation)[:,None], np.array(mouse.path_no_list)[:,None]) )

def gen_path(fn, save=False, environment='crosswalk1'):

    if save:
        vehicle_poses = get_way_points(environment=environment)
        np.save(fn, vehicle_poses)
    else:
        vehicle_poses = np.load(fn)
    print(vehicle_poses)

    pl.close('all')
    fig = pl.figure()  # (9,5)
    ax = fig.add_subplot(111, aspect='equal')
    for dx in range(5,15,2):
        ax.add_patch(patches.Rectangle(
            (dx+0.5, 12.5-5/2),   # (x,y)
            1,          # width
            5,          # height
            alpha=0.3,
            facecolor="#000000"
        ))
    ax.add_patch(patches.Rectangle(
        (10-5, 12.5-35/2), 10, 35, fill=False,
        linewidth=3
    ))
    pl.gca().set_ylim(0, 25)
    pl.gca().set_xlim(0, 20)

    ncolors = 11
    cmap = pl.get_cmap('jet')
    colors = cmap(np.arange(0, ncolors)/ncolors)
    levels = np.arange(0, ncolors+1)
    from matplotlib.colors import from_levels_and_colors
    cmap, norm = from_levels_and_colors(levels, colors)

    pl.scatter(vehicle_poses[:,0], vehicle_poses[:,1], c=vehicle_poses[:,3], marker='.', cmap=cmap, norm=norm, edgecolor='')
    cbar = pl.colorbar()
    cbar.set_label('Paths')
    pl.show()

def plot_path(fn):

    pl.close('all')
    fig = pl.figure(figsize=(4,2))  # (4,2), (3,1)
    ax = fig.add_subplot(111, aspect='equal')
    for dx in range(5,15,2):
        ax.add_patch(patches.Rectangle(
            (dx+0.5, 12.5-5/2),   # (x,y)
            1,          # width
            5,          # height
            alpha=0.8,
            facecolor="k"
        ))
    ax.add_patch(patches.Rectangle(
        (10-5, 12.5-35/2), 10, 35, fill=False,
        linewidth=1,
        alpha=0.8,
        facecolor="k",
        edgecolor="k"
    ))
    major_ticks = np.array([0, 5, 10, 15, 20, 25])
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    #pl.grid(linestyle='dotted')
    ax.xaxis.set_ticks(major_ticks)
    pl.gca().set_ylim(0, 25)
    pl.gca().set_xlim(0, 20)

    vehicle_poses = np.load(fn) #2b
    vehicle_poses = vehicle_poses[::20,:]

    ins = np.arange(1,22,2) #22==11 for 1D
    ins22 = [1, 3, 5, 7, 9, 14, 15, 17, 19, 21]
    ins11 = [1, 3, 5, 7, 9]
    print(ins)
    #ncolors = int(vehicle_poses[:,3].max())
    ncolors = len(ins22)
    cmap = pl.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, ncolors))

    col = -1
    amp = 0.9
    for i in range(vehicle_poses.shape[0]-1):
        if vehicle_poses[i,3] not in ins22:
            continue

        ang = vehicle_poses[i,2] + np.pi/2 #head_width=0.4, head_length=0.05, overhang=0,
        if vehicle_poses[i,3] != vehicle_poses[i+1,3]:
            col += 1
            print(col)
        pl.arrow(vehicle_poses[i,0], vehicle_poses[i,1], amp*np.cos(ang), amp*np.sin(ang), head_width=0.35, head_length=0.3, overhang=0.1, fc=colors[col,:], ec=colors[col,:])

    #cbar = pl.colorbar()
    #cbar.set_label('Paths')
    #pl.show()
    pl.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom='off',
        direction='in',
        length=2)  # labels along the bottom edge are off
    pl.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='on',      # ticks along the bottom edge are off
        right='on',         # ticks along the top edge are off
        labelleft='off',
        direction='in',
        length=2) # labels along the bottom edge are off
    #pl.savefig('/home/ransalu/PycharmProjects/GP/out/bg.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

    #implot = pl.imshow(im, extent=[0, 5, 0, 5])
    #implot = pl.imshow(im, aspect=1)
    pl.show()
    #pl.savefig('/home/ransalu/PycharmProjects/GP/out/bg.svg', format='svg', dpi=300, bbox_inches='tight', pad_inches=0)

    sys.exit()

if __name__ == "__main__":
    fn = 'generated_path.npy'
    gen_path(fn, save=True)
    plot_path(fn)