import numpy as np
import matplotlib.pyplot as pl
from spherecluster import VonMisesFisherMixture
from spherecluster import von_mises_fisher_mixture
from sklearn.cluster import DBSCAN


class TimeLocDB():
    def __init__(self, attribs=['TrID', 'x', 'y', 'vel', 'angle']):
        self.cell_time_keys_ = np.empty((0, 2))
        self.cell_time_vals_ = []
        self.attribs_ = attribs

    def insert(self, cell, time, vals):
        loc_time_exists = np.logical_and((self.cell_time_keys_[:, 0] == cell), (self.cell_time_keys_[:, 1] == time))
        loc_time_indx = np.where(loc_time_exists == True)[0]
        if loc_time_indx >= 0:
            # this loc-time exists
            self.cell_time_vals_[loc_time_indx[0]].append(vals)
        else:
            # create new exits
            self.cell_time_keys_ = np.vstack((self.cell_time_keys_, np.array([cell, time])))
            self.cell_time_vals_.append([vals])

    def read(self, cell='all', time='all'):
        print(cell, time)

        if np.logical_and(cell == 'all', time == 'all'):
            loc_time_exists = np.array([True] * self.cell_time_keys_.shape[0])
        else:
            if cell == 'all':
                self.cell_time_keys_ = self.cell_time_keys_.astype(np.int32)
                loc_time_exists = self.cell_time_keys_[:, 1] == time
                # loc_time_exists1 = self.cell_time_keys_[:, 1] <= time
                # loc_time_exists2 = self.cell_time_keys_[:, 1] > time -25
                # loc_time_exists = np.logical_and(loc_time_exists1,loc_time_exists2)
                # print('sum:',np.sum(loc_time_exists))

            elif time == 'all':
                loc_time_exists = self.cell_time_keys_[:, 0] == cell
            else:
                loc_time_exists = np.logical_and((self.cell_time_keys_[:, 0] == cell),
                                                 (self.cell_time_keys_[:, 1] == time))

        try:
            loc_time_indx = np.where(loc_time_exists == True)[0]

            vals = np.empty((0, len(self.cell_time_vals_[0][0])))
            for i in loc_time_indx:
                vals = np.vstack((vals, np.array(self.cell_time_vals_[i])))
        except:
            vals = None
            print('Cant read! (cell,time)=(' + str(cell) + '-' + str(time) + ') does not exist.')

        return vals

    def pop(self, cell, time):
        print('Method not implemented')

def read_sim(c_no, f_in='unimodal_sim1.npy', f_out=None, break_when=10e10):

    def get_mesh(vals=[0,20,5,0,25,5]):
        #:param vals: grid paras [x_min, x_max, x_resolution, y_min, y_max, y_resolution]
        if vals[1]/vals[2] == 0:
            extra1 = 0
        else:
            extra1 = vals[2]
        if vals[4] / vals[5] == 0:
            extra2 = 0
        else:
            extra2 = vals[5]
        xx, yy = np.meshgrid(np.arange(vals[0],vals[1]+extra1,vals[2]), np.arange(vals[3],vals[4]+extra2,vals[5]))
        cell_nos = np.arange(xx.size).reshape(xx.shape)
        yy, cell_nos = np.flipud(yy), np.flipud(cell_nos)
        mesh = np.vstack((xx.ravel(), yy.ravel())).T
        return mesh, xx, yy, cell_nos

    def get_cell_no(xx, yy, cell_nos, x_t=np.array([1.5,2.5])):
        """
        :param x_t: check the cell number for this point: np.array([x,y])
        :return: cell number w.r.t. origin bottom-left
        """

        try:
            grid_pos_x = np.where((xx[0, :] - x_t[0]) <= 0)[0][-1]
            grid_pos_y = np.where((yy[:, 0] - x_t[1]) <= 0)[0][0]
        except:
            print('x_t out of grid. Please extend the grid. x_t=', x_t)

        cell_no = cell_nos[grid_pos_y, grid_pos_x]

        #print('xx\n', xx)
        #print('yy\n', yy)
        #print('cell_nos\n', cell_nos)

        return cell_no

    def get_vels(line):
        diffs = line[1:] - line[:-1]
        speed = np.sqrt(np.sum(diffs[:,:2]**2, axis=1))
        dirs = np.arctan2(diffs[:,1], diffs[:,0])

        return speed, dirs

    res = 5
    mesh, xx, yy, cell_nos = get_mesh([0,20,res,-5,30,res])

    f_sim = np.load(f_in) # x, y, theta, path_no
    f_sim = np.hstack((0*np.zeros((f_sim.shape[0],1)), f_sim)) # time, x, y, theta, path_no

    db = TimeLocDB(attribs=['TrID', 'x', 'y', 'vel', 'angle'])

    for path_id in np.unique(f_sim[:,4]):
        path_id_locs = np.where(f_sim[:, 4] == path_id)[0]
        path_data = f_sim[path_id_locs,:]
        vels, dirs = get_vels(path_data[:, 1:3])
        #print("dirs", path_data.shape, vels.shape)

        for i in range(1,path_data.shape[0]-1):
            cell_no = get_cell_no(xx, yy, cell_nos, x_t=path_data[i, 1:3])
            db.insert(cell=cell_no, time=path_data[i,0], vals=[path_data[i,4], path_data[i, 1], path_data[i, 2], vels[i], dirs[i]])

    if f_out is not None:
        data = db.read(cell=c_no, time='all') #c_no
        #print(xx,yy,cell_nos,mesh)
        #pl.scatter(mesh[:,0], mesh[:,1])
        #pl.scatter(data[:,1], data[:,2],marker='*')
        #pl.show()
        data = data.astype(np.float)
        np.savez(f_out, data=data, xx=xx, yy=yy)
        #print(str(f_out) + ' saved!')

def sim_unimodal():

    to_save = []
    data_cells = []

    # Data pre-processing
    for i in range(40):
        read_sim(i, f_in='datasets/unimodal_sim1.npy', f_out='datasets/transformed/unimodal_sim1_cell' + str(i))

    # Angles to query
    Thq = np.linspace(-np.pi, np.pi, 360)[:, None]
    Xq = np.hstack((np.cos(Thq), np.sin(Thq)))

    # Fit one cell at a time
    for i in range(40):
        print('cell no={}'.format(i))

        try:
            # Read data
            read_data = np.load('datasets/transformed/unimodal_sim1_cell' + str(i) + '.npz')
            data, xx, yy = read_data['data'], read_data['xx'], read_data['yy']
            if data.shape[0] <= 1:
                continue

            # Data
            Th = data[:, 4][:, None]
            X = np.hstack((np.cos(Th), np.sin(Th)))

            # Von Mises clustering (soft)
            vmf_soft = VonMisesFisherMixture(n_clusters=1, posterior_type='soft', n_init=20)
            vmf_soft.fit(X)
            y0 = np.exp(von_mises_fisher_mixture._vmf_log(Xq, vmf_soft.concentrations_[0], vmf_soft.cluster_centers_[0]))
            y = y0*vmf_soft.weights_[0]

            # Query
            yq = np.array(y)[:, None]
            to_save.append(yq)
            data_cells.append(i)

            # Plot
            pl.figure(figsize=(15, 4))

            pl.subplot(131)
            mesh = np.vstack((xx.ravel(), yy.ravel())).T
            pl.scatter(mesh[:, 0], mesh[:, 1], c='k', marker='.')
            pl.scatter(data[:, 1], data[:, 2], c=data[:, 0], marker='*', cmap='jet')
            pl.colorbar()
            pl.xlim([0, 20])
            pl.ylim([-5, 30])
            pl.title('data')

            pl.subplot(132)
            pl.scatter(Xq[:, 0], Xq[:, 1], c=y0[:], cmap='jet')
            pl.colorbar()
            pl.scatter(X[:, 0] * 0.9, X[:, 1] * 0.9, c='k', marker='+')
            pl.title('data and extimated distribution')

            pl.subplot(133, projection='polar')
            pl.polar(Thq, yq)
            pl.title('polar plot')
            #pl.show()
            pl.savefig('outputs/unimodal_sim1_cell{}'.format(i))
        except:
            print(' skipped...')
            continue

def sim_multimodal():

    to_save = []
    data_cells = []

    # Data pre-processing
    for i in range(40):
        read_sim(i, f_in='datasets/multimodal_sim2.npy', f_out='datasets/transformed/multimodal_sim2_cell' + str(i))

    # Angles to query
    Thq = np.linspace(-np.pi, np.pi, 360)[:, None]
    Xq = np.hstack((np.cos(Thq), np.sin(Thq)))

    # Fit one cell at a time
    for i in range(40):
        print('\ncell no={}'.format(i))

        try:
            # Read data
            read_data = np.load('datasets/transformed/multimodal_sim2_cell' + str(i) + '.npz')
            data, xx, yy = read_data['data'], read_data['xx'], read_data['yy']
            if data.shape[0] <= 1:
                continue

            # Data
            Th = data[:, 4][:, None]
            X = np.hstack((np.cos(Th), np.sin(Th)))
            db = DBSCAN().fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)
            print("n_clusters_={}, labels={}".format(n_clusters_,unique_labels))
            for k in unique_labels:
                if k == -1:  # noisy samples
                    continue
                class_member_mask = (labels == k)
                xy = X[class_member_mask & core_samples_mask]
                if k == 0:
                    db_centers = np.mean(xy, axis=0)[None, :]
                else:
                    db_centers = np.concatenate((db_centers, np.mean(xy, axis=0)[None, :]), axis=0)
            print("db_centers=", db_centers)

            # TBD: "NOTE:: play with max_iter if you get the denom=inf error"

            # Mixture of von Mises Fisher clustering (soft)
            vmf_soft = VonMisesFisherMixture(n_clusters=n_clusters_, posterior_type='soft', init=db_centers, n_init=1,
                                             verbose=True, max_iter=20)
            vmf_soft.fit(X)

            y = 0
            for cn in range(n_clusters_):
                y += vmf_soft.weights_[cn] * np.exp(
                    von_mises_fisher_mixture._vmf_log(Xq, vmf_soft.concentrations_[cn], vmf_soft.cluster_centers_[cn]))
            yq = np.array(y)[:, None]
            to_save.append(yq)
            data_cells.append(i)

            # Plot
            pl.figure(figsize=(15, 4))

            pl.subplot(131)
            mesh = np.vstack((xx.ravel(), yy.ravel())).T
            pl.scatter(mesh[:, 0], mesh[:, 1], c='k', marker='.')
            pl.scatter(data[:, 1], data[:, 2], c=data[:, 0], marker='*', cmap='jet')
            pl.colorbar()
            pl.xlim([0, 20])
            pl.ylim([-5, 30])
            pl.title('data')

            pl.subplot(132)
            pl.scatter(Xq[:, 0], Xq[:, 1], c=yq[:], cmap='jet')
            pl.colorbar()
            pl.scatter(X[:, 0] * 0.9, X[:, 1] * 0.9, c='k', marker='+')
            pl.title('data and extimated distribution')

            pl.subplot(133, projection='polar')
            pl.polar(Thq, yq)
            pl.title('polar plot')
            pl.savefig('outputs/multimodal_sim2_cell{}'.format(i))
            #pl.show()
        except:
            print(' skipped...')
            continue

if __name__ == "__main__":
    sim_multimodal()
    #sim_unimodal()