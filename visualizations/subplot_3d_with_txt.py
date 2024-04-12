import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
from matplotlib import rc
import matplotlib
from matplotlib import colors
import os
from matplotlib.transforms import Affine2D
from visualizations.attention_visualization import shift_poses
def graph_colors(dataset_name):
    global colors,graph
    if dataset_name=='h3D' :  # HUMAN ML3D GRAPH
        colors = ['sienna','sienna','sienna','sienna','orange','orange','orange','orange',"red",'red', 'red', 'indigo', 'indigo','blue', 'blue',
                  "cyan","cyan","teal","teal","cyan","cyan","cyan","cyan"]
        graph= [[0,2,5,8,0,1,4,7,0,3,6,9,12,9,14,17,19,9,13,16,18],
                     [2,5,8,11,1,4,7,10,3,6,9,12,15,14,17,19,21,13,16,18,20]]
    else: # KIT-ML GRAPH
        colors = ['black','black','black','black','orange','orange','orange','red',"red",'red','blue', 'blue', 'blue', 'blue', 'blue',
                  "cyan","cyan","cyan","cyan","cyan","cyan"]
        graph = [[0,1,2,3,3,5,6,3,8, 9,0,11,12,13,14,0,16,17,18,19],
                     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, poses, frames, use_kps_3d=range(30), down_sample_factor=1,sample = None,
                 pred=None,idxs=None,ref=None,dataset_name='h3D',intensity=None,beta_words=None,att_temp=None):

        graph_colors(dataset_name)

        fig = plt.figure(figsize=(10,8),facecolor=(0.7,0.7,0.7))
        n_joint = len(use_kps_3d)
        #fig.set_size_inches(8,8)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #fig.subplots_adjust(wspace=-0.1)
        self.paused = False
        self.poses = poses
        self.beta_words = beta_words
        self.pred_words = pred
        self.indx_w2p= idxs
        self.w_t_1 = ""
        self.sample = sample
        self.ax = fig.gca(projection='3d')
        self.colors = np.zeros(intensity.shape[1:] + (3,)) # (Tx,V,3) = (*,22,3) RGB
        intes_per_word = np.mean(intensity, axis=0)
        self.att_temp = att_temp
        self.colors[:,:,0] = intes_per_word #RED
        self.colors[:,:,1] = intes_per_word #RED
        self.colors[:,:,2] = intes_per_word #RED

        self.intensity = intensity #(Ty,Tx,V)
        self.pos_text = fig.transFigure #self.ax.figure.transFigure
        self.it = 0
        self.ref = '' if ref is None else ref
        self.s = down_sample_factor  # For down_sampling
        self.t = range(0, int(len(frames) / self.s))
        self.line = Line3D([], [], [], marker='o', color='blue', linestyle='', markersize=3)
        self.line_root = Line3D([], [], [], color='w', linestyle='--', markersize=4,linewidth=3)
        self.bones = Line3D([], [], [], color='w', linestyle='-', markersize=3)
        self.data_name = dataset_name
        self.pred_len = 0
        self.use_3dkps = use_kps_3d


        if self.sample is not  None:
            self.clip_poses = self.poses[self.sample].reshape(-1, n_joint, 3)
        else:
            self.clip_poses = self.poses.reshape(-1, n_joint, 3)
        if self.data_name=='h3D':
            MINS = self.clip_poses.min(axis=0).min(axis=0)
            height_offset = MINS[1]
            self.clip_poses[:, :, 1] -= height_offset


        r = 4 #if self.data_name=='h3D' else 1
        min_y,min_z,min_x = -r,-r,-r
        max_x,max_z,max_y = r,r,r
        self.ax.set_xlim3d([min_x, max_x])
        min_foot = self.clip_poses[:, :, 1].min()
        self.ax.set_ylim3d([min_foot, 2*max_y])
        self.ax.set_zlim3d([min_z, max_z])

        # XYZ labels---------------
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # hide the grid
        self.ax.grid = True

        #set the background color to gray
        self.ax.set_facecolor((0.7, 0.7, 0.7))

        # create a plane at the bottom of the human motion feet
        fsc =  1.25
        x = np.linspace(fsc*min_x, fsc*max_x, 10)
        y = np.linspace(fsc*min_y, fsc*max_y, 10)
        if self.data_name=='h3D':
            self.ax.view_init(elev=120, azim=-90,vertical_axis='z')
            self.ax.dist = 10 # control the distance of 3D space from the camera
            z = np.linspace(fsc * min_z, fsc * max_z, 10)
            X, Z = np.meshgrid(x, z)
            min_foot = self.clip_poses[:, :, 1].min()
            Y = fsc * min_foot * np.ones_like(X)


            self.ax.plot_surface(X, Y, Z, color='k', alpha=0.3)  # ,facecolors=illuminated_surface
        else:
            X, Y = np.meshgrid(x, y)
            min_foot = self.clip_poses[:, :, 1].min()
            Z = fsc*min_foot*np.ones_like(X)   #/fc
            self.ax.plot_surface(X,Y,Z, color='k', alpha=0.3)


        plt.axis('off')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        #Set title descriptions
        self.title = self.ax.set_title('',fontsize=18)
        self.ax.add_line(self.line)
        self.ax.add_line(self.line_root)
        self.k=0

        fps = 5
        self.animation = animation.TimedAnimation.__init__(self, fig, interval= int(1000 / fps), blit=True)
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs ):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def _draw_frame(self, i):
        fc  = 2 # skeleton scale only for better visualization
        #self.title.set_text(' Frame={} '.format(i))
        #if i>=1: self.ax.figure.remove(self.old_text)
        self.ax.figure.texts.clear()
        self.old_text =  self.ax.figure.text(0.38,0.95, ' Frame={} '.format(i),fontsize=25,weight='bold')
        canvas = self.ax.figure.canvas
        v = 0
        self.f = 0
        if self.pred_words is not None and self.indx_w2p is not None:
            _s = ' '.join(self.pred_words)
            print(_s)
            self.pos_text_ = self.ax.figure.transFigure
            # TODO CHANGE THIS PARAMETER TO VISUALIZE ITS IMPACT ON ACTION LOCALIZATION
            gate_threshold = 0.8

            # List action words
            idw_motions = np.where(self.beta_words>=gate_threshold)[0]
            L = self.att_temp[idw_motions, i, ]#:].max(axis=-1)

            for idx,_word in enumerate(self.pred_words):
                if idx in idw_motions:
                    color = "gold"  # Only for motion wordts beta>=gate_threshold
                    id_relative= np.where(idw_motions==idx)[0][0]
                    #alpha_f = L[mf] if mf==id_relative else 0
                    alpha_f = L[id_relative] if L[id_relative]>.95 else 0
                else :
                    alpha_f = 0
                    color = 'none'
                shift = 0.04*int((idx-7)>=0)

                bbox_props = dict(boxstyle="square,pad=0.0", fc=color, ec="none", alpha=alpha_f )
                _text = self.ax.figure.text(0.05,0.9-shift,_word+' ',fontsize=20,
                                            transform=self.pos_text_ if idx!=7 else self.ax.figure.transFigure,
                                            color='black',weight="bold",bbox=bbox_props)
                ex = _text.get_window_extent()
                self.pos_text_  = _text.get_transform() + Affine2D().translate(ex.width, 0)

        x = fc * self.clip_poses[self.s * i, :, 0]
        y = fc * self.clip_poses[self.s * i, :, 1]
        z = fc * self.clip_poses[self.s * i, :, 2]


        xroot = fc * self.clip_poses[:self.s * i, 0, 0]
        yroot = fc * self.clip_poses[:self.s * i, 0, 1]
        zroot = fc * self.clip_poses[:self.s * i, 0, 2]


        starts = graph[0]
        ends = graph[1]
        skeletonFrame = np.vstack([x,y,z]).T

        try:
            for line in self.skel_connect: self.ax.lines.remove(line)
            self.skel_connect = []
        except AttributeError:
            self.skel_connect = []
        except ValueError:
            pass
        c = 0
        if i > 0:
            self.points.remove()
            for line in self.skel_connect:
                line.remove()

        if self.data_name=='h3D' :
            m,n,p = [0,1,2]
            self.line_root.set_data_3d(xroot, yroot, zroot)
            self.line.set_data_3d(x, y, z)
            self.line_root.set_data_3d(xroot, yroot, zroot)

            # -------------------- Visualize Spatio-temporal Attention ---------------------------------------
            self.points = self.ax.scatter3D(x, y, z, marker='o', c=self.colors[i],
                                            s=fc*300 * self.colors[i, :, 0],alpha=0.5,edgecolor='g')
        else :
            m,n,p= [0,2,1]
            self.line_root.set_data_3d(xroot, zroot, yroot)
            self.line.set_data_3d(x, z, y)
            self.line_root.set_data_3d(xroot, zroot, yroot)
            # -------------------- Visualize Spatio-temporal Attention ---------------------------------------
            self.points = self.ax.scatter3D(x, z, y, marker='o', c=self.colors[i],
                                            s=200 * self.colors[i, :, 0])

        self.skel_connect = []
        for k, j in zip(starts, ends):
            self.skel_connect += self.ax.plot3D([skeletonFrame[k][m], skeletonFrame[j][m]], [skeletonFrame[k][n], skeletonFrame[j][n]],[skeletonFrame[k][p], skeletonFrame[j][p]],c=colors[c%(len(colors)-1)],linewidth=4.5)
            c+=1

    def new_frame_seq(self):
        return iter(range(len(self.t)))

    def _init_draw(self):
        self.line.set_data_3d([], [], [])

if __name__=='__main__':
    home_path = r"C:\Users\karim\PycharmProjects"
    dataset_name ='h3D' # 'kit'
    if dataset_name == 'kit':
    #------------------------------------ For KIT-ML ANIMATIONS -----------------------------------------------------------------
        n_joint = 21
        sample = 43 # 1437
        # data = np.load("/home/karim/semMotion/datasets/kitmld_anglejoint_2016_30s_final_cartesian.npz", allow_pickle=True)
        # normalized_poses = np.load("/home/karim/kitmld/kit_normalized_poses.npz",allow_pickle=True)["normalized_poses"]
        kit_data = np.load(home_path+"/HumanML3D/kit_with_splits_2023.npz", allow_pickle=True)
        data = np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in kit_data['kitmld_array']], dtype=object)
        descriptions = kit_data['descriptions']
        #data = kitmld['kitmld_array']

    #---------------------------------- For Human-ML3D ANIMATIONS --------------------------------------------------------------
    if dataset_name == 'h3D':
        n_joint = 22
        sample = 43
        #normalized_poses =  np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in data['kitmld_array']], dtype=object)
        h3d_data = np.load(home_path + "/HumanML3D/all_humanML3D.npz",allow_pickle=True)
        data = np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in h3d_data['kitmld_array']], dtype=object)
        # -------------- Mean/Std Normalization -------------------------------------------
        data[sample] = shift_poses(data[sample])
        descriptions = h3d_data["old_desc"] #[S for S in data["descriptions"]]

        # avoid skipping some frames due to memory limit
        matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

    Tmax = 700*4
    ani = SubplotAnimation(data,frames=range(len(data[sample])), use_kps_3d=range(n_joint),
                           down_sample_factor=2,sample=sample,pred=descriptions[sample][0],
                           idxs=[7*i for i in range(9)],dataset_name='h3D')

    # """"save animation
    rc('animation', html='jshtml')
    name_file = f"test_sample_{sample}.gif"
    ani.save(name_file)
    print("Animation saved in "+name_file)#os.getcwd()+


    display_static_pose= False
    if display_static_pose:
        # -------------------------------- DISPLAY THE POSE GRAPH STRUCTURE ------------------------------------

        import numpy as np
        import matplotlib.pyplot as plt
        colors = ['sienna', 'sienna', 'sienna', 'sienna', 'orange', 'orange', 'orange', 'orange', "red", 'red', 'red',
                  'indigo', 'indigo', 'blue', 'blue', "cyan", "cyan", "teal", "teal", "cyan", "cyan", "cyan", "cyan"]

        # HUMAN ML3d GRAPH
        graph = [[0, 2, 5, 8, 0, 1, 4, 7, 0, 3, 6, 9, 12, 9, 14, 17, 19, 9, 13, 16, 18],
                 [2, 5, 8, 11, 1, 4, 7, 10, 3, 6, 9, 12, 15, 14, 17, 19, 21, 13, 16, 18, 20]]
        data = np.load("/home/karim/PycharmProjects/HumanML3D/all_humanML3D.npz", allow_pickle=True)

        n_joint = 22
        id_sample = 151 # best index to see the structure correctly
        poses_arry =  data['kitmld_array'][id_sample].reshape(-1, n_joint * 3)
        f = 5
        use_kps_3d = range(n_joint)
        skeletonFrame = poses_arry[f].reshape(n_joint, 3)
        x = skeletonFrame[:, 0]
        y = skeletonFrame[:, 1]
        z = skeletonFrame[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=120, azim=-90, vertical_axis='z')
        # r = 2
        # ax.set_zlim3d([-r, r])
        # ax.set_xlim3d([-r//3, r//3])
        # ax.set_ylim3d([-r//3, r//3])
        #------------------ DISPLAY JOINT INDEXES ----------------------------------------------²
        for k in use_kps_3d:
            ax.scatter(x[k], y[k], z[k]) # Y is the Z axis only for KIT
            ax.text(x[k], y[k], z[k], str(k),fontsize=12,fontweight='bold',color="black")

        # -------- CONNECT JOINTS ---------------------------------------------------------------
        starts = graph[0]
        ends = graph[1]
        skeletonFrame = np.vstack([x, y, z]).T
        c = 0
        skel_connect = []
        for k, j in zip(starts, ends):
            skel_connect += ax.plot3D([skeletonFrame[k][0], skeletonFrame[j][0]],
                                                [skeletonFrame[k][1], skeletonFrame[j][1]],
                                                zs=[skeletonFrame[k][2], skeletonFrame[j][2]], c=colors[c % (len(colors) - 1)],
                                                linewidth=4.5)
            # for line in self.skel_connect:line.set_alpha(alphas[i%5])
            c += 1

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()