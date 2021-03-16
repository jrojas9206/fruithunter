import sys
import glob
import numpy
from os.path import join, exists
from randlanet.RandLANet import Network
from randlanet.tester_Semantic3D import ModelTester
from randlanet.helper_ply import read_ply
from randlanet.helper_tool import Plot
from randlanet.helper_tool import DataProcessing as DP
from randlanet.helper_tool import ConfigAppleTreeField as cfg_field
from randlanet.helper_tool import ConfigAppleTreeSynthetic as cfg_synthetic

cfg = None

import tensorflow as tf
import numpy as np
import pickle, argparse, os

parser = argparse.ArgumentParser("Randla-NET applied to tree's organ detection")

class AppleTree:
    def __init__(self, protocol, path2dataset=None):
        """
        protocol : "synthetic_HiHiRes", "field", "field_only_xyz"
        """
        self.name = 'AppleTree'

        if(protocol == "synthetic_HiHiRes"):
            self.path = "/gpfswork/rech/wwk/uqr22pt/data_RandLa-Net/apple_tree_synthetic_HiHiRes" if path2dataset is None else path2dataset
        elif(protocol == "field_only_xyz"):
            self.path = "/gpfswork/rech/wwk/uqr22pt/data_RandLa-Net/apple_tree_field_only_xyz" if path2dataset is None else path2dataset
        elif(protocol == "field" ):
            self.path = "/gpfswork/rech/wwk/uqr22pt/data_RandLa-Net/apple_tree_field" if path2dataset is None else path2dataset
        else:
            exit("wrong protocol")
        self.label_to_names = {0: 'unlabeled',
                               1: 'apple'}
                               
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        if protocol == "synthetic_HiHiRes" or protocol == "field_only_xyz" or "field":
            # Path to the training files
            self.training_folder = join(self.path, "training")
            filenames = glob.glob(join(self.training_folder, "*.ply"))
            # Generic selection for train and validation
            # Select 90% of the training files for train the model 
            # and 10% to validate 
            train_p_file = int(len(filenames)*0.9)
            self.train_files = filenames[:train_p_file]
            self.val_files = filenames[train_p_file:]
        else:
            validation_fold = 1
            self.train_files = []

            for i in range(1, 6):
                if i != validation_fold:
                    self.training_folder = join(self.path, "fold_{}".format(i))
                    self.train_files += glob.glob(join(self.training_folder, "*.ply"))

            self.val_folder = join(self.path, "fold_{}".format(validation_fold))
            self.val_files = glob.glob(join(self.val_folder, "*.ply"))


        self.test_folder = join(self.path, "test")
        self.test_files = glob.glob(join(self.test_folder, "*.ply"))

        # if protocol == "synthetic_HiHiRes":
        #     self.test_files = glob.glob(join(self.test_folder, "*.ply"))
        # elif protocol == "HiRes":
        #     self.test_files = glob.glob(join(self.test_folder, "*high_quality*.ply"))
        # else:
        #     self.test_files = glob.glob(join(self.test_folder, "*2018*.ply"))
        #     self.test_files += glob.glob(join(self.test_folder, "*2019*.ply"))
        #     self.test_files = [f for f in self.test_files if "high_quality" not in f]

        self.val_split = 1


        print("Train files :")
        print(*self.train_files, sep='\n')
        print("Validation files :")
        print(*self.val_files, sep='\n')
        print("Test files :")
        print(*self.test_files, sep='\n')

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        self.ascii_files = dict()
        for test_file in self.test_files:
            self.ascii_files[os.path.basename(test_file)] = os.path.basename(test_file)[:-4] + ".labels"


        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
        
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pt_weight = 1
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = tf.concat([transformed_xyz, rgb], axis=-1)
        return stacked_features

    def init_input_pipeline(self, mode="train"):
        print('Initiating input pipelines')
        if(mode=="train"):
            cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
            gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
            gen_function_val, _, _ = self.get_batch_gen('validation')
            gen_function_test, _, _ = self.get_batch_gen('test')
            self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
            self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

            self.batch_train_data = self.train_data.batch(cfg.batch_size)
            self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
            self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
            map_func = self.get_tf_mapping()

            self.batch_train_data = self.batch_train_data.map(map_func=map_func)
            self.batch_val_data = self.batch_val_data.map(map_func=map_func)
            self.batch_test_data = self.batch_test_data.map(map_func=map_func)

            self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
            self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
            self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

            iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
            self.flat_inputs = iter.get_next()
            self.train_init_op = iter.make_initializer(self.batch_train_data)
            self.val_init_op = iter.make_initializer(self.batch_val_data)
            self.test_init_op = iter.make_initializer(self.batch_test_data)
        elif(mode=="test"):
            cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
            gen_function_test, gen_types, gen_shapes = self.get_batch_gen('test')
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
            map_func = self.get_tf_mapping()
            self.batch_test_data = self.batch_test_data.map(map_func=map_func)
            self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)
            iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
            self.flat_inputs = iter.get_next()
            self.train_init_op = iter.make_initializer(self.batch_test_data)
            self.test_init_op = iter.make_initializer(self.batch_test_data)

def launch_training(protocol, inputDir, parameters=None):
    GPU_ID = parameters["gpu"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = parameters["mode"]

    dataset = AppleTree(protocol, path2dataset=inputDir)
    dataset.init_input_pipeline(mode=Mode)
    
    if Mode == 'train' and not parameters["restoreTrain"]:
        model = Network(dataset, cfg)
        model.train(dataset)

    elif Mode == "train" and parameters["restoreTrain"]:
        model = Network(dataset, cfg, restore_snap=parameters["model_path"])
        model.train(dataset)

    elif Mode == "validation":

        cfg.saving = False
        if parameters["model_path"] is not 'None':
            chosen_snap = parameters["model_path"]
        else:
            chosen_snapshot = -1
            # logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            # chosen_folder = logs[-1]
            chosen_folder = parameters["outputDir"]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        
        model = Network(dataset, cfg, restore_snap=chosen_snap)
        model.evaluate(dataset)

    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if parameters["model_path"] is not 'None':
            chosen_snap = parameters["model_path"]
        else:
            chosen_snapshot = -1
            # logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            # chosen_folder = logs[-1]
            chosen_folder = parameters["outputDir"]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        
        print(chosen_snap)
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])

def train_field(inputDir, outputDir, parameters=None):
    global cfg
    cfg = cfg_field
    cfg.saving_path = outputDir
    print(cfg.saving_path, flush=True)
    launch_training("field", inputDir, parameters=parameters)

def train_field_only_xyz(inputDir, outputDir, parameters=None):
    global cfg
    cfg = cfg_field
    cfg.saving_path = outputDir
    print(cfg.saving_path, flush=True)
    launch_training("field_only_xyz", inputDir, parameters=parameters)

def train_synthetic_HiHiRes(inputDir, outputDir, parameters=None):
    global cfg
    cfg = cfg_synthetic
    cfg.saving_path = outputDir
    print(cfg.saving_path, flush=True)    
    launch_training("synthetic_HiHiRes", inputDir, parameters=parameters)

if __name__ == '__main__':
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--inputDir', type=str, help="Path to the folder with the train/test/validation & the input folders", default=None)
    parser.add_argument("--outputDir", type=str, help="Path to the output folder", default="./output/")
    parser.add_argument("--protocol", type=str, help="Measurement protocol /synthetic/field_xyz/filed", default="synthetic")
    parser.add_argument("--restoreTrain", type=bool, help="Restore training True/False", default=False)
    args = parser.parse_args()
    # Parameters copy to be able to use launch_training in other projects 
    param = {"gpu":args.gpu, "mode":args.mode, "model_path":args.model_path, "path2data":args.inputDir, 
             "path2output": args.outputDir, "protocol":args.protocol, "restoreTrain":args.restoreTrain} 
    print("-> RandLA-NET")
    print(" -> GPU[ID]: %i" %args.gpu)
    print(" -> Mode[train/test/vis]: %s" %args.mode)
    print(" -> Output[Path]: %s" %args.outputDir)
    print("  -> Status: %s" %("OK" if os.path.isdir(args.inputDir) else "Is going to be created")) 
    print(" -> Input [Path]: %s" %("None" if args.inputDir is None else args.inputDir))
    print("  -> Status: %s"%("OK" if os.path.isdir(args.inputDir) else "Error"))
    print(" -> Protocol: %s" %args.protocol)
    if(args.protocol == "synthetic"):
        train_synthetic_HiHiRes(args.inputDir, args.outputDir, parameters=param)
    elif(args.protocol == "field_only_xyz"):
        train_field_only_xyz(args.inputDir, args.outputDir, parameters=param)
    elif(args.protocol == "field"):
        train_field(args.inputDir, args.outputDir, parameters=param)
    else:
        print("-> Error: Unknow options, please execute the following command and verify the defined args. \n$$>python main_apple_tree.py -h")
    print("-> END")
    sys.exit(0)
    