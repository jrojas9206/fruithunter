import glob
import numpy
import os.path


input_folder = "/gpfswork/rech/wwk/uqr22pt/field_afef_apple_tree_filtered"

filenames = glob.glob("{}/*.txt".format(input_folder))
vmax, vmin = list(), list()
for i, filename in enumerate(filenames):
    print("{}/{} : {}".format(i, len(filenames), filename))

    npy_filename = filename[:-3] + 'npy'
    if os.path.exists(npy_filename):
        data_label = numpy.load(npy_filename)
    else:
        data_label = numpy.loadtxt(filename)
        numpy.save(npy_filename, data_label)


    vmax.append(numpy.max(data_label, axis=0))
    vmin.append(numpy.min(data_label, axis=0))

vmax = numpy.max(numpy.array(vmax), axis=0)
vmin = numpy.min(numpy.array(vmin), axis=0)

arr = numpy.stack([vmin, vmax], axis=0)
print(arr, arr.shape)
numpy.savetxt("mean_data.txt", arr)
