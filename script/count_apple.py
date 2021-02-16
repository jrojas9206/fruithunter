import numpy
import glob
import os.path
import pandas
from multiprocessing import Process
from joblib import Parallel, delayed


#import openalea.fruithunter.multiprocess


def count_apple(filename):

    print(filename)
    basename = os.path.basename(filename)
    date, line, position = basename[5: 15], basename[17:19], basename[21:23]
    pc = numpy.loadtxt(filename)

    nb_point = pc.shape[0]
    nb_point_of_fruit = numpy.count_nonzero(pc[:, 3] >= 1)

    nb_fruit = numpy.max(pc[:, 3])

    nb_fruit_floor = 0
    for i in range(int(nb_fruit)):
        z = numpy.mean(pc[pc[:, 3] == i][:, 2])
        if z < -1.70:
            nb_fruit_floor += 1

    nb_fruit_tree = nb_fruit - nb_fruit_floor

    return (basename,
            date,
            line,
            position,
            nb_fruit,
            nb_fruit_tree,
            nb_fruit_floor,
            nb_point,
            nb_point_of_fruit)


def main(argv):

    input = "/mnt/pp_conditional_euclidian_cluster_075"
    filenames = glob.glob(os.path.join(input, "*.txt"))

    results = Parallel(n_jobs=4)(delayed(count_apple)(i) for i in filenames)

    #results = openalea.fruithunter.multiprocess.multiprocess_function(
    #    count_apple,
    #    filenames,
    #    nb_process=14)

    results = numpy.array(results)
    print(results)
    df = pandas.DataFrame(results,
                          columns=['basename',
                                   'date',
                                   'line',
                                   'position',
                                   'nb_fruit',
                                   'nb_fruit_tree',
                                   'nb_fruit_floor',
                                   'nb_point',
                                   'nb_point_of_fruit'])

    df.to_csv('field_measurements.csv', index=None)


if __name__ == "__main__":
    main()
