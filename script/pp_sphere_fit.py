import numpy
import glob
import os.path
import pandas
import scipy
import scipy.optimize


def detect_one_apple(cluster):
    radius_min = 0.01
    radius_max = 0.07

    center = numpy.mean(cluster, axis=0)
    x0 = numpy.concatenate([center, [0.05]])
    bounds = [tuple(numpy.concatenate([center - radius_min, [radius_min]])),
              tuple(numpy.concatenate([center + radius_max, [radius_max]]))]

    v = scipy.optimize.least_squares(
        lambda x: numpy.mean(
            numpy.abs(numpy.linalg.norm(cluster - x[:3], axis=1) - x[3])) + x[3],
        x0,
        bounds=bounds)

    cond = numpy.abs(numpy.linalg.norm(cluster - v.x[:3], axis=1) -
                     v.x[3]) <= 0.01

    if numpy.count_nonzero(cond) > cluster.shape[0] * 0.75:
        return cluster[cond]
    else:
        None


def detect_several_apple(cluster):

    radius_min = 0.01
    radius_max = 0.07
    mask = numpy.ones(cluster.shape[0], numpy.bool)

    for k in range(10):
        sub_sample = numpy.random.choice(cluster, 20)
        sub_inter = cluster[mask][sub_sample]

        # evaluate model
        center = numpy.mean(sub_inter, axis=0)
        x0 = numpy.concatenate([center, [0.05]])
        bounds = [tuple(numpy.concatenate([center - radius_min, [radius_min]])),
                  tuple(numpy.concatenate([center + radius_max, [radius_max]]))]

        v = scipy.optimize.least_squares(lambda x: numpy.mean(
            numpy.abs(
                numpy.linalg.norm(sub_inter - x[:3], axis=1) - x[3])),
            x0,
            bounds=bounds)

        cond = numpy.abs(numpy.linalg.norm(cluster - v.x[:3], axis=1) -
                         v.x[3]) <= 0.01

        if numpy.count_nonzero(cond) > best:
            model = v
            best_cond = cond
            best = numpy.count_nonzero(cond)

        if best > 100:
            pts_ok.append(model.x)
            volumes.append((4.0 * math.pi * (model.x[3] * 100)**3) / 3.0)
            mask[numpy.where(mask)[0][intercept][best_cond]] = False
            kdtree = scipy.spatial.cKDTree(point_cloud[mask], leafsize=1000)


def main():

    input = "/home/artzet_s/code/dataset/pp_conditional_euclidian_cluster"
    # input = "/home/artzet_s/code/dataset/synthetic_data/pp_labeled_conditional_euclidian_cluster"

    radius_min = 0.01
    radius_max = 0.07

    results = list()
    for filename in glob.glob(os.path.join(input, "*.txt")):
        print(filename)
        basename = os.path.basename(filename)
        date, line, position = basename[5: 15], basename[17:19], basename[21:23]
        pc = numpy.loadtxt(filename)

        apples = list()

        number_of_apple = 0
        nb_cluster = numpy.max(pc[:, 3])

        print(nb_cluster)
        for i in range(1, int(nb_cluster)):

            cluster = pc[pc[:, 3] == i][:, :3]

            point_apple = detect_one_apple(cluster)

            if point_apple is not None:
                number_of_apple += 1
                apples.append(numpy.insert(
                    point_apple, 3, values=number_of_apple, axis=1))
            else:
                print('here')
                detect_several_apple(cluster)

        if number_of_apple > 0:
            numpy.savetxt(basename + "_simple_apple.txt",
                          numpy.concatenate(apples))

        print(number_of_apple)

        results.append((basename,
                        date,
                        line,
                        position,
                        number_of_apple))

    df = pandas.DataFrame(results,
                          columns=['basename',
                                   'date',
                                   'line',
                                   'position',
                                   'nb_fruit'])

    df.to_csv('field_synthetic_labeled_measurements.csv', index=None)


if __name__ == "__main__":
    main()
