import sklearn
import numpy
import scipy
import math

def filtering(point_cloud, cluster_eps, cluster_size):

    db = sklearn.cluster.DBSCAN(min_samples=1,
                                eps=cluster_eps,
                                metric="euclidean",
                                algorithm="ball_tree").fit(point_cloud)

    nb_cluster = max(db.labels_)

    for i in range(nb_cluster + 1):
        cond = db.labels_ == i
        if numpy.count_nonzero(cond) < cluster_size:
            db.labels_[cond] = -1

    # return point_cloud[db.labels_ >= 0]
    return db.labels_ >= 0


def clustering(pc, min_samples, eps, filter_cluster_size=None, filter_cluster_eps=None):

    labels = -1 * numpy.ones((pc.shape[0], 1), dtype=numpy.int)
    cond = numpy.full((pc.shape[0], ), True)

    if pc.shape[0] > 0:

        if filter_cluster_size is not None and filter_cluster_eps is not None:
            cond = filtering(pc, filter_cluster_eps, filter_cluster_size)
        
        pc_filtered = pc[cond]
        if pc_filtered.shape[0] > 0:
            db = sklearn.cluster.DBSCAN(min_samples=min_samples,
                                        eps=eps,
                                        metric="euclidean",
                                        algorithm="ball_tree").fit(pc_filtered)

            labels[cond, 0] = db.labels_

    return numpy.concatenate([pc, labels], axis=1)


def measure_volume_cluster(cluster, radius_min=0.01, radius_max=0.08):

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

    thruth_rate = (numpy.count_nonzero(cond) * 100) / cluster.shape[0]
    radius = v.x[3]

    volume = (4 * math.pi * radius ** 3) / 4

    return volume, thruth_rate


def measure_cluster(cluster):

    nb_cluster = 0
    mean_volume = 0
    total_volume = 0

    if cluster.shape[0] == 0:
        return nb_cluster, mean_volume, total_volume

    pc = cluster[:, :3]
    labels = cluster[:, 3]

    # Get number of cluster
    nb_cluster = int(max(labels))

    # #Get radius size of the 10 best cluster looking as a ball
    # lpr = list()
    # for i in range(nb_cluster + 1):
    #     lpr.append(measure_volume_cluster(pc[labels == i]))
    # # lpr.sort(key=lambda x: - x[1])

    # if len(lpr) > 1:
    #     mean_volume = numpy.mean(numpy.array(lpr), axis=1)[1]
    #     total_volume = numpy.sum(numpy.array(lpr), axis=1)[1]

    return nb_cluster, mean_volume, total_volume