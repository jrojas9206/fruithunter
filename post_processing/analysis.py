import numpy
import sklearn
import matplotlib
import matplotlib.pyplot
import pandas
import os
import math

def plot_point(axis, df_colum_1, df_colum_2, color='r'):
    x = df_colum_1.to_numpy()
    X = numpy.array([x]).T
    y = df_colum_2.to_numpy()

    axis.plot(X, y, c=color, marker='.', linestyle='', markersize=20)

def plot_R2_linear_model(axis, df_colum_1, df_colum_2, color='r', xlabel=None, ylabel=None, plot_point=True):

    x = df_colum_1.to_numpy()
    X = numpy.array([x]).T
    y = df_colum_2.to_numpy()

    if plot_point:
        axis.plot(X, y, c=color, marker='.', linestyle='', markersize=20)

    # reg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X, y)
    # R2 = numpy.round(reg.score(X, y), decimals=2)
    # val = reg.predict(numpy.array([[0], [1]]))
    # coef = numpy.round((val[1] - val[0]), decimals=2)
    # axis.plot(X, reg.predict(X), 'b--',
    #           label="R² = {:.2f} coef: {}".format(R2, coef))

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X, y)
    R2 = numpy.round(reg.score(X, y), decimals=2)
    val = reg.predict(numpy.array([[0], [1]]))
    coef = numpy.round((val[1] - val[0]), decimals=2)
    axis.plot(X, reg.predict(X), 'k--',
              label="R² = {:.2f} coef: {}".format(R2, coef))

    m = numpy.max(X)
    axis.plot([0, m], [0, m], 'c')

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, fontsize=40)

    if xlabel is None:
        axis.set_xlabel(df_colum_1.name, fontsize=40)
    else:
        axis.set_xlabel(xlabel, fontsize=40)

    if ylabel is None:
        axis.set_ylabel(df_colum_2.name, fontsize=40)
    else:
        axis.set_ylabel(ylabel, fontsize=40)


def save_synthetic(df_comparison, output_dir, title="default"):

    filename = os.path.join(output_dir, "measurements.csv")
    df_comparison.to_csv(filename)

    if df_comparison.shape[0] == 0:
        return None

    matplotlib.pyplot.rcParams["figure.figsize"] = (15, 15)

    fig = matplotlib.pyplot.figure()
    axis = matplotlib.pyplot.axes()

    fig.suptitle(title, fontsize=32)

    plot_R2_linear_model(axis,
                         df_comparison['number_of_apple'],
                         df_comparison['nb_cluster'],
                         xlabel='Number of actual fruits',
                         ylabel='Number of detected clusters')

    filename = os.path.join(output_dir, "figure_cluster_vs_apple " + title + ".png")
    matplotlib.pyplot.savefig(filename)



def save_field_in_one(df_comparison, output_dir, title="default"):
    filename = os.path.join(output_dir, "measurements.csv")
    df_comparison.to_csv(filename)

    if df_comparison.shape[0] == 0:
        return None

    matplotlib.pyplot.rcParams["figure.figsize"] = (15, 15)

    fig = matplotlib.pyplot.figure()
    axis = matplotlib.pyplot.axes()

    df = df_comparison[df_comparison['position'].isin([1, 5, 6, 10, 11, 15, 16, 20])]
    plot_point(axis, 
               df['harvest_nb_fruit'],
               df['nb_cluster'],
               color='r')
    
    df = df_comparison[df_comparison['position'].isin([2, 4, 7, 9, 12, 14, 17, 19, 3, 8, 13, 18])]
    plot_point(axis, 
               df['harvest_nb_fruit'],
               df['nb_cluster'],
               color='b')

    plot_R2_linear_model(axis,
                         df_comparison['harvest_nb_fruit'],
                         df_comparison['nb_cluster'],
                         xlabel='Number of actual fruits',
                         ylabel='Number of detected clusters',
                         plot_point=False)


    filename = os.path.join(output_dir, "figure_cluster_vs_apple_" + title + ".png")
    matplotlib.pyplot.savefig(filename)


def save_field(df_comparison, output_dir, title="default"):

    filename = os.path.join(output_dir, "measurements.csv")
    df_comparison.to_csv(filename)

    if df_comparison.shape[0] == 0:
        return None

    matplotlib.pyplot.rcParams["figure.figsize"] = (15, 15)

    # fig = matplotlib.pyplot.figure()
    # axis = matplotlib.pyplot.axes()

    # plot_R2_linear_model(axis,
    #                      df_comparison['poids_moy_1_fruit'] * 100,
    #                      df_comparison['mean_volume'],
    #                      xlabel='Mean weight per fruit',
    #                      ylabel='Mean volume per cluster')

    # filename = os.path.join(output_dir, "figure_mean_radius_vs_poids_moy_1_fruit" + title + ".png")
    # matplotlib.pyplot.savefig(filename)

    # fig = matplotlib.pyplot.figure()
    # axis = matplotlib.pyplot.axes()

    # plot_R2_linear_model(axis,
    #                      df_comparison['harvest_weight_fruit'] * 100,
    #                      df_comparison['total_volume'],
    #                      xlabel='Total fruit weight',
    #                      ylabel='Total cluster volume')

    # filename = os.path.join(output_dir, "figure_total_volume_vs_total_fruit" + title + ".png")
    # matplotlib.pyplot.savefig(filename)


    fig = matplotlib.pyplot.figure()
    axis = matplotlib.pyplot.axes()

    plot_R2_linear_model(axis,
                         df_comparison['harvest_nb_fruit'],
                         df_comparison['nb_cluster'],
                         xlabel='Number of actual fruits',
                         ylabel='Number of detected clusters')

    filename = os.path.join(output_dir, "figure_cluster_vs_apple_" + title + ".png")
    matplotlib.pyplot.savefig(filename)


def R2_value(df_colum_1, df_colum_2):
    X = numpy.array([df_colum_1.to_numpy()]).T
    y = df_colum_2.to_numpy()

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X, y)
    R2 = max(0, numpy.round(reg.score(X, y), decimals=2))

    return R2


def r2_rmse_opti_value(df_colum_obs, df_colum_pred):

    obs = df_colum_obs.to_numpy()
    pred = df_colum_pred.to_numpy()
    
    r2 = sklearn.metrics.r2_score(obs, pred)
    r2 = max(0, numpy.round(r2, decimals=2))

    rmse = math.sqrt(sklearn.metrics.mean_squared_error(obs, pred))
    rmse = numpy.round(rmse, decimals=2)

    opti_value = r2 - (1 - rmse / numpy.mean(obs))
    opti_value = numpy.round(opti_value, decimals=2)

    return r2, rmse, opti_value


def R2_value_by_name(df_comparison, column_name_1, column_name_2, positions=None):

    if positions is not None:
        df = df_comparison[df_comparison['position'].isin(positions)]

        return R2_value(df[column_name_1],
                        df[column_name_2])
    else:
        return R2_value(df_comparison[column_name_1],
                        df_comparison[column_name_2])


def field_comparison(df_harvest, df_measurements):

    df_comparison = pandas.merge(df_harvest,
                                 df_measurements,
                                 left_on=['line', 'position', 'year'],
                                 right_on=['line', 'position', 'year'])

    fruit_all_r2, fruit_all_rmse, fruit_all_opti_value = r2_rmse_opti_value(
        df_comparison['harvest_nb_fruit'],
        df_comparison['nb_cluster'])

    return fruit_all_r2, fruit_all_rmse, fruit_all_opti_value

    # fruit_close_R2 = R2_value_by_name(
    #     df_comparison,
    #     'nb_cluster',
    #     'harvest_nb_fruit',
    #     positions=[1, 5, 6, 10, 11, 15, 16, 20])

    # fruit_middle_R2 = R2_value_by_name(
    #     df_comparison,
    #     'nb_cluster',
    #     'harvest_nb_fruit',
    #     positions=[2, 4, 7, 9, 12, 14, 17, 19])

    # fruit_far_R2 = R2_value_by_name(
    #     df_comparison,
    #     'nb_cluster',
    #     'harvest_nb_fruit',
    #     positions=[3, 8, 13, 18])

    # # df = df_comparison[df_comparison['mean_radius'] > 0]

    # size_all_R2 = R2_value_by_name(
    #     df_comparison,
    #     'mean_volume',
    #     'poids_moy_1_fruit')

    # size_close_R2 = R2_value_by_name(
    #     df_comparison,
    #     'mean_volume',
    #     'poids_moy_1_fruit',
    #     positions=[1, 5, 6, 10, 11, 15, 16, 20])

    # return (fruit_all_R2,
    #         fruit_close_R2,
    #         fruit_middle_R2,
    #         fruit_far_R2,
    #         size_all_R2,
    #         size_close_R2)


def synthetic_comparison(df_synthetic, df_measurements):

    df_comparison = pandas.merge(df_measurements,
                                 df_synthetic)

    return r2_rmse_opti_value(df_comparison['number_of_apple'],
                              df_comparison['nb_cluster'])