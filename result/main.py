from sklearn.linear_model import LinearRegression
import sklearn.metrics
import sklearn.linear_model
import pandas
import numpy
import scipy
import datetime
import matplotlib.pyplot
import matplotlib.dates
import math
import glob
import os


def load_harvest_2018():
    df_gt_harvest = pandas.read_csv("data/2018_08_ground_truth.csv")

    for column_name in ['poids_moy_1_fruit',
                        'poids_recolte',
                        'poids_recoltes_avec_sol',
                        'Nb_fruits_recoltes',
                        'Nb_fruits_recoltes_avec_sol']:

        # Convert in numeric format
        df_gt_harvest[column_name] = pandas.to_numeric(df_gt_harvest[column_name],
                                                       downcast='float',
                                                       errors='coerce')

    # Convert date in datetime
    df_gt_harvest['date_recolte'] = pandas.to_datetime(df_gt_harvest['date_recolte'],
                                                       format='%m/%d/%Y',
                                                       errors='coerce',
                                                       utc=False)

    # Select measurements only for data scanned
    # df_gt_harvest = df_gt_harvest[numpy.bitwise_and(df_gt_harvest['Ligne'] <= 4,
    #                                                 df_gt_harvest['Position'] <= 20)]

    # Select data after the scan time periode
    df_gt_harvest = df_gt_harvest[df_gt_harvest['date_recolte']
                                  >= pandas.Timestamp('2018-08-02')]

    # Remove NAN value
    df_gt_harvest = df_gt_harvest.dropna()

    df_gt_harvest["harvest_nb_fruit_floor"] = df_gt_harvest["Nb_fruits_recoltes_avec_sol"] - \
        df_gt_harvest["Nb_fruits_recoltes"]
    df_gt_harvest["harvest_weight_fruit_floor"] = df_gt_harvest["poids_recoltes_avec_sol"] - \
        df_gt_harvest["poids_recolte"]

    df_gt_harvest["year"] = "2018"

    df_gt_harvest = df_gt_harvest.rename(
        columns={"Ligne": "line",
                 "Position": "position",
                 "Nb_fruits_recoltes_avec_sol": "harvest_nb_fruit",
                 "poids_recoltes_avec_sol": "harvest_weight_fruit",
                 "poids_recolte": "harvest_weight_fruit_tree",
                 "Nb_fruits_recoltes": "harvest_nb_fruit_tree"})

    # Plot the dataframe
    return df_gt_harvest


def load_harvest_2019():

    df_gt_harvest = pandas.read_csv("data/harvest_data_2019.csv")

    for column_name in ['poids_moy_1_fruit',
                        'poids_recolte',
                        'poids_recoltes_avec_sol',
                        'Nb_fruits_recoltes',
                        'Nb_fruits_recoltes_avec_sol']:

        # Convert in numeric format
        df_gt_harvest[column_name] = pandas.to_numeric(df_gt_harvest[column_name],
                                                       downcast='float',
                                                       errors='coerce')

    # Convert date in datetime
    df_gt_harvest['date_recolte'] = pandas.to_datetime(df_gt_harvest['date_recolte'],
                                                       format='%m/%d/%Y',
                                                       errors='coerce',
                                                       utc=False)

    # Select data after the scan time periode
    df_gt_harvest = df_gt_harvest[df_gt_harvest['date_recolte']
                                  >= pandas.Timestamp('2019-09-09')]

    df_gt_harvest["harvest_nb_fruit_floor"] = df_gt_harvest["Nb_fruits_recoltes_avec_sol"] - \
        df_gt_harvest["Nb_fruits_recoltes"]
    df_gt_harvest["harvest_weight_fruit_floor"] = df_gt_harvest["poids_recoltes_avec_sol"] - \
        df_gt_harvest["poids_recolte"]

    df_gt_harvest["year"] = "2019"

    df_gt_harvest = df_gt_harvest.rename(
        columns={"Ligne": "line",
                 "Position": "position",
                 "Nb_fruits_recoltes_avec_sol": "harvest_nb_fruit",
                 "poids_recoltes_avec_sol": "harvest_weight_fruit",
                 "poids_recolte": "harvest_weight_fruit_tree",
                 "Nb_fruits_recoltes": "harvest_nb_fruit_tree"})

    # Plot the dataframe
    return df_gt_harvest


def load_measurements(filename):

    # Load the autmatic measurements
    df_automatic_measurements = pandas.read_csv(filename)

    df_automatic_measurements['date'] = pandas.to_datetime(df_automatic_measurements['date'],
                                                           format='%Y_%m_%d',
                                                           errors='coerce',
                                                           utc=False)

    df_automatic_measurements['year'] = df_automatic_measurements['basename'].str[5:9]

    # df_automatic_measurements = df_automatic_measurements[df_automatic_measurements['basename'].str.contains("high_quality")]

    return df_automatic_measurements


def plot_R2_linear_model(axis, df_colum_1, df_colum_2, color='r', xlabel=None, ylabel=None):

    X = numpy.array([df_colum_1.to_numpy()]).T
    y = df_colum_2.to_numpy()

    axis.plot(X, y, c=color, marker='.', linestyle='', markersize=20)

    reg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X, y)
    val = reg.predict(numpy.array([[0], [1]]))
    coef = numpy.round((val[1] - val[0]), decimals=2)
    axis.plot(X, reg.predict(X), 'b--',
              label="R² = {:.2f} coef: {}".format(reg.score(X, y), coef))

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X, y)
    val = reg.predict(numpy.array([[0], [1]]))
    coef = numpy.round((val[1] - val[0]), decimals=2)
    axis.plot(X, reg.predict(X), 'k--',
              label="R² = {:.2f} coef: {}".format(reg.score(X, y), coef))

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


def save(df_comparison, output_dir, title):

    if df_comparison.shape[0] == 0:
        return None

    matplotlib.pyplot.rcParams["figure.figsize"] = (15, 15)

    camp = matplotlib.pyplot.get_cmap('hsv', 1)
    fig = matplotlib.pyplot.figure()
    axis = matplotlib.pyplot.axes()

    fig.suptitle(title, fontsize=32)
    plot_R2_linear_model(axis,
                         df_comparison['nb_fruit'],
                         df_comparison['harvest_nb_fruit'],
                         color=camp(0),
                         xlabel='Number of cluster',
                         ylabel='Number of fruit')

    filename = os.path.join(output_dir, title + ".png")
    matplotlib.pyplot.savefig(filename)


if __name__ == "__main__":

    df_harvest_2018 = load_harvest_2018()
    df_harvest_2019 = load_harvest_2019()
    df_harvest = pandas.concat([df_harvest_2018, df_harvest_2019])

    for filename in glob.glob("*.csv"):
        print(filename)
        df_measurements = load_measurements(filename)

        df_comparison = pandas.merge(df_harvest,
                                     df_measurements,
                                     left_on=['line', 'position', 'year'],
                                     right_on=['line', 'position', 'year'])

        output_dir = os.path.basename(filename)[:-4]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        elements = [
            (df_comparison,
             "Apple Tree: 2018 & 2019"),
            (df_comparison[df_comparison['position'].isin([1, 5, 6, 10, 11, 15, 16, 20])],
             "Apple Tree : 2018 & 2019 | POS 1&5"),
            (df_comparison[df_comparison['position'].isin([2, 4, 7, 9, 12, 14, 17, 19])],
             "Apple Tree : 2018 & 2019 | POS 2&4"),
            (df_comparison[df_comparison['position'].isin([3, 8, 13, 18])],
             "Apple Tree : 2018 & 2019 | POS 3"),
            (df_comparison[df_comparison['basename'].str.contains("high_quality")],
             "Apple Tree : 2019 | High Quality"),
            (df_comparison[df_comparison['year'] == "2018"],
             "Apple Tree: 2018"),
            (df_comparison[df_comparison['year'] == "2019"],
             "Apple Tree: 2019")]

        for df, title in elements:
            print(title)
            save(df, output_dir, title)
