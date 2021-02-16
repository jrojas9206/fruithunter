import pandas
import logging
import os
import numpy

def init_log(output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = os.path.join(output_dir, "post_process.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO)


def load_harvest_2018():
    df_gt_harvest = pandas.read_csv("data/harvest_data_2018.csv")

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

def load_harvest():
    
    df_harvest_2018 = load_harvest_2018()
    df_harvest_2019 = load_harvest_2019()
    return pandas.concat([df_harvest_2018, df_harvest_2019], sort=True)

def load_synthetic():

    filename = "./data/synthetic_tree_apple_number.csv"
    df_synthetic = pandas.read_csv(filename)

    return df_synthetic

def load_data(filenames):
    data = dict()
    for filename in filenames:
        print("Load : {}".format(filename), flush=True)

        npy_filename = filename[:-3] + 'npy'
        if os.path.exists(npy_filename):
            data[filename] = numpy.load(npy_filename)
        else:
            data[filename] = numpy.loadtxt(filename)
            numpy.save(npy_filename, data[filename])

    return data

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