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
