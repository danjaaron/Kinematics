""" Plots the results in a simple way """
# TODO: systematically save the plots

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

RESULTS_PATH = './results/'

if len(sys.argv) > 1:
    print(sys.argv)
    RESULTS_PATH = sys.argv[-1]

file_list = os.listdir(sys.argv[-1])

for filename in file_list:
    if not '.csv' in filename:
        continue

    print("attempting to load {}".format(filename))
    try:
        filename = RESULTS_PATH+filename
        title_name = filename.split('/')[-1].split('.')[0]

        # load results
        print("loading {}".format(filename))
        results_df = pd.read_csv(filename)
        print(results_df.columns)

        # get data by optimizer
        df_list = []
        #columns_to_convert = ['loss', 'accuracy', 'epoch', 'batch']
        columns_to_convert = list(results_df.columns.values)
        columns_to_convert = [c for c in columns_to_convert if not c in ['Unnamed: 0', 'optimizer']]
        for opt_name in results_df.optimizer.unique():
            if opt_name == 'optimizer':
                continue
            print("parsing {} optimizer results".format(opt_name))
            opt_df = results_df[results_df.optimizer == opt_name]
            # convert values to float
            for col_name in columns_to_convert:
                print("... converting {}".format(col_name))
                opt_df[col_name] = opt_df[col_name].apply(lambda x: float(x))
            # get epoch level results
            opt_df = opt_df.drop_duplicates('epoch')
            df_list.append(opt_df)

        # combine, pivot and plot
        print('got df_list')
        master_df = pd.concat(df_list)

        # pivot src: https://stackoverflow.com/questions/29233283/plotting-multiple-lines-with-pandas-dataframe#29233885

        # plot each interestng column
        mdf_col = master_df.columns.values
        plot_cols = [c for c in mdf_col if 'loss' in c]
        for col_name in plot_cols:
            loss_df = master_df.pivot(index="epoch", columns="optimizer", values=col_name)
            loss_df.plot(ylabel=col_name, title=title_name, logy=True)
            plt.show()

    except Exception as e:
        print(">>> Exception occurred, skipping...")
        print(e)
