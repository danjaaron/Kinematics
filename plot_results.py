""" Plots the results in a simple way """

import pandas as pd
import matplotlib.pyplot as plt

# filename = './results/cifar100_resnet18_epoch200.csv'
filename = './results/cifar10_resnet18.csv'

title_name = filename.split('/')[-1].split('.')[0]

# load results
print("loading {}".format(filename))
results_df = pd.read_csv(filename)

# get data by optimizer
df_list = []
columns_to_convert = ['loss', 'accuracy', 'epoch', 'batch']
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

# plot each interesting column
for col_name in ["loss", "accuracy"]:
    loss_df = master_df.pivot(index="epoch", columns="optimizer", values=col_name)
    loss_df.plot(ylabel=col_name, title=title_name, logy=True)
    plt.show()
