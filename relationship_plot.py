import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import itertools
from math import sqrt
import numpy as np


class RelationshipPlot():
    def __init__(self, data, core_column:str, relation_columns:list, legend:bool = False, palette:str = "husl", dir:str = None, resolution:list = [1920, 1080]):
        self.data = data
        self.core_column = core_column
        self.relation_columns = relation_columns
        self.legend = legend
        self.palette = palette
        self.directory = dir
        self.resolution = resolution


    def divisorGenerator(self, n):
        large_divisors = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                yield i
                if i*i != n:
                    large_divisors.append(n / i)
        for divisor in reversed(large_divisors):
            yield int(divisor)

    def calculate_matrix(self):
        x = 0
        y = 0
        num_columns = len(self.relation_columns)
        if num_columns <= 0:
            raise ValueError
        if num_columns <= 3:
            x = 1
            y = num_columns
            return x, y

        sqrt_num = sqrt(num_columns)
        if sqrt_num == int(sqrt_num):
            x = sqrt_num
            y = sqrt_num
            return x, y

        divisors = [i for i in self.divisorGenerator(num_columns)]
        if len(divisors) == 2:
            num_columns = num_columns+1

            divisors = [i for i in self.divisorGenerator(num_columns)]
        pairs = []
        old_diff = num_columns * num_columns

        for d1 in divisors:
            for d2 in divisors:
                if d1*d2 == num_columns:
                    diff = d1-d2
                    if diff < old_diff and d1 < d2:
                        x = d1
                        y = d2
        return x, y


    def create_fig(self):

        x, y = self.calculate_matrix()
        fig, axes = plt.subplots(x, y, sharex = True, sharey = True, figsize=(self.resolution[0]/72,self.resolution[1]/72))

        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("{} x All measures".format(self.core_column))

        return fig, axes

    def create_plots(self):

        x, y = self.calculate_matrix()
        fig, axes = self.create_fig()
        sns.set_style("ticks")
        sns.despine()
        relation_columns = self.relation_columns
        core_column = self.core_column
        df = self.data

        n_grids = x * y
        xlist = []
        ylist = []

        for i in range(0, x):
            xlist.append(i)
        for i in range(0 , y):
            ylist.append(i)

        product = list(itertools.product(xlist, ylist))

        if len(relation_columns) == n_grids:

            for ax, feature in zip(axes.flatten(), relation_columns):
                sns.lineplot(data = df[[core_column, feature]], palette = self.palette, legend = self.legend, ax = ax)
                ax.set(title="{} x {}\ncorr = {}".format(core_column, feature, round(df[[core_column, feature]].corr()[core_column][1], 2)))
                sns.despine()

        if len(relation_columns) < n_grids:
            relation_columns.insert(0, core_column)
            axes_list = []

            for ax, feature in zip(axes.flatten(), relation_columns):
                axes_list.append(ax)

                if len(axes_list) == 1:
                    sns.lineplot(data = df[core_column], palette = self.palette, legend = self.legend, ax = ax)
                    ax.set(title="{}".format(core_column))
                    sns.despine()

                else:
                    sns.lineplot(data = df[[core_column, feature]], palette = self.palette, legend = self.legend, ax = ax)
                    ax.set(title="{} x {}\ncorr = {}".format(core_column, feature, round(df[[core_column, feature]].corr()[core_column][1], 2)))
                    sns.despine()
        if self.directory is not None:
            plt.savefig(self.directory, format = 'png')
        return plt.show()





def main():
    d = {'abc': [1, 2, 1], 'def': [3, 4, 5], 'ghi': [2, 1, 1], 'bla4': [1, 3, 5], 'bla5': [2, 1, 4], 'bla6': [1, 3, 4], 'bla7': [3, 1, 4], 'bla8': [1, 3, 2], 'bla9': [2, 3, 1]}

    df = pd.DataFrame(data = d)
    cl = RelationshipPlot(df, "def", ["abc", "ghi", "bla4", "bla5", "bla6", "bla7", "bla8", "bla9"], dir = "images/relation_teste.png")
    plots = cl.create_plots()
    return plots


if __name__ == "__main__":
    main()
