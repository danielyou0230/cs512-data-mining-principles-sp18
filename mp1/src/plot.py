import argparse
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns


def ass1_plot():
    # Fix Single word
    high_s = [itr / 10 for itr in range(0, 11, 2)]
    high_m = [1.0] * 6
    dblp = [6474.506, 6474.506, 6474.506, 3700.344, 956.355, 829.164]
    yelp = [5620.537, 5620.537, 5620.537, 2712.355, 1560.970, 378.531]

    # plot
    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    d, = plt.plot(high_s, dblp)
    y, = plt.plot(high_s, yelp)
    plt.title('Parameter study on HIGHLIGHT_SINGLE\n(HIGHLIGHT_MULTI = 1.0)')
    plt.xlabel('HIGHLIGHT_SINGLE')
    plt.ylabel('Number of phrases (k)')
    plt.legend([d, y], ("DBLP.300K", "YELP.100K"))
    plt.ylim([0, 7000])
    fig.savefig("plot_s.png", dpi=300)

    # Fix Multi-word
    high_s = [1.0] * 6
    high_m = [itr / 10 for itr in range(0, 11, 2)]
    dblp = [2767.982, 1224.934, 912.284, 746.795, 512.180, 47.565]
    yelp = [1609.528, 477.466, 297.388, 229.658, 163.845, 28.895]

    # plot
    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    d, = plt.plot(high_m, dblp)
    y, = plt.plot(high_m, yelp)
    plt.title('Parameter study on HIGHLIGHT_MULTI \n(HIGHLIGHT_SINGLE = 1.0)')
    plt.xlabel('HIGHLIGHT_MULTI')
    plt.ylabel('Number of phrases (k)')
    plt.legend([d, y], ("DBLP.300K", "YELP.100K"))
    plt.ylim([0, 7000])
    fig.savefig("plot_m.png", dpi=300)


if __name__ == '__main__':

    ass1_plot()