import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y

def logistic(x):
    """
    Elementwise logistic sigmoid.
    :param x: numpy array
    :return: numpy array
    """
    return 1.0 / (1.0 + np.exp(-x))

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))

def disp_imdata(xs, imsize, layout=(1,1)):
    """
    Displays an array of images, a page at a time. The user can navigate pages with
    left and right arrows, start over by pressing space, or close the figure by esc.
    :param xs: an numpy array with images as rows
    :param imsize: size of the images
    :param layout: layout of images in a page
    :return: none
    """

    num_plots = np.prod(layout)
    num_xs = xs.shape[0]
    idx = [0]

    # create a figure with suplots
    fig, axs = plt.subplots(layout[0], layout[1])

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    def plot_page():
        """Plots the next page."""

        ii = np.arange(idx[0], idx[0]+num_plots) % num_xs

        for ax, i in zip(axs, ii):
            ax.imshow(xs[i].reshape(imsize), cmap='gray', interpolation='none')
            ax.set_title(str(i))

        fig.canvas.draw()

    def on_key_event(event):
        """Event handler after key press."""

        key = event.key

        if key == 'right':
            # show next page
            idx[0] = (idx[0] + num_plots) % num_xs
            plot_page()

        elif key == 'left':
            # show previous page
            idx[0] = (idx[0] - num_plots) % num_xs
            plot_page()

        elif key == ' ':
            # show first page
            idx[0] = 0
            plot_page()

        elif key == 'escape':
            # close figure
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_event)
    plot_page()