r"""
Contains parameters and functions for plotting which fit the LaTeX setup we
have. Plots with the figsize herein are intended to be included in the size
they are created and saved, not to be scaled by e.g. `[width=\linewidth]` in
the includegraphics command in the manuscript.
"""
from typing import Dict
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class PlotConfig:
    # ratio W/H is 4/3
    fig_width = 3.2
    fig_height = 2.4

    fontsize_axis_labels = 10
    fontsize_axis_ticks = 8
    fontsize_title = 10
    fontsize_suptitle = 10
    fontsize_legends = 8
    fontsize_parameters = 8
    parameters_alpha = 0.5

    line_styles = ['-', '--', ':']

    save_dir = 'plots'

    @staticmethod
    def use_tex():
        """Call to enable LaTeX fonts in matplotlib."""
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "font.serif": ["Computer Modern"]
        })

    @staticmethod
    def set_ax_info(
        ax: mpl.axes.Axes,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        legend: bool = False
    ):
        """Write labels, titles, and legends on an axis with the PlotConfig fontsizes.

        Args:
            ax (matplotlib.axes.Axes): the axis on which to display information
            xlabel (str): the desired label on the x-axis
            ylabel (str): the desired label on the y-axis
            title (str, optional): the desired title on the axis
                default: None
            legend (bool, optional): whether or not to add labels/legend
                default: False
        """
        ax.set_xlabel(xlabel, fontsize=PlotConfig.fontsize_axis_labels)
        ax.set_ylabel(ylabel, fontsize=PlotConfig.fontsize_axis_labels)
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=PlotConfig.fontsize_axis_ticks
        )
        # ax.ticklabel_format(style='plain')

        if title is not None:
            ax.set_title(title, fontsize=PlotConfig.fontsize_title)

        if legend:
            ax.legend(fontsize=PlotConfig.fontsize_legends)

    @staticmethod
    def tex_std_form(num: str) -> str:
        """Replace calculator notation with standard form in LaTeX."""
        # remove e+00
        num = num.replace('e+00', '')
        # match digit followed by e, optional minus and digits
        pattern = r'(\d)e(-?\+?\d+)'
        num = re.sub(pattern, r'\1 \\cdot 10^{\2}', num)
        # remove leading + and 0
        num = num.replace(r'10^{+', r'10^{')
        num = num.replace(r'10^{0', r'10^{')
        num = num.replace(r'10^{-0', r'10^{-')
        return num

    @staticmethod
    def tight_layout(fig: mpl.figure.Figure, ax_aspect: float | None = 3/2):
        """Expand the content of a figure to fill the entire figure.

        Only implemented for figures with one Axes object. Also has
        functionality to resize the figure vertically to get the desired aspect
        ratio for the Axes. A fixed figure size will give different Axes sizes
        for different number of lines in title, labels etc. Supplying a desired
        aspect ratio will correct for this variation.

        Args:
            fig (matplotlib.figure.Figure): the figure to adjust.
            ax_aspect (float, optional): the desired aspect ratio
                (width/height) for the Axes object in the figure. If supplied,
                the vertical figure size will be adjusted to get the desired
                aspect ratio.
                Default: 3/2. Use None to turn off size adjustment.
        """
        assert len(fig.axes) == 1, 'PlotConfig.tight_layout can only work ' \
            f'with figures with 1 axis, you have {len(fig.axes)}.'
        ax = fig.axes[0]

        fig.tight_layout(pad=0.1)
        if ax_aspect is None:
            return

        # Store the original yticks. If the rescaling causes the yticks to
        # become larger in size, the width will not adjust properly and the
        # left side of the figure may disappear outside the figure. This can
        # happen e.g. if the ticks start start to label on every 0.5 instead of
        # every 1, thus making each tick take up more space after the resize
        # than before.
        old_yticks = ax.get_yticks()
        old_ylim = ax.get_ylim()

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        height_increase = width / ax_aspect - height
        fig.set_figheight(fig.get_figheight() + height_increase)

        fig.tight_layout(pad=0.1)
        new_yticks = ax.get_yticks()
        if new_yticks.shape != old_yticks.shape or np.any(new_yticks != old_yticks):
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            msg = 'Warning in PlotConfig.tight_layout:\n' \
                '\tResizing caused the yticks to change. This can cause\n' \
                '\tproblems, so the original yticks will be used instead.\n' \
                '\tTo avoid undesired effects and inaccuracies, choose an\n' \
                '\tinitial figure height which makes the necessary\n' \
                '\tadjustment as small as possible down the line.\n' \
                f'\tThe figure height was increased by {height_increase:.2e}.\n'\
                f'\tNew aspect ratio is: {width / height}, requested\n' \
                f'\twas {ax_aspect}.'
            print(msg)
            l, u = old_ylim
            # Use the old ticks, but make sure to only use the ones which are
            # within the bounds set by ylim. Sometimes the array of ticks
            # includes a value below the lower limit in ylim and one above the
            # upper limit.
            ax.set_yticks([t for t in old_yticks if l <= t <= u])

    @staticmethod
    def save_fname(prefix: str, suffix: str, parameters: Dict[str, float]) -> Path:
        result = Path(PlotConfig.save_dir)
        fname = prefix
        for parameter, value in parameters.items():
            fname += f'__{parameter}_{value:.4f}'
        result = result / fname
        return result.with_suffix(suffix)

    @staticmethod
    def parameter_text_box(ax: mpl.axes.Axes, s: str, loc: str) -> None:
        if loc == 'upper right':
            x = 0.99
            y = 0.98
            horizontalalignment = 'right'
            verticalalignment = 'top'
        elif loc == 'lower right':
            x = 0.99
            y = 0.02
            horizontalalignment = 'right'
            verticalalignment = 'bottom'
        else:
            raise ValueError(
                f'Text box location {loc} is currently not implemented.'
            )

        ax.text(
            x=x,
            y=y,
            s=s,
            alpha=PlotConfig.parameters_alpha,
            fontsize=PlotConfig.fontsize_parameters,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transAxes
        )
