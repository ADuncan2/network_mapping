"""
Module containing plotting routines.
"""

from functools import singledispatch
from typing import Any
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.axes._axes import Axes
from shapely.geometry import (
    Polygon,
    Point,
    LineString
)
from shapely import wkb
from gridstock.recorder import NetworkData


@singledispatch
def plot_geometry(geom: Any, _: Axes):
    """
    Function to plot an individual shapely geometry.
    Each type of geometry (Point, Polygon, LineString)
    needs to be plotted differently. To handle this, this
    function is overloaded using singledispatch to accept
    each of these three types.
    """
    # If geom is not of a type for which an overloaded
    # function definition is given, then this
    # code will be reached, giving a type error.
    msg = f"Type: {type(geom)} cannot be used with function plot_geometry()"
    raise TypeError(msg)

@plot_geometry.register
def _(geom: Polygon, ax: Axes, **kwargs) -> PatchCollection:
    """
    Overloaded Polygon implementation of the
    plot_geometry function.
    """
    path = Path.make_compound_path(
        Path(np.asarray(geom.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in geom.interiors])
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

@plot_geometry.register
def _(geom: Point, ax: Axes, **kwargs) -> None:
    """
    Overloaded Point implementation of the
    plot_geometry function.
    """
    ax.plot(geom.x, geom.y, **kwargs)

@plot_geometry.register
def _(geom: LineString, ax: Axes, **kwargs) -> None:
    """
    Overloaded LineString implementation of the
    plot_geometry function.
    """
    ax.plot(*geom.xy, **kwargs)


def plot_net_data(
        net_data: NetworkData,
        ax: Axes,
        sub_colour = "b",
        edge_colour = "r",
        **kwargs
        ) -> None:
    """
    Function to plot the data contained in a NetworkData
    object.
    """
    
    # if len(net_data.substations) > 0:
    #     for substation in net_data.substations:
    geom = wkb.loads(net_data.substation_geom)
    if geom != None:
        plot_geometry(geom, ax, color=sub_colour, **kwargs)

    if len(net_data.edge_list) > 0:
        for edge in net_data.edge_list:
            geom = wkb.loads(edge[1])
            plot_geometry(geom, ax, color=edge_colour, **kwargs)






