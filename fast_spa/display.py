import itertools
from typing import Any, Mapping, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


class AutoAnimator(FuncAnimation):
    images: list[AxesImage]

    def __init__(
        self,
        data: np.ndarray,
        extent: Sequence[float],
        features: Sequence = [],
        origin: str = "lower",
        cmap: Any = "viridis",
        fig_scale: float = 4,
        vmin_max: Sequence[dict[str, float]] | None = None,
        **kwargs: Any,
    ):
        if data.ndim == 4:
            data = data[np.newaxis, ...]
        assert data.ndim == 5
        self.extent = extent
        self.origin = origin
        self.fig_scale = fig_scale
        self.features = features
        self.cmap = cmap
        vmin_max = vmin_max or list(itertools.repeat({}, data.shape[1]))

        # construct figure and axes
        B, C, T, Y, X = data.shape
        fig, ax = self._subplots(rows=B, channels=C)

        # reshape data to 4D tensor
        data = data.reshape(-1, T, Y, X)
        # config = config or list(itertools.repeat({}, C))

        self.images = [
            self._imshow(
                ax[row, col],  # type: ignore
                data[row * C + col, 0],
                **vmin_max[row * C + col],
            )
            for row, col in itertools.product(range(B), range(C))
        ]

        self.data = data

        super().__init__(
            fig,
            self._animate,
            init_func=lambda: self.images,
            frames=T,
            blit=True,
            **kwargs,
        )

    def _subplots(self, rows: int, *, channels: int) -> tuple[Figure, Mapping[tuple[int, int], Axes]]:
        scale = self.fig_scale

        fig, axes = plt.subplots(
            rows,
            channels,
            figsize=(scale * channels, scale * rows),
            sharex=True,
            sharey=True,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        fig.tight_layout()

        if rows == 1:
            axes = axes[np.newaxis, ...]
        if channels == 1:
            axes = axes[..., np.newaxis]

        x, y = axes.shape
        for i, j in itertools.product(range(x), range(y)):
            ax = axes[i, j]
            ax.coastlines()
            ax.set_extent(self.extent)
            for feature in self.features:
                ax.add_feature(feature)

        return fig, axes  # type: ignore

    def _animate(self, tidx: int) -> list[AxesImage]:
        for i in range(len(self.images)):
            self.images[i].set_data(self.data[i, tidx, ...])
        return self.images

    def _imshow(self, ax: GeoAxes, data: np.ndarray, **kwargs) -> AxesImage:
        return ax.imshow(
            data,
            origin=self.origin,
            extent=self.extent,
            transform=ccrs.PlateCarree(),
            animated=True,
            cmap=self.cmap,
            **kwargs,
        )
