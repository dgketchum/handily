"""PyVista primitives for the 3-D register of the figure series.

Run as::

    uv run python -m utils.figures.render3d

which renders a small self-check image next to the toy caches.

Conventions
-----------
* **Coordinates.** World x, y in metres exactly as in the 2-D register; z is
  ``VE * elevation``. Use :func:`zs` for every z you hand to a mesh
  so nothing is ever drawn at true scale by accident; :func:`unzs` inverts it.
* **Camera.** One oblique view from the south-east at ~30 deg elevation, applied
  by :func:`shoot` *after* the meshes are added (adding a mesh resets the camera,
  so aiming it up front does not survive). Every 3-D panel in the series uses it,
  so panels are directly comparable. Re-aim with :func:`set_camera` and pass
  ``camera=False`` to :func:`shoot` only if a figure is about another viewpoint.
* **Scale bar.** :func:`standard_plotter` draws a labelled vertical bar stating
  the exaggeration; it is the only text baked into a 3-D render.
* **Land surface.** Uniform pale sand (:data:`fig_common.LAND_3D`) at opacity
  :data:`fig_common.LAND_3D_OPACITY` -- the 3-D register never colour-codes the
  ground, so the water surface underneath stays readable.
* **Off-screen.** ``pv.OFF_SCREEN`` is set True at import. A "bad X server
  connection" warning on a headless box is expected and harmless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from utils.figures.fig_common import (
    CELL,
    CMAPS,
    COLORS,
    EXTENT,
    LAND_3D,
    LAND_3D_OPACITY,
    NX,
    NY,
    TOY_DIR,
    VE,
    VE_LABEL,
)

pv.OFF_SCREEN = True

#: Default oblique camera: azimuth measured clockwise from north (152 deg = SSE,
#: which keeps the 24 km long axis running left to right), elevation above the
#: horizontal, and distance as a multiple of domain width.
CAM_AZIMUTH_DEG = 152.0
CAM_ELEVATION_DEG = 30.0
CAM_DISTANCE_FACTOR = 2.0
#: Zoom applied after ``reset_camera`` fits the scene; < 1 leaves a margin.
CAM_ZOOM = 1.45

#: Length of the vertical scale bar drawn by :func:`standard_plotter`, in true metres.
SCALE_BAR_M = 200.0

#: Shared scalar-bar geometry: vertical, right-hand side, out of the scene.
SCALAR_BAR_ARGS = dict(
    vertical=True,
    title_font_size=20,
    label_font_size=18,
    color="black",
    width=0.04,
    height=0.45,
    position_x=0.90,
    position_y=0.30,
)


def zs(z) -> np.ndarray:
    """True elevation (m) -> exaggerated render z (m x :data:`VE`)."""
    return np.asarray(z, dtype=float) * VE


def unzs(z) -> np.ndarray:
    """Render z -> true elevation (m). Inverse of :func:`zs`."""
    return np.asarray(z, dtype=float) / VE


def _xy():
    """Cell-centre coordinate vectors of the world grid (m)."""
    xs = (np.arange(NX) + 0.5) * CELL
    ys = (np.arange(NY) + 0.5) * CELL
    return xs, ys


def structured_grid(z, scalars: dict | None = None) -> pv.StructuredGrid:
    """Build a ``(NY, NX)`` elevation array into a StructuredGrid at ``VE`` scale.

    ``scalars`` is an optional ``{name: (NY, NX) array}`` attached as point data.
    Returned points carry ``"elevation_m"`` (true metres, not exaggerated) so a
    mesh can always be coloured by its own height.

    VTK stores a StructuredGrid's points in Fortran order (first array axis
    fastest), which is how PyVista flattens the coordinate arrays it is given.
    Every ``(NY, NX)`` field attached here must therefore be flattened the same
    way -- a C-order ravel binds each value to the wrong point.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (NY, NX):
        raise ValueError(f"expected shape {(NY, NX)}, got {z.shape}")
    xs, ys = _xy()
    xx, yy = np.meshgrid(xs, ys)
    grid = pv.StructuredGrid(xx, yy, zs(z))
    grid.point_data["elevation_m"] = z.ravel(order="F")
    for name, arr in (scalars or {}).items():
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (NY, NX):
            raise ValueError(f"{name}: expected shape {(NY, NX)}, got {arr.shape}")
        grid.point_data[name] = arr.ravel(order="F")
    return grid


def surface(
    pl: pv.Plotter,
    z,
    scalars=None,
    scalar_name: str = "value",
    cmap=None,
    clim=None,
    color=None,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    scalar_bar_title: str | None = None,
    **kw,
):
    """Add a world-grid surface at elevation ``z`` (true m; exaggerated on the way in).

    Pass either ``color`` (uniform) or ``scalars`` (a ``(NY, NX)`` array, named
    ``scalar_name``) with ``cmap``/``clim``. Returns the actor.
    """
    extra = {scalar_name: scalars} if scalars is not None else None
    grid = structured_grid(z, extra)
    kw.setdefault("smooth_shading", True)
    args = dict(opacity=opacity, show_scalar_bar=show_scalar_bar, lighting=True, **kw)
    if scalars is not None:
        args.update(scalars=scalar_name, cmap=cmap, clim=clim)
        if show_scalar_bar:
            args["scalar_bar_args"] = SCALAR_BAR_ARGS | {
                "title": scalar_bar_title or scalar_name
            }
    else:
        args["color"] = color if color is not None else LAND_3D
    return pl.add_mesh(grid, **args)


def land_surface(pl: pv.Plotter, world: dict, opacity: float = LAND_3D_OPACITY, **kw):
    """The standard translucent pale-sand ground surface.

    High ambient and no specular: with default lighting a translucent surface at
    0.35 opacity turns into a dark grey shell, and the pale sand is the whole
    point of it.
    """
    kw.setdefault("ambient", 0.55)
    kw.setdefault("diffuse", 0.55)
    kw.setdefault("specular", 0.0)
    return surface(pl, world["dem"], color=LAND_3D, opacity=opacity, **kw)


def well_sticks(
    pl: pv.Plotter,
    wells,
    visible_mask,
    radius: float = 90.0,
    level_radius: float = 240.0,
    level_thickness: float = 60.0,
):
    """Draw the well ledger in 3-D: a stick per well plus its observed water level.

    ``visible_mask`` is a boolean array aligned with ``wells``: True = the label
    was visible to the surface being drawn (black stick), False = hidden (white
    stick). The observed water level is a small black band on the stick -- the
    3-D counterpart of the 2-D horizontal tick -- on both stick colours, so a
    white held-out stick still shows its level against a pale surface.

    Sticks run from ``dem`` down to ``dem - drilled_depth``. Returns
    ``{"visible": actor|None, "hidden": actor|None, "levels": [actor, ...]}``.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    if vis.shape[0] != len(wells):
        raise ValueError("visible_mask must align with wells")
    x = wells["x"].to_numpy(float)
    y = wells["y"].to_numpy(float)
    top = wells["dem"].to_numpy(float)
    bot = top - wells["drilled_depth"].to_numpy(float)
    lev = wells["wte_obs"].to_numpy(float)

    out = {"visible": None, "hidden": None, "levels": []}
    for key, sel, col in (("visible", vis, "black"), ("hidden", ~vis, "white")):
        if not sel.any():
            continue
        pts = np.empty((2 * int(sel.sum()), 3), float)
        pts[0::2] = np.column_stack([x[sel], y[sel], zs(top[sel])])
        pts[1::2] = np.column_stack([x[sel], y[sel], zs(bot[sel])])
        n = pts.shape[0] // 2
        cells = np.column_stack(
            [np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]
        ).ravel()
        poly = pv.PolyData(pts, lines=cells).tube(radius=radius, n_sides=10)
        out[key] = pl.add_mesh(
            poly,
            color=col,
            show_scalar_bar=False,
            ambient=0.45,
            diffuse=0.6,
            specular=0.0,
        )

    # The water-level band is always black, on white and black sticks alike -- it
    # is the 3-D counterpart of the 2-D horizontal tick and has to be legible on
    # a held-out (white) stick standing against a pale surface.
    for xi, yi, li in zip(x, y, lev):
        disc = pv.Cylinder(
            center=(xi, yi, zs(li)),
            direction=(0, 0, 1),
            radius=level_radius,
            height=level_thickness,
        )
        out["levels"].append(
            pl.add_mesh(
                disc, color="black", show_scalar_bar=False, ambient=0.5, diffuse=0.5
            )
        )
    return out


def stream_tubes(
    pl: pv.Plotter,
    world: dict,
    z=None,
    radius: float = 70.0,
    color: str | None = None,
    lift: float = 5.0,
):
    """Draw the D8 stream network as tubes following ``z`` (default the DEM).

    Each stream cell is joined to its D8 receiver, so the network is drawn as
    real flow paths rather than a cloud of cells. ``lift`` (true m) floats the
    tubes just above the surface so they are not z-fought by the ground.
    """
    z = world["dem"] if z is None else z
    z = np.asarray(z, dtype=float) + float(lift)
    streams = np.asarray(world["streams"], dtype=bool)
    receiver = np.asarray(world["receiver"], dtype=np.int64).ravel()
    xs, ys = _xy()
    xx, yy = np.meshgrid(xs, ys)
    xf, yf, zf = xx.ravel(), yy.ravel(), z.ravel()

    src = np.flatnonzero(streams.ravel())
    dst = receiver[src]
    ok = dst >= 0
    src, dst = src[ok], dst[ok]
    if src.size == 0:
        return None
    pts = np.empty((2 * src.size, 3), float)
    pts[0::2] = np.column_stack([xf[src], yf[src], zs(zf[src])])
    pts[1::2] = np.column_stack([xf[dst], yf[dst], zs(zf[dst])])
    n = src.size
    cells = np.column_stack(
        [np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]
    ).ravel()
    poly = pv.PolyData(pts, lines=cells).tube(radius=radius, n_sides=8)
    return pl.add_mesh(poly, color=color or COLORS["r"], show_scalar_bar=False)


def fence_section(
    pl: pv.Plotter,
    line,
    top,
    bottom,
    scalars=None,
    scalar_name: str = "value",
    cmap=None,
    clim=None,
    color=None,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    **kw,
):
    """Hang a vertical curtain along a section line between two profiles.

    ``line`` is ``(x0, y0, x1, y1)`` in metres; ``top`` and ``bottom`` are
    length-``n`` profiles of true elevation (m) sampled along it -- get them from
    :func:`fig_common.profile_along` with the same ``n``. ``scalars`` may be a
    length-``n`` array (constant down the curtain) or an ``(n, m)`` array giving
    values on ``m`` levels from bottom to top.

    Returns the actor.
    """
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    if top.shape != bottom.shape or top.ndim != 1:
        raise ValueError("top and bottom must be 1-D profiles of the same length")
    n = top.size
    x0, y0, x1, y1 = (float(v) for v in line)
    t = np.linspace(0.0, 1.0, n)
    px = x0 + t * (x1 - x0)
    py = y0 + t * (y1 - y0)

    m = 2
    if scalars is not None:
        scalars = np.asarray(scalars, dtype=float)
        if scalars.ndim == 2:
            if scalars.shape[0] != n:
                raise ValueError(f"scalars must have {n} rows, got {scalars.shape}")
            m = scalars.shape[1]
        elif scalars.shape != (n,):
            raise ValueError(f"scalars must be ({n},) or ({n}, m), got {scalars.shape}")

    frac = np.linspace(0.0, 1.0, m)[None, :]
    zz = bottom[:, None] + frac * (top - bottom)[:, None]
    xx = np.repeat(px[:, None], m, axis=1)
    yy = np.repeat(py[:, None], m, axis=1)
    grid = pv.StructuredGrid(xx, yy, zs(zz))
    args = dict(opacity=opacity, show_scalar_bar=show_scalar_bar, **kw)
    if scalars is not None:
        vals = scalars if scalars.ndim == 2 else np.repeat(scalars[:, None], m, axis=1)
        grid.point_data[scalar_name] = vals.ravel(order="F")
        args.update(scalars=scalar_name, cmap=cmap, clim=clim)
    else:
        args["color"] = color if color is not None else COLORS["truth"]
    return pl.add_mesh(grid, **args)


def cutaway(
    pl: pv.Plotter,
    z,
    keep_mask,
    scalars=None,
    scalar_name: str = "value",
    cmap=None,
    clim=None,
    color=None,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    **kw,
):
    """Add a surface with part of it removed, so a surface beneath stays visible.

    ``keep_mask`` is a ``(NY, NX)`` boolean array: True cells are drawn. The usual
    use is to cut the ground away on one side of a section line and leave the
    water table exposed. Returns the actor.
    """
    keep = np.asarray(keep_mask, dtype=bool)
    if keep.shape != (NY, NX):
        raise ValueError(f"keep_mask: expected shape {(NY, NX)}, got {keep.shape}")
    extra = {scalar_name: scalars} if scalars is not None else None
    grid = structured_grid(z, extra)
    grid.point_data["_keep"] = keep.ravel(order="F").astype(np.uint8)
    kept = grid.threshold(0.5, scalars="_keep")
    kw.setdefault("smooth_shading", True)
    args = dict(opacity=opacity, show_scalar_bar=show_scalar_bar, **kw)
    if scalars is not None:
        args.update(scalars=scalar_name, cmap=cmap, clim=clim)
        if show_scalar_bar:
            args["scalar_bar_args"] = SCALAR_BAR_ARGS | {"title": scalar_name}
    else:
        args["color"] = color if color is not None else LAND_3D
    return pl.add_mesh(kept, **args)


def half_mask(line, side: str = "left") -> np.ndarray:
    """A ``(NY, NX)`` boolean mask of the cells on one side of a section line.

    ``side`` is ``"left"`` or ``"right"`` of the direction of travel -- the same
    sign convention as ``fig_common.wells_near_line``'s ``d_offset``. Handy as the
    ``keep_mask`` for :func:`cutaway`.
    """
    x0, y0, x1, y1 = (float(v) for v in line)
    dx, dy = x1 - x0, y1 - y0
    ln = float(np.hypot(dx, dy))
    ux, uy = dx / ln, dy / ln
    xs, ys = _xy()
    xx, yy = np.meshgrid(xs, ys)
    offset = -(xx - x0) * uy + (yy - y0) * ux
    return offset >= 0 if side == "left" else offset <= 0


def standard_plotter(
    window_size=(2400, 1600), scale_bar: bool = True, off_screen: bool = True
) -> pv.Plotter:
    """The one 3-D viewpoint used by the whole series.

    White background, and (unless ``scale_bar=False``) a labelled vertical bar in
    the south-west corner stating the vertical exaggeration.

    The fixed oblique south-east camera is applied by :func:`shoot`, *after* the
    meshes are in -- adding a mesh resets the camera, so aiming it up front does
    not survive. Call :func:`set_camera` yourself and pass ``camera=False`` to
    :func:`shoot` only if a figure genuinely needs another viewpoint.
    """
    pl = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    pl.set_background("white")
    # SSAA replaces the renderer's default render pass, and that pass is what
    # would run depth peeling: calling enable_depth_peeling alongside it is a
    # no-op (GetLastRenderingUsedDepthPeeling stays 0, either call order). The
    # series' scenes carry at most two translucent actors, which VTK's
    # back-to-front actor sort blends correctly, so supersampling is the better
    # trade -- it is what keeps the well sticks and stream tubes clean.
    pl.enable_anti_aliasing("ssaa")
    if scale_bar:
        add_vertical_scale_bar(pl)
    return pl


def set_camera(
    pl: pv.Plotter,
    azimuth_deg: float = CAM_AZIMUTH_DEG,
    elevation_deg: float = CAM_ELEVATION_DEG,
    distance_factor: float = CAM_DISTANCE_FACTOR,
    focus_z_m: float | None = None,
    zoom: float = CAM_ZOOM,
) -> None:
    """Aim the camera at the scene centre from the series' fixed oblique bearing.

    The bearing is set first, then ``reset_camera`` fits the actual scene bounds
    along it and ``zoom`` backs off to leave a margin, so the whole block is in
    frame whatever the vertical exaggeration or how tall the meshes happen to be.
    Pass ``focus_z_m`` to override the focal height (true metres).
    """
    b = pl.bounds
    cx = 0.5 * (b[0] + b[1]) if b[1] > b[0] else 0.5 * (EXTENT[0] + EXTENT[1])
    cy = 0.5 * (b[2] + b[3]) if b[3] > b[2] else 0.5 * (EXTENT[2] + EXTENT[3])
    cz = 0.5 * (b[4] + b[5]) if focus_z_m is None else zs(focus_z_m)
    width = EXTENT[1] - EXTENT[0]
    r = distance_factor * width
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    pos = (
        cx + r * np.cos(el) * np.sin(az),
        cy + r * np.cos(el) * np.cos(az),
        cz + r * np.sin(el),
    )
    pl.camera_position = [pos, (cx, cy, cz), (0.0, 0.0, 1.0)]
    pl.reset_camera()
    pl.camera.zoom(zoom)


def add_vertical_scale_bar(
    pl: pv.Plotter, length_m: float = SCALE_BAR_M, at=None, color: str = "black"
) -> None:
    """A vertical bar of ``length_m`` true metres, labelled with the exaggeration."""
    if at is None:
        at = (EXTENT[0] + 900.0, EXTENT[2] + 900.0, 1500.0)
    x, y, z0 = (float(v) for v in at)
    bar = pv.Line((x, y, zs(z0)), (x, y, zs(z0 + length_m)))
    pl.add_mesh(bar.tube(radius=90.0), color=color, show_scalar_bar=False)
    pl.add_point_labels(
        np.array([[x, y, zs(z0 + length_m)]]),
        [f"{length_m:g} m ({VE_LABEL})"],
        font_size=20,
        text_color=color,
        shape=None,
        show_points=False,
        always_visible=True,
    )


def shoot(
    pl: pv.Plotter, stem: str, out_dir=TOY_DIR, close: bool = True, camera: bool = True
) -> Path:
    """Render off-screen to ``<out_dir>/<stem>.png`` and return the path.

    Applies the series' standard camera first (``camera=False`` to keep whatever
    view the caller aimed).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    if camera:
        set_camera(pl)
    pl.screenshot(str(path))
    if close:
        pl.close()
    return path


def export_html(pl: pv.Plotter, stem: str, out_dir=TOY_DIR) -> Path | None:
    """Best-effort interactive export to ``<out_dir>/<stem>.html``.

    Needs the optional ``trame``/``vtk`` web stack; returns None (and prints the
    reason) when that is not installed, so a figure script never fails on it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.html"
    try:
        pl.export_html(str(path))
    except Exception as exc:  # optional dependency, never fatal
        print(f"export_html skipped ({type(exc).__name__}: {exc})")
        return None
    return path


def _main() -> None:
    from utils.figures.toy_world import load_world

    world, wells = load_world()
    pl = standard_plotter(window_size=(1600, 1100))
    land_surface(pl, world)
    surface(
        pl,
        world["wte_true"],
        scalars=np.asarray(world["dtw_true"], float),
        scalar_name="DTW (m)",
        cmap=CMAPS["dtw"],
        clim=(0.0, 100.0),
        show_scalar_bar=True,
    )
    stream_tubes(pl, world)
    sel = wells.sample(60, random_state=0).sort_index()
    well_sticks(pl, sel, sel["is_source"].to_numpy(bool))
    path = shoot(pl, "render3d_selfcheck")
    print(f"wrote {path}")


if __name__ == "__main__":
    _main()
