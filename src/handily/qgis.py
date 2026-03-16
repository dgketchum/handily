"""QGIS project integration for handily outputs.

This module provides utilities to update QGIS projects (.qgz files) with
handily output layers without requiring PyQGIS.

Approach: Manipulate .qgz files directly using zipfile + xml.etree.ElementTree.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger("handily.qgis")

# Layer type detection patterns
RASTER_EXTENSIONS = (".tif", ".tiff", ".vrt")
VECTOR_EXTENSIONS = (".fgb", ".shp", ".gpkg", ".geojson")


def discover_outputs(out_dir: str, view_root: str | None = None) -> list[dict]:
    """Scan output directory for GeoTIFFs and FlatGeobufs, return layer metadata.

    Args:
        out_dir: Directory to scan for output files
        view_root: If set, remap paths for viewing on a different machine.
            The home directory portion of paths will be replaced with this root.
            E.g., if out_dir is ~/data/proj/outputs and view_root is /Volumes/user/data,
            layer paths will use /Volumes/user/data/proj/outputs/...

    Returns a list of dicts with keys:
        - name: layer name (file stem)
        - path: absolute file path (remapped if view_root is set)
        - type: 'raster' or 'vector'
        - category: 'Rasters' or 'Vectors'
    """
    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    layers = []

    for fname in os.listdir(out_dir):
        fpath = os.path.join(out_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()
        name = os.path.splitext(fname)[0]

        abs_path = os.path.abspath(fpath)

        # Remap path if view_root is specified
        if view_root:
            abs_path = _remap_path_for_view(abs_path, view_root)

        if ext in RASTER_EXTENSIONS:
            layers.append({
                "name": name,
                "path": abs_path,
                "type": "raster",
                "category": "Rasters",
            })
        elif ext in VECTOR_EXTENSIONS:
            layers.append({
                "name": name,
                "path": abs_path,
                "type": "vector",
                "category": "Vectors",
            })

    # Sort by category then name
    layers.sort(key=lambda x: (x["category"], x["name"]))
    LOGGER.info("Discovered %d layers in %s", len(layers), out_dir)
    return layers


def _remap_path_for_view(path: str, view_root: str) -> str:
    """Remap a path for viewing on a different machine.

    Replaces the home directory portion with view_root.
    E.g., /home/user/data/proj -> /Volumes/user/data/proj
    """
    home = os.path.expanduser("~")
    if path.startswith(home):
        # Replace home with view_root
        relative = path[len(home):]
        if relative.startswith("/"):
            relative = relative[1:]
        return os.path.join(view_root, relative)
    return path


def update_project(project_path: str, layers: list[dict], group_name: str) -> None:
    """Add/update layers in existing .qgs project file.

    Args:
        project_path: Path to .qgs file
        layers: List of layer dicts from discover_outputs()
        group_name: Name of layer group to create/update
    """
    project_path = os.path.expanduser(project_path)
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project file not found: {project_path}")

    if not project_path.endswith(".qgs"):
        raise ValueError("Project must be a .qgs file")

    # Create backup
    backup_path = project_path + ".bak"
    if os.path.exists(backup_path):
        os.remove(backup_path)
    shutil.copyfile(project_path, backup_path)

    # Parse and modify the XML
    tree = ET.parse(project_path)
    root = tree.getroot()

    # Update project layers and tree
    _update_project_layers(root, layers, group_name)

    # Write modified XML
    tree.write(project_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Updated project: %s (backup: %s)", project_path, backup_path)


def _update_project_layers(root: ET.Element, layers: list[dict], group_name: str) -> None:
    """Update QGIS project XML with new layers."""
    # Get or create projectlayers element
    projectlayers = root.find("projectlayers")
    if projectlayers is None:
        projectlayers = ET.SubElement(root, "projectlayers")

    # Get or create layer-tree-group
    layer_tree = root.find("layer-tree-group")
    if layer_tree is None:
        layer_tree = ET.SubElement(root, "layer-tree-group")

    # Find or create our group in the layer tree
    group_elem = None
    for child in layer_tree:
        if child.tag == "layer-tree-group" and child.get("name") == group_name:
            group_elem = child
            break

    if group_elem is None:
        group_elem = ET.SubElement(layer_tree, "layer-tree-group")
        group_elem.set("name", group_name)
        group_elem.set("checked", "Qt::Checked")
        group_elem.set("expanded", "1")

    # Create subgroups for Rasters and Vectors
    raster_group = None
    vector_group = None
    for child in group_elem:
        if child.tag == "layer-tree-group":
            if child.get("name") == "Rasters":
                raster_group = child
            elif child.get("name") == "Vectors":
                vector_group = child

    if raster_group is None:
        raster_group = ET.SubElement(group_elem, "layer-tree-group")
        raster_group.set("name", "Rasters")
        raster_group.set("checked", "Qt::Checked")
        raster_group.set("expanded", "1")

    if vector_group is None:
        vector_group = ET.SubElement(group_elem, "layer-tree-group")
        vector_group.set("name", "Vectors")
        vector_group.set("checked", "Qt::Checked")
        vector_group.set("expanded", "1")

    # Track existing layer IDs to remove stale entries
    existing_layer_ids = set()
    for maplayer in projectlayers.findall("maplayer"):
        layer_id = maplayer.find("id")
        if layer_id is not None:
            existing_layer_ids.add(layer_id.text)

    # Add/update layers
    for layer in layers:
        layer_id = _add_or_update_layer(projectlayers, layer)

        # Add to appropriate group in layer tree
        target_group = raster_group if layer["type"] == "raster" else vector_group

        # Check if layer already in tree
        found = False
        for child in target_group:
            if child.tag == "layer-tree-layer" and child.get("id") == layer_id:
                found = True
                break

        if not found:
            layer_tree_elem = ET.SubElement(target_group, "layer-tree-layer")
            layer_tree_elem.set("id", layer_id)
            layer_tree_elem.set("name", layer["name"])
            layer_tree_elem.set("checked", "Qt::Checked")
            layer_tree_elem.set("expanded", "0")
            layer_tree_elem.set("source", layer["path"])
            layer_tree_elem.set("providerKey", "gdal" if layer["type"] == "raster" else "ogr")


def _add_or_update_layer(projectlayers: ET.Element, layer: dict) -> str:
    """Add or update a maplayer element. Returns the layer ID."""
    # Check if layer with same source already exists
    for maplayer in projectlayers.findall("maplayer"):
        datasource = maplayer.find("datasource")
        if datasource is not None and datasource.text == layer["path"]:
            # Update existing layer
            layer_name = maplayer.find("layername")
            if layer_name is not None:
                layer_name.text = layer["name"]
            layer_id = maplayer.find("id")
            if layer_id is not None:
                return layer_id.text
            # Generate ID if missing
            new_id = f"{layer['name']}_{uuid.uuid4().hex[:8]}"
            id_elem = ET.SubElement(maplayer, "id")
            id_elem.text = new_id
            return new_id

    # Create new maplayer
    layer_id = f"{layer['name']}_{uuid.uuid4().hex[:8]}"

    maplayer = ET.SubElement(projectlayers, "maplayer")

    if layer["type"] == "raster":
        maplayer.set("type", "raster")
        _configure_raster_layer(maplayer, layer, layer_id)
    else:
        maplayer.set("type", "vector")
        _configure_vector_layer(maplayer, layer, layer_id)

    return layer_id


def _configure_raster_layer(maplayer: ET.Element, layer: dict, layer_id: str) -> None:
    """Configure a raster maplayer element."""
    maplayer.set("hasScaleBasedVisibilityFlag", "0")
    maplayer.set("autoRefreshEnabled", "0")

    id_elem = ET.SubElement(maplayer, "id")
    id_elem.text = layer_id

    datasource = ET.SubElement(maplayer, "datasource")
    datasource.text = layer["path"]

    layername = ET.SubElement(maplayer, "layername")
    layername.text = layer["name"]

    provider = ET.SubElement(maplayer, "provider")
    provider.text = "gdal"

    # Add pseudocolor renderer for REM layers
    if "rem" in layer["name"].lower():
        _add_raster_pseudocolor(maplayer, vmin=0.0, vmax=10.0)
    elif "ndwi" in layer["name"].lower():
        _add_raster_pseudocolor(maplayer, vmin=-1.0, vmax=1.0, colormap="RdYlGn")
    elif "streams" in layer["name"].lower():
        _add_raster_singleband(maplayer)


def _configure_vector_layer(maplayer: ET.Element, layer: dict, layer_id: str) -> None:
    """Configure a vector maplayer element."""
    maplayer.set("geometry", "Polygon")
    maplayer.set("hasScaleBasedVisibilityFlag", "0")

    id_elem = ET.SubElement(maplayer, "id")
    id_elem.text = layer_id

    datasource = ET.SubElement(maplayer, "datasource")
    datasource.text = layer["path"]

    layername = ET.SubElement(maplayer, "layername")
    layername.text = layer["name"]

    provider = ET.SubElement(maplayer, "provider")
    provider.text = "ogr"

    # Add categorized renderer for strata/pattern columns
    if "stratified" in layer["name"].lower():
        _add_categorized_renderer(maplayer, "strata")
    elif "pattern" in layer["name"].lower():
        _add_categorized_renderer(maplayer, "pattern")
    elif "flowlines" in layer["name"].lower():
        _add_simple_line_renderer(maplayer, color="0,100,200,255")


def _add_raster_pseudocolor(maplayer: ET.Element, vmin: float, vmax: float,
                            colormap: str = "Spectral") -> None:
    """Add a pseudocolor raster renderer."""
    pipe = ET.SubElement(maplayer, "pipe")
    renderer = ET.SubElement(pipe, "rasterrenderer")
    renderer.set("type", "singlebandpseudocolor")
    renderer.set("opacity", "1")
    renderer.set("band", "1")

    shader = ET.SubElement(renderer, "rastershader")
    colorramp = ET.SubElement(shader, "colorrampshader")
    colorramp.set("colorRampType", "INTERPOLATED")
    colorramp.set("minimumValue", str(vmin))
    colorramp.set("maximumValue", str(vmax))

    # Blue to red color ramp (similar to what was in viz.py)
    if colormap == "Spectral":
        colors = [
            (vmin, "44,123,182,255"),  # blue
            ((vmin + vmax) / 2, "255,255,191,255"),  # yellow
            (vmax, "215,25,28,255"),  # red
        ]
    elif colormap == "RdYlGn":
        colors = [
            (vmin, "215,25,28,255"),  # red
            (0.0, "255,255,191,255"),  # yellow
            (vmax, "26,150,65,255"),  # green
        ]
    else:
        colors = [
            (vmin, "0,0,0,255"),
            (vmax, "255,255,255,255"),
        ]

    for val, color in colors:
        item = ET.SubElement(colorramp, "item")
        item.set("value", str(val))
        item.set("color", color)
        item.set("alpha", "255")


def _add_raster_singleband(maplayer: ET.Element) -> None:
    """Add a simple grayscale raster renderer."""
    pipe = ET.SubElement(maplayer, "pipe")
    renderer = ET.SubElement(pipe, "rasterrenderer")
    renderer.set("type", "singlebandgray")
    renderer.set("opacity", "1")
    renderer.set("band", "1")


def _add_categorized_renderer(maplayer: ET.Element, field: str) -> None:
    """Add a categorized vector renderer."""
    renderer = ET.SubElement(maplayer, "renderer-v2")
    renderer.set("type", "categorizedSymbol")
    renderer.set("attr", field)
    renderer.set("enableorderby", "0")

    # Add default symbol
    symbols = ET.SubElement(renderer, "symbols")
    symbol = ET.SubElement(symbols, "symbol")
    symbol.set("type", "fill")
    symbol.set("name", "0")
    symbol.set("alpha", "0.5")

    layer_sym = ET.SubElement(symbol, "layer")
    layer_sym.set("class", "SimpleFill")
    layer_sym.set("enabled", "1")

    prop = ET.SubElement(layer_sym, "prop")
    prop.set("k", "color")
    prop.set("v", "200,200,200,255")


def _add_simple_line_renderer(maplayer: ET.Element, color: str = "0,0,255,255") -> None:
    """Add a simple line renderer for flowlines."""
    renderer = ET.SubElement(maplayer, "renderer-v2")
    renderer.set("type", "singleSymbol")

    symbols = ET.SubElement(renderer, "symbols")
    symbol = ET.SubElement(symbols, "symbol")
    symbol.set("type", "line")
    symbol.set("name", "0")
    symbol.set("alpha", "1")

    layer_sym = ET.SubElement(symbol, "layer")
    layer_sym.set("class", "SimpleLine")
    layer_sym.set("enabled", "1")

    prop = ET.SubElement(layer_sym, "prop")
    prop.set("k", "line_color")
    prop.set("v", color)

    prop_width = ET.SubElement(layer_sym, "prop")
    prop_width.set("k", "line_width")
    prop_width.set("v", "0.5")


def generate_qlr(layers: list[dict], output_path: str) -> None:
    """Generate QGIS Layer Definition file for drag-and-drop import.

    QLR files can be dragged into QGIS to add layers without modifying
    the project file directly.
    """
    output_path = os.path.expanduser(output_path)

    root = ET.Element("qlr")
    layer_tree = ET.SubElement(root, "layer-tree-group")
    layer_tree.set("name", "handily")
    layer_tree.set("checked", "Qt::Checked")
    layer_tree.set("expanded", "1")

    maplayers = ET.SubElement(root, "maplayers")

    for layer in layers:
        layer_id = f"{layer['name']}_{uuid.uuid4().hex[:8]}"

        # Add to layer tree
        layer_elem = ET.SubElement(layer_tree, "layer-tree-layer")
        layer_elem.set("id", layer_id)
        layer_elem.set("name", layer["name"])
        layer_elem.set("checked", "Qt::Checked")
        layer_elem.set("source", layer["path"])
        layer_elem.set("providerKey", "gdal" if layer["type"] == "raster" else "ogr")

        # Add maplayer
        maplayer = ET.SubElement(maplayers, "maplayer")
        if layer["type"] == "raster":
            maplayer.set("type", "raster")
            _configure_raster_layer(maplayer, layer, layer_id)
        else:
            maplayer.set("type", "vector")
            _configure_vector_layer(maplayer, layer, layer_id)

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Generated QLR: %s", output_path)


def open_project(project_path: str) -> None:
    """Open QGIS with the specified project.

    Attempts to find and launch QGIS executable.
    """
    import subprocess
    import platform

    project_path = os.path.expanduser(project_path)
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project file not found: {project_path}")

    system = platform.system()

    if system == "Linux":
        # Try common QGIS executable names
        for cmd in ["qgis", "qgis3", "qgis-ltr"]:
            try:
                subprocess.Popen([cmd, project_path])
                LOGGER.info("Opened QGIS with: %s %s", cmd, project_path)
                return
            except FileNotFoundError:
                continue
        raise FileNotFoundError("QGIS executable not found. Install QGIS or add it to PATH.")

    elif system == "Darwin":
        # macOS
        subprocess.Popen(["open", "-a", "QGIS", project_path])
        LOGGER.info("Opened QGIS with: open -a QGIS %s", project_path)

    elif system == "Windows":
        # Windows - try to find QGIS
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\QGIS 3")
            qgis_path, _ = winreg.QueryValueEx(key, "InstallPath")
            qgis_exe = os.path.join(qgis_path, "bin", "qgis-ltr-bin.exe")
            if not os.path.exists(qgis_exe):
                qgis_exe = os.path.join(qgis_path, "bin", "qgis-bin.exe")
            subprocess.Popen([qgis_exe, project_path])
            LOGGER.info("Opened QGIS: %s", project_path)
        except (WindowsError, FileNotFoundError):
            # Fallback to shell open
            os.startfile(project_path)
            LOGGER.info("Opened project with default application: %s", project_path)
    else:
        raise OSError(f"Unsupported platform: {system}")
