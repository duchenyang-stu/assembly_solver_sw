from __future__ import annotations

import re
import shutil
import struct
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .input_loader import AssemblyInput
from .solve import SolveRecord

VIEW_DEFS = {
    "iso_xp_yp_zp": (1.0, 1.0, 1.0),
    "iso_xp_yp_zn": (1.0, 1.0, -1.0),
    "iso_xp_yn_zp": (1.0, -1.0, 1.0),
    "iso_xp_yn_zn": (1.0, -1.0, -1.0),
    "iso_xn_yp_zp": (-1.0, 1.0, 1.0),
    "iso_xn_yp_zn": (-1.0, 1.0, -1.0),
    "iso_xn_yn_zp": (-1.0, -1.0, 1.0),
    "iso_xn_yn_zn": (-1.0, -1.0, -1.0),
}
DEFAULT_VIEWS = tuple(VIEW_DEFS)
FIXED_PART_COLOR = "#2563eb"
MOVING_PART_COLOR = "#facc15"


def save_solution_views(
    assembly: AssemblyInput,
    record: SolveRecord,
    output_dir: str | Path,
    *,
    width: int = 1200,
    height: int = 900,
) -> dict[str, str]:
    if record.status != "ok" or record.transform is None:
        return {}

    occ = _require_occ()
    pair_dir, fixed_shape, moving_shape = _solution_shapes(occ, assembly, record, output_dir, record.transform)
    artifacts = _write_solution_step(
        occ,
        record,
        pair_dir,
        fixed_shape=fixed_shape,
        moving_shape=moving_shape,
    )
    artifacts.update(_write_png_views((fixed_shape, moving_shape), pair_dir, width, height))
    artifacts["render_format"] = "png"
    return artifacts


def save_solution_step(
    assembly: AssemblyInput,
    record: SolveRecord,
    output_dir: str | Path,
    *,
    transform: Sequence[Sequence[float]] | None = None,
    use_pair_subdir: bool = True,
) -> dict[str, str]:
    matrix = transform if transform is not None else record.transform
    if matrix is None:
        return {}

    occ = _require_occ()
    if not use_pair_subdir:
        _remove_pair_subdir(output_dir, record)
    pair_dir, fixed_shape, moving_shape = _solution_shapes(
        occ,
        assembly,
        record,
        output_dir,
        matrix,
        use_pair_subdir=use_pair_subdir,
    )
    return _write_solution_step(
        occ,
        record,
        pair_dir,
        fixed_shape=fixed_shape,
        moving_shape=moving_shape,
    )


def _remove_pair_subdir(output_dir: str | Path, record: SolveRecord) -> None:
    root = Path(output_dir)
    pair_dir = root / _pair_stem(record)
    if not pair_dir.exists():
        return
    if pair_dir.resolve().parent != root.resolve():
        raise RuntimeError(f"Refusing to remove unexpected pair directory: {pair_dir}")
    if pair_dir.is_symlink():
        pair_dir.unlink()
        return
    if pair_dir.is_dir():
        shutil.rmtree(pair_dir)


def _write_png_views(
    shapes: tuple[object, object],
    pair_dir: Path,
    width: int,
    height: int,
) -> dict[str, str]:
    from OCC.Display.OCCViewer import OffscreenRenderer

    renderer = OffscreenRenderer(screen_size=(int(width), int(height)))
    _configure_offscreen_renderer(renderer)
    fixed_shape, moving_shape = shapes
    renderer.DisplayShape(fixed_shape, color=_quantity_color("#2563eb"), update=False)
    renderer.DisplayShape(moving_shape, color=_quantity_color("#f97316"), update=False)

    artifacts: dict[str, str] = {}
    for view_name, direction in VIEW_DEFS.items():
        _set_offscreen_view(renderer, view_name, direction)
        renderer.FitAll()
        renderer.Repaint()
        png_path = pair_dir / f"{view_name}.png"
        renderer.ExportToImage(str(png_path))
        if not png_path.exists() or png_path.stat().st_size == 0:
            raise RuntimeError(f"Offscreen renderer did not create {png_path.name}.")
        _make_png_background_transparent(png_path)
        artifacts[f"png_{view_name}"] = str(png_path.resolve())
    return artifacts


def _configure_offscreen_renderer(renderer) -> None:
    try:
        renderer.SetOrthographicProjection()
    except Exception:
        pass
    try:
        renderer.EnableAntiAliasing()
    except Exception:
        pass
    try:
        renderer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    except Exception:
        pass
    try:
        renderer.hide_triedron()
    except Exception:
        pass


def _quantity_color(value: str):
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_sRGB

    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}.")
    red = int(text[0:2], 16) / 255.0
    green = int(text[2:4], 16) / 255.0
    blue = int(text[4:6], 16) / 255.0
    return Quantity_Color(red, green, blue, Quantity_TOC_sRGB)


def _make_png_background_transparent(path: Path) -> None:
    signature, width, height, color_type, payload = _read_png_pixels(path)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Unsupported PNG signature in {path}.")

    bytes_per_pixel = 3 if color_type == 2 else 4
    rows = _unfilter_png_rows(payload, width, height, bytes_per_pixel)
    transparent_rows = bytearray()
    for row in rows:
        transparent_rows.append(0)
        for offset in range(0, len(row), bytes_per_pixel):
            red, green, blue = row[offset : offset + 3]
            alpha = row[offset + 3] if color_type == 6 else 255
            if max(abs(int(red) - 255), abs(int(green) - 255), abs(int(blue) - 255)) <= 8:
                alpha = 0
            transparent_rows.extend((red, green, blue, alpha))

    path.write_bytes(
        signature
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(bytes(transparent_rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _read_png_pixels(path: Path) -> tuple[bytes, int, int, int, bytes]:
    data = path.read_bytes()
    signature = data[:8]
    offset = 8
    width = height = color_type = None
    idat_parts: list[bytes] = []

    while offset + 8 <= len(data):
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + chunk_length]
        offset += 12 + chunk_length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8 or color_type not in {2, 6} or compression != 0 or filter_method != 0 or interlace != 0:
                raise RuntimeError(f"Unsupported PNG format in {path}.")
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None or not idat_parts:
        raise RuntimeError(f"Invalid PNG data in {path}.")
    return signature, int(width), int(height), int(color_type), zlib.decompress(b"".join(idat_parts))


def _unfilter_png_rows(payload: bytes, width: int, height: int, bytes_per_pixel: int) -> list[bytearray]:
    stride = width * bytes_per_pixel
    rows: list[bytearray] = []
    offset = 0
    previous = bytearray(stride)

    for _ in range(height):
        filter_type = payload[offset]
        offset += 1
        row = bytearray(payload[offset : offset + stride])
        offset += stride

        for index, value in enumerate(row):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 1:
                row[index] = (value + left) & 0xFF
            elif filter_type == 2:
                row[index] = (value + up) & 0xFF
            elif filter_type == 3:
                row[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (value + _paeth_predictor(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"Unsupported PNG filter type {filter_type}.")

        rows.append(row)
        previous = row
    return rows


def _paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _set_offscreen_view(renderer, view_name: str, direction: Sequence[float]) -> None:
    view = getattr(renderer, "View", None)
    if view is None or not hasattr(view, "SetProj"):
        raise RuntimeError("Offscreen renderer does not expose View.SetProj for isometric PNG views.")

    try:
        view.SetProj(*(float(value) for value in direction))
    except TypeError:
        from OCC.Core.gp import gp_Dir

        view.SetProj(gp_Dir(*(float(value) for value in direction)))
    except Exception as exc:
        raise RuntimeError(f"Failed to set PNG view {view_name}.") from exc

    for method_name in ("SetViewOrientationDefault", "ZFitAll"):
        method = getattr(view, method_name, None)
        if method is not None:
            try:
                method()
            except Exception:
                pass


def _require_occ() -> dict[str, object]:
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
        from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
        from OCC.Core.TCollection import TCollection_ExtendedString
        from OCC.Core.TDataStd import TDataStd_Name
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.XCAFDoc import (
            XCAFDoc_ColorGen,
            XCAFDoc_ColorSurf,
            XCAFDoc_DocumentTool_ColorTool,
            XCAFDoc_DocumentTool_ShapeTool,
        )
        from OCC.Core.gp import gp_Trsf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing pythonocc-core/OCC; run inside the CAD environment.") from exc
    return locals()


def _load_step_shape(occ: dict[str, object], path: Path):
    reader = occ["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != occ["IFSelect_RetDone"] or reader.TransferRoots() == 0:
        raise RuntimeError(f"Failed to read STEP: {path}")
    return reader.OneShape()


def _transform_shape(occ: dict[str, object], shape, transform: Sequence[Sequence[float]]):
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {matrix.shape}.")
    trsf = occ["gp_Trsf"]()
    trsf.SetValues(*(float(matrix[row, col]) for row in range(3) for col in range(4)))
    return occ["BRepBuilderAPI_Transform"](shape, trsf, True).Shape()


def _solution_shapes(
    occ: dict[str, object],
    assembly: AssemblyInput,
    record: SolveRecord,
    output_dir: str | Path,
    transform: Sequence[Sequence[float]],
    *,
    use_pair_subdir: bool = True,
) -> tuple[Path, object, object]:
    pair_dir = Path(output_dir) / _pair_stem(record) if use_pair_subdir else Path(output_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)

    fixed_shape = _load_step_shape(occ, assembly.part_paths[record.fixed_part])
    moving_shape = _transform_shape(
        occ,
        _load_step_shape(occ, assembly.part_paths[record.moving_part]),
        transform,
    )
    return pair_dir, fixed_shape, moving_shape


def _write_solution_step(
    occ: dict[str, object],
    record: SolveRecord,
    pair_dir: Path,
    *,
    fixed_shape,
    moving_shape,
) -> dict[str, str]:
    step_path = pair_dir / "assembled.step"
    _write_colored_step(
        occ,
        [
            (record.fixed_part, fixed_shape, FIXED_PART_COLOR),
            (record.moving_part, moving_shape, MOVING_PART_COLOR),
        ],
        step_path,
    )
    return {
        "step": str(step_path.resolve()),
        "fixed_color": FIXED_PART_COLOR,
        "moving_color": MOVING_PART_COLOR,
    }


def _make_compound(occ: dict[str, object], shapes: Iterable[object]):
    builder = occ["BRep_Builder"]()
    compound = occ["TopoDS_Compound"]()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def _write_step(occ: dict[str, object], shape, path: Path) -> None:
    writer = occ["STEPControl_Writer"]()
    writer.Transfer(shape, occ["STEPControl_AsIs"])
    if writer.Write(str(path)) != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"Failed to write STEP: {path}")


def _write_colored_step(occ: dict[str, object], parts: Sequence[tuple[str, object, str]], path: Path) -> None:
    doc = occ["TDocStd_Document"](occ["TCollection_ExtendedString"]("MDTV-XCAF"))
    shape_tool = occ["XCAFDoc_DocumentTool_ShapeTool"](doc.Main())
    color_tool = occ["XCAFDoc_DocumentTool_ColorTool"](doc.Main())

    for part_name, shape, color in parts:
        label = shape_tool.AddShape(shape, False)
        occ["TDataStd_Name"].Set(label, occ["TCollection_ExtendedString"](str(part_name)))
        quantity_color = _quantity_color(color)
        color_tool.SetColor(label, quantity_color, occ["XCAFDoc_ColorGen"])
        color_tool.SetColor(label, quantity_color, occ["XCAFDoc_ColorSurf"])

    writer = occ["STEPCAFControl_Writer"]()
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    if not writer.Transfer(doc, occ["STEPControl_AsIs"]):
        raise RuntimeError(f"Failed to transfer colored STEP document: {path}")
    if writer.Write(str(path)) != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"Failed to write STEP: {path}")


def _pair_stem(record: SolveRecord) -> str:
    raw = f"{record.index:02d}_{record.fixed_part}__{record.moving_part}"
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("._")
    return safe or f"{record.index:02d}_pair"
