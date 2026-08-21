#!/usr/bin/env python3
"""Delineate a fast-ice boundary and mask geocoded InSAR GeoTIFFs.

The script selects the FastIce raster closest to the requested date and creates
two GeoTIFFs plus two PNG previews for each input raster:

1. An RGB visualization with the fast-ice boundary shown in red.
2. A single-band raster retaining the original values only over fast ice.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject


DEFAULT_TIFS = (
    "filt_topophase_phase_geo.tif",
    "topophase_amp_geo.tif",
    "topophase_coherence_geo.tif",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fast-ice boundary and masked InSAR products."
    )
    parser.add_argument(
        "--geotiff-dir",
        type=Path,
        default=Path(
            "/Volumes/shared/Sci-EAE/lahuang_work/working/"
            "Shackleton1/test_IW1/merged/"
        ),
        help=(
            "Directory containing the geocoded InSAR GeoTIFFs "
            "(default: the local path used in the notebook)."
        ),
    )
    parser.add_argument(
        "--fast-ice-tif-dir",
        type=Path,
        default=Path(
            "/Users/lhua0106/Library/CloudStorage/"
            "Dropbox-MonashUniEnterprise/Lanqing Huang/Monash2026/fastice/"
            "Large_py_output/FastIce_70_2017"
        ),
        help=(
            "Directory containing FastIce_70_YYYYMMDD.tif files "
            "(default: the local path used in the notebook)."
        ),
    )
    parser.add_argument(
        "--input-date",
        default="2017-06-15",
        help="Requested date in YYYY-MM-DD format (default: 2017-06-15).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--fast-ice-category",
        type=int,
        default=4,
        help="Fast-ice class value (default: 4).",
    )
    parser.add_argument(
        "--boundary-width-pixels",
        type=int,
        default=2,
        help="Boundary width in pixels (default: 2).",
    )
    parser.add_argument(
        "--tif-files",
        nargs="+",
        default=list(DEFAULT_TIFS),
        help="Input filenames relative to --geotiff-dir.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300 dpi).",
    )
    return parser.parse_args()


def select_fast_ice_tif(directory: Path, input_date: str) -> tuple[Path, pd.Timestamp]:
    date_pattern = re.compile(r"FastIce_70_(\d{8})\.tif$", re.IGNORECASE)
    candidates = []

    for path in directory.glob("*.tif"):
        match = date_pattern.match(path.name)
        if match:
            candidates.append(
                (path, pd.to_datetime(match.group(1), format="%Y%m%d"))
            )

    if not candidates:
        raise FileNotFoundError(
            f"No FastIce_70_YYYYMMDD.tif files found in {directory.resolve()}"
        )

    target_date = pd.to_datetime(input_date, format="%Y-%m-%d")
    selected_path, selected_date = min(
        candidates, key=lambda item: abs(item[1] - target_date)
    )

    print(f"Requested date: {target_date:%Y-%m-%d}")
    print(f"Selected date:  {selected_date:%Y-%m-%d}")
    print(f"Selected file:  {selected_path}")
    return selected_path, selected_date


def read_source_mask(
    fast_ice_tif: Path, fast_ice_category: int
) -> tuple[np.ndarray, object, object]:
    with rasterio.open(fast_ice_tif) as src:
        if src.crs is None:
            raise ValueError(
                f"{fast_ice_tif.name} has no CRS. Assign the correct CRS first."
            )

        fast_ice_data = src.read(1, masked=True)
        source_mask = (
            (~np.ma.getmaskarray(fast_ice_data))
            & (np.asarray(fast_ice_data.data) == fast_ice_category)
        ).astype("uint8")
        source_transform = src.transform
        source_crs = src.crs

        print(f"Fast-ice CRS: {src.crs}")
        print(f"Fast-ice resolution: {src.res}")
        print(f"Fast-ice bounds: {src.bounds}")
        print(f"Class-{fast_ice_category} pixels: {source_mask.sum():,}")

        if source_mask.sum() == 0:
            values = fast_ice_data.compressed()
            print(f"Available values: {np.unique(values)}")
            raise ValueError(
                f"No pixels equal category {fast_ice_category} in "
                f"{fast_ice_tif.name}."
            )

    return source_mask, source_transform, source_crs


def erode_binary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbours = [
        padded[row : row + mask.shape[0], col : col + mask.shape[1]]
        for row in range(3)
        for col in range(3)
    ]
    return np.logical_and.reduce(neighbours)


def dilate_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row : row + result.shape[0], col : col + result.shape[1]]
            for row in range(3)
            for col in range(3)
        ]
        result = np.logical_or.reduce(neighbours)
    return result


def mask_on_reference_grid(
    src: rasterio.io.DatasetReader,
    source_mask: np.ndarray,
    source_transform: object,
    source_crs: object,
) -> np.ndarray:
    destination = np.zeros((src.height, src.width), dtype="uint8")
    reproject(
        source=source_mask,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=src.transform,
        dst_crs=src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
    )
    return destination == 1


def display_parameters(
    filename: str, data: np.ndarray, valid: np.ndarray
) -> tuple[float, float, str]:
    if "phase" in filename and "amp" not in filename:
        return -np.pi, np.pi, "twilight"
    if "coherence" in filename:
        return 0.0, 1.0, "viridis"

    finite_values = data[valid]
    if finite_values.size == 0:
        raise ValueError(f"{filename} contains no valid pixels.")
    vmin, vmax = np.nanpercentile(finite_values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid display range for {filename}: {vmin}, {vmax}")
    return float(vmin), float(vmax), "gray"


def save_png(tif_path: Path, dpi: int) -> Path:
    with rasterio.open(tif_path) as src:
        extent = [
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        ]
        fig, ax = plt.subplots(figsize=(12, 8))

        if src.count == 3:
            image = np.moveaxis(src.read(), 0, 2)
            ax.imshow(image, extent=extent, origin="upper")
        else:
            image = src.read(1, masked=True)
            data = image.filled(np.nan).astype("float32")
            valid = ~np.ma.getmaskarray(image) & np.isfinite(data)
            vmin, vmax, cmap = display_parameters(tif_path.name, data, valid)
            plotted = ax.imshow(
                image,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
            )
            fig.colorbar(plotted, ax=ax, shrink=0.8, label="Pixel value")

        ax.set_title(tif_path.stem)
        ax.set_xlabel("Easting / Longitude")
        ax.set_ylabel("Northing / Latitude")
        fig.tight_layout()

        png_path = tif_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved PNG: {png_path.name}")
    return png_path


def process_tif(
    input_tif: Path,
    output_dir: Path,
    source_mask: np.ndarray,
    source_transform: object,
    source_crs: object,
    fast_ice_category: int,
    selected_date: pd.Timestamp,
    boundary_width_pixels: int,
    png_dpi: int,
) -> None:
    if not input_tif.exists():
        raise FileNotFoundError(f"Required input not found: {input_tif}")

    with rasterio.open(input_tif) as src:
        if src.crs is None:
            raise ValueError(f"{input_tif.name} has no CRS.")

        raster = src.read(1, masked=True)
        data = raster.filled(np.nan).astype("float32")
        valid = ~np.ma.getmaskarray(raster) & np.isfinite(data)
        fast_ice_on_tif = mask_on_reference_grid(
            src, source_mask, source_transform, source_crs
        )

        print(f"\n{input_tif.name}")
        print(f"  GeoTIFF CRS: {src.crs}")
        print(f"  GeoTIFF bounds: {src.bounds}")
        print(f"  Class-{fast_ice_category} pixels: {fast_ice_on_tif.sum():,}")

        if not fast_ice_on_tif.any():
            raise ValueError(
                f"No fast-ice overlap for {input_tif.name}. Check CRS and bounds."
            )

        boundary = fast_ice_on_tif & ~erode_binary(fast_ice_on_tif)
        if boundary_width_pixels > 1:
            boundary = dilate_binary(boundary, boundary_width_pixels - 1)

        vmin, vmax, cmap = display_parameters(input_tif.name, data, valid)
        scaled = np.zeros(data.shape, dtype="float32")
        scaled[valid] = np.clip(
            (data[valid] - vmin) / (vmax - vmin), 0.0, 1.0
        )
        rgb = (plt.get_cmap(cmap)(scaled)[..., :3] * 255).astype("uint8")
        rgb[~valid] = [0, 0, 0]
        rgb[boundary & valid] = [255, 0, 0]

        boundary_tif = output_dir / f"{input_tif.stem}_fast_ice_boundary.tif"
        rgb_profile = src.profile.copy()
        rgb_profile.update(
            count=3,
            dtype="uint8",
            nodata=None,
            compress="deflate",
            photometric="RGB",
        )
        with rasterio.open(boundary_tif, "w", **rgb_profile) as dst:
            dst.write(np.moveaxis(rgb, 2, 0))
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            dst.update_tags(
                fast_ice_class=str(fast_ice_category),
                fast_ice_date=f"{selected_date:%Y-%m-%d}",
                boundary_colour="red",
            )

        keep = valid & fast_ice_on_tif
        masked_data = np.where(keep, data, np.nan).astype("float32")
        masked_tif = output_dir / f"{input_tif.stem}_fast_ice_only.tif"
        mask_profile = src.profile.copy()
        mask_profile.update(
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
        )
        with rasterio.open(masked_tif, "w", **mask_profile) as dst:
            dst.write(masked_data, 1)
            description = src.descriptions[0] or input_tif.stem
            dst.set_band_description(1, f"{description} - fast ice only")
            dst.update_tags(
                fast_ice_class=str(fast_ice_category),
                fast_ice_date=f"{selected_date:%Y-%m-%d}",
            )

    print(f"  Saved boundary GeoTIFF: {boundary_tif.name}")
    print(f"  Saved masked GeoTIFF:   {masked_tif.name}")
    save_png(boundary_tif, png_dpi)
    save_png(masked_tif, png_dpi)


def main() -> None:
    args = parse_args()

    if args.boundary_width_pixels < 1:
        raise ValueError("--boundary-width-pixels must be at least 1.")
    if args.png_dpi < 1:
        raise ValueError("--png-dpi must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved in: {args.output_dir.resolve()}")

    selected_tif, selected_date = select_fast_ice_tif(
        args.fast_ice_tif_dir, args.input_date
    )
    source_mask, source_transform, source_crs = read_source_mask(
        selected_tif, args.fast_ice_category
    )

    for tif_name in args.tif_files:
        process_tif(
            input_tif=args.geotiff_dir / tif_name,
            output_dir=args.output_dir,
            source_mask=source_mask,
            source_transform=source_transform,
            source_crs=source_crs,
            fast_ice_category=args.fast_ice_category,
            selected_date=selected_date,
            boundary_width_pixels=args.boundary_width_pixels,
            png_dpi=args.png_dpi,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
