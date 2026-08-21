from pathlib import Path
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns


# Prevent figures from appearing in the notebook
plt.ioff()


# ---------------------------------------------------------
# Input and export folders
# ---------------------------------------------------------

base_folder = Path(
    "/Volumes/shared/Sci-EAE/lahuang_work/working/"
    "Fastice_analyse_m3_results"
)

batch_root = base_folder / "Sentinel"
batch_export_root = base_folder / "time_series_figures"

batch_export_root.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Files required from each maskInSAR folder
# ---------------------------------------------------------

files = {
    "phase_boundary":
        "filt_topophase_phase_geo_fast_ice_boundary.png",

    "amplitude_boundary":
        "topophase_amp_geo_fast_ice_boundary.png",

    "coherence_boundary":
        "topophase_coherence_geo_fast_ice_boundary.png",

    "coherence_only_png":
        "topophase_coherence_geo_fast_ice_only.png",

    "coherence_only_tif":
        "topophase_coherence_geo_fast_ice_only.tif",
}


# ---------------------------------------------------------
# Find all *_InSAR folders
# ---------------------------------------------------------

insar_folders = sorted(
    folder
    for folder in batch_root.rglob("*_InSAR")
    if folder.is_dir()
)

print(f"Found {len(insar_folders)} InSAR folders")


# ---------------------------------------------------------
# Process every *_InSAR folder
# ---------------------------------------------------------

for insar_root in insar_folders:

    relative_folder = insar_root.relative_to(
        batch_root
    )

    export_folder = (
        batch_export_root / relative_folder
    )

    export_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nProcessing: {relative_folder}")
    print(f"Exporting to: {export_folder}")


    # -----------------------------------------------------
    # Find all complete datasets
    # -----------------------------------------------------

    datasets = []

    for mask_folder in sorted(
        insar_root.rglob("maskInSAR")
    ):

        pair_name = mask_folder.parent.name

        match = re.search(
            r"_(\d{8}T\d{6})_(\d{8}T\d{6})",
            pair_name
        )

        if not match:
            print(
                "Skipped: cannot extract dates from "
                f"{pair_name}"
            )
            continue

        paths = {
            key: mask_folder / filename
            for key, filename in files.items()
        }

        missing = [
            path.name
            for path in paths.values()
            if not path.is_file()
        ]

        if missing:
            print(
                f"Skipped {pair_name}; missing: "
                f"{', '.join(missing)}"
            )
            continue

        datasets.append(
            {
                "pair_name": pair_name,

                "first_time": datetime.strptime(
                    match.group(1),
                    "%Y%m%dT%H%M%S"
                ),

                "second_time": datetime.strptime(
                    match.group(2),
                    "%Y%m%dT%H%M%S"
                ),

                **paths,
            }
        )

    datasets.sort(
        key=lambda item: (
            item["first_time"],
            item["second_time"]
        )
    )

    if not datasets:
        print(
            f"Skipped {relative_folder}: "
            "no complete datasets"
        )
        continue

    print(
        f"Found {len(datasets)} complete datasets"
    )


    # -----------------------------------------------------
    # Figure settings
    # -----------------------------------------------------

    ncols = 2

    nrows = int(
        np.ceil(len(datasets) / ncols)
    )


    def short_date(item):

        return (
            f"{item['first_time']:%Y-%m-%d}"
            f" → "
            f"{item['second_time']:%Y-%m-%d}"
        )


    # -----------------------------------------------------
    # Function for plotting all PNGs
    # -----------------------------------------------------

    def plot_all_pngs(
        file_key,
        figure_title,
        output_name
    ):

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 5 * nrows)
        )

        axes = np.atleast_1d(
            axes
        ).ravel()

        for axis, item in zip(
            axes,
            datasets
        ):

            axis.imshow(
                plt.imread(item[file_key])
            )

            axis.set_title(
                short_date(item)
            )

            axis.axis("off")

        for axis in axes[len(datasets):]:
            axis.axis("off")

        fig.suptitle(
            figure_title,
            fontsize=16
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.97)
        )

        output_file = (
            export_folder / output_name
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {output_file}")


    # -----------------------------------------------------
    # Figure 1: all amplitude images
    # -----------------------------------------------------

    plot_all_pngs(
        "amplitude_boundary",
        "Amplitude with fast-ice boundary",
        "all_amplitude.png"
    )


    # -----------------------------------------------------
    # Figure 2: all filtered topophase images
    # -----------------------------------------------------

    plot_all_pngs(
        "phase_boundary",
        "Filtered topophase with fast-ice boundary",
        "all_filtered_topophase.png"
    )


    # -----------------------------------------------------
    # Figure 3: all fast-ice-only coherence images
    # -----------------------------------------------------

    plot_all_pngs(
        "coherence_only_png",
        "Coherence inside the fast-ice area",
        "all_coherence_fast_ice_only.png"
    )


    # -----------------------------------------------------
    # Figure 4: all coherence histograms
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 4 * nrows),
        sharex=True
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for axis, item in zip(
        axes,
        datasets
    ):

        with rasterio.open(
            item["coherence_only_tif"]
        ) as source:

            values = source.read(
                1,
                masked=True
            ).compressed()

        values = values[
            np.isfinite(values)
        ]

        values = values[
            (values >= 0)
            & (values <= 1)
        ]

        axis.hist(
            values,
            bins=50,
            range=(0, 1),
            color="steelblue",
            edgecolor="black",
            linewidth=0.4
        )

        axis.set_xlim(0, 1)

        axis.set_title(
            short_date(item)
        )

        axis.set_xlabel("Coherence")
        axis.set_ylabel("Number of pixels")
        axis.grid(alpha=0.25)

    for axis in axes[len(datasets):]:
        axis.axis("off")

    fig.suptitle(
        "Coherence distributions",
        fontsize=16
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    histogram_output = (
        export_folder
        / "all_coherence_histograms.png"
    )

    fig.savefig(
        histogram_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {histogram_output}")


    # -----------------------------------------------------
    # Figure 5: coherence boxplot over time
    # -----------------------------------------------------

    boxplot_records = []
    coherence_statistics = []

    for item in datasets:

        with rasterio.open(
            item["coherence_only_tif"]
        ) as source:

            values = source.read(
                1,
                masked=True
            ).compressed()

        values = values[
            np.isfinite(values)
        ]

        values = values[
            (values >= 0)
            & (values <= 1)
        ]

        date_label = item[
            "first_time"
        ].strftime("%Y-%m-%d")

        coherence_statistics.append(
            {
                "date": date_label,
                "mean": np.mean(values),
                "std": np.std(values),
            }
        )

        # Use all valid pixels
        boxplot_records.extend(
            {
                "date": date_label,
                "coherence": value,
            }
            for value in values
        )

    boxplot_data = pd.DataFrame(
        boxplot_records
    )

    statistics_data = pd.DataFrame(
        coherence_statistics
    )

    sns.set_theme(
        style="whitegrid"
    )

    fig, axis = plt.subplots(
        figsize=(13, 6)
    )

    sns.boxplot(
        data=boxplot_data,
        x="date",
        y="coherence",
        color="lightsteelblue",
        width=0.6,
        showfliers=False,
        ax=axis
    )

    x_positions = np.arange(
        len(statistics_data)
    )

    axis.errorbar(
        x=x_positions,
        y=statistics_data["mean"],
        yerr=statistics_data["std"],
        fmt="o",
        color="red",
        ecolor="red",
        capsize=5,
        markersize=6,
        linewidth=1.5,
        label="Mean ± 1 standard deviation"
    )

    axis.set_ylim(0, 1)
    axis.set_xlabel("Time")
    axis.set_ylabel("Coherence")

    axis.set_title(
        "Temporal variation of fast-ice "
        "InSAR coherence"
    )

    axis.tick_params(
        axis="x",
        rotation=35
    )

    axis.legend()

    fig.tight_layout()

    boxplot_output = (
        export_folder
        / "coherence_boxplot.png"
    )

    fig.savefig(
        boxplot_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {boxplot_output}")
    print(f"Completed: {relative_folder}")