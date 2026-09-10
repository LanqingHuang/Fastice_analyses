from pathlib import Path
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import rasterio
import seaborn as sns


# Prevent figures from appearing in the notebook
plt.ioff() 

# Enable Panel/Bokeh components used by the interactive HTML viewers
pn.extension()

# Display width of the images inside the interactive viewers
IMAGE_WIDTH = 900


# ---------------------------------------------------------
# Input and export folders
# ---------------------------------------------------------

base_folder = Path(
    "/Volumes/shared/Sci-EAE/lahuang_work/working/"
    "Fastice_analyse_m3_results"
)

batch_root = base_folder / "Sentinel"
batch_export_root = base_folder / "time_series_figures_local"

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

    coherence_arrays = []
    group_code_arrays = []
    date_labels = []
    coherence_statistics = []

    for group_index, item in enumerate(datasets):

        with rasterio.open(
            item["coherence_only_tif"]
        ) as source:

            values = source.read(
                1,
                masked=True
            ).compressed()

        values = values[np.isfinite(values)]
        values = values[
            (values >= 0) & (values <= 1)
        ]

        # Use float32 to reduce memory
        values = values.astype(
            np.float32,
            copy=False
        )

        date_label = item[
            "first_time"
        ].strftime("%Y-%m-%d")

        date_labels.append(date_label)

        coherence_statistics.append(
            {
                "date": date_label,
                "mean": np.mean(
                    values,
                    dtype=np.float64
                ),
                "std": np.std(
                    values,
                    dtype=np.float64
                ),
            }
        )

        coherence_arrays.append(values)

        # Store small integer codes instead of repeating date strings
        group_code_arrays.append(
            np.full(
                values.size,
                group_index,
                dtype=np.int16
            )
        )


    # Combine the arrays only once
    all_coherence = np.concatenate(
        coherence_arrays
    )

    all_group_codes = np.concatenate(
        group_code_arrays
    )


    # Use categorical dates to avoid storing one date string per pixel
    boxplot_data = pd.DataFrame(
        {
            "date": pd.Categorical.from_codes(
                all_group_codes,
                categories=date_labels,
                ordered=True
            ),
            "coherence": all_coherence,
        }
    )

    statistics_data = pd.DataFrame(
        coherence_statistics
    )


    # Plot
    sns.set_theme(style="whitegrid")

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


    # Add mean ± one standard deviation
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


    # -----------------------------------------------------
    # Interactive time-series viewers
    # -----------------------------------------------------

    def date_heading(item):

        return pn.pane.Markdown(
            f"## {short_date(item)}",
            width=IMAGE_WIDTH
        )


    def png_pane(image_path):

        return pn.pane.PNG(
            str(image_path),
            width=IMAGE_WIDTH,
            sizing_mode="fixed"
        )


    def coherence_histogram(tif_path):

        with rasterio.open(tif_path) as source:

            values = source.read(
                1,
                masked=True
            ).compressed()

        values = values[np.isfinite(values)]
        values = values[
            (values >= 0) & (values <= 1)
        ]

        fig, axis = plt.subplots(
            figsize=(9, 4)
        )

        axis.hist(
            values,
            bins=50,
            range=(0, 1),
            color="steelblue",
            edgecolor="black",
            linewidth=0.4
        )

        axis.set_xlim(0, 1)
        axis.set_xlabel("Coherence")
        axis.set_ylabel("Number of pixels")
        axis.grid(alpha=0.25)
        fig.tight_layout()

        # The figure remains available to the Panel pane,
        # while preventing it from appearing separately.
        plt.close(fig)

        return fig


    def build_viewer(show_function):

        frame_player = pn.widgets.Player(
            name="Frame",
            start=0,
            end=len(datasets) - 1,
            value=0,
            step=1,
            interval=1000,
            loop_policy="loop",
            width=IMAGE_WIDTH
        )

        frame_view = pn.bind(
            show_function,
            frame=frame_player
        )

        return pn.Column(
            frame_player,
            frame_view,
            width=IMAGE_WIDTH
        )


    # -----------------------------------------------------
    # Viewer 1: filtered phase and amplitude
    # -----------------------------------------------------

    def show_phase_amplitude(frame):

        item = datasets[frame]

        return pn.Column(

            date_heading(item),

            pn.pane.Markdown(
                "### Filtered topophase with fast-ice boundary"
            ),

            png_pane(item["phase_boundary"]),

            pn.pane.Markdown(
                "### Amplitude with fast-ice boundary"
            ),

            png_pane(item["amplitude_boundary"]),

        )


    phase_amplitude_viewer = build_viewer(
        show_phase_amplitude
    )


    # -----------------------------------------------------
    # Viewer 2: coherence images and histogram
    # -----------------------------------------------------

    def show_coherence(frame):

        item = datasets[frame]

        histogram = coherence_histogram(
            str(item["coherence_only_tif"])
        )

        return pn.Column(

            date_heading(item),

            pn.pane.Markdown(
                "### Coherence with fast-ice boundary"
            ),

            png_pane(item["coherence_boundary"]),

            pn.pane.Markdown(
                "### Coherence inside the fast-ice area"
            ),

            png_pane(item["coherence_only_png"]),

            pn.pane.Markdown(
                "### Coherence value distribution"
            ),

            pn.pane.Matplotlib(
                histogram,
                width=IMAGE_WIDTH,
                tight=True
            ),

        )


    coherence_viewer = build_viewer(
        show_coherence
    )


    # In a notebook, either viewer can optionally be displayed with:
    # display(pn.ipywidget(phase_amplitude_viewer))
    # display(pn.ipywidget(coherence_viewer))


    # -----------------------------------------------------
    # Save interactive HTML viewers
    # -----------------------------------------------------

    phase_amplitude_html = (
        export_folder
        / "interactive_phase_amplitude.html"
    )

    coherence_html = (
        export_folder
        / "interactive_coherence.html"
    )

    phase_amplitude_viewer.save(
        phase_amplitude_html,
        embed=True,
        resources="inline",
        embed_json=False,
        max_states=50,
        max_opts=len(datasets)
    )

    coherence_viewer.save(
        coherence_html,
        embed=True,
        resources="inline",
        embed_json=False,
        max_states=50,
        max_opts=len(datasets)
    )

    print(f"Saved: {phase_amplitude_html}")
    print(f"Saved: {coherence_html}")


    # Release memory before processing the next InSAR folder
    del boxplot_data
    del statistics_data
    del all_coherence
    del all_group_codes
    del coherence_arrays
    del group_code_arrays

    import gc
    gc.collect()

    print(f"Saved: {boxplot_output}")
