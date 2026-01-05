from __future__ import annotations

from functools import partial

import gradio as gr

from controllers.skeletonization import (
    DEFAULT_SKELETON_CONFIG,
    SkeletonizationConfig,
    process_mask,
    resolve_mask_source,
)
from controllers.data_paths import DataPaths
from views.components import file_selector
from views.config import DataBrowser, resolve_runtime_paths
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from skimage.measure import label

def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    """Upload or reuse segmented masks and inspect the resulting skeleton."""

    cfg, browser = resolve_runtime_paths(data_paths, data_browser)

    gr.Markdown(
        "Upload a **segmented binary mask** or point to an existing "
        "`data/segmented/segmented_*.png`. Uploaded files are persisted with the "
        "canonical `segmented_` prefix automatically. Use the controls below to tune "
        "preprocessing before moving on to routing/log-signatures."
    )

    upload_input = gr.File(
        label="Upload segmented mask",
        file_types=["image"],
        file_count="single",
    )
    with gr.Row():
        segmented_selector = file_selector(
            label="Or pick an existing segmented filename",
            choices_provider=browser.segmented,
            refresh_label="Refresh segmented list",
        )
        existing_selector = segmented_selector.dropdown

    gr.Markdown("### Skeletonization Settings")
    smooth_radius_input = gr.Slider(
        minimum=0,
        maximum=10,
        value=DEFAULT_SKELETON_CONFIG.smooth_radius,
        step=1,
        label="Closing Radius (smooth rough edges)",
    )
    hole_area_input = gr.Slider(
        minimum=0,
        maximum=2000,
        value=DEFAULT_SKELETON_CONFIG.hole_area_threshold,
        step=10,
        label="Hole Fill Area (px^2) - fill black holes up to this size",
    )
    erode_radius_input = gr.Slider(
        minimum=0,
        maximum=5,
        value=DEFAULT_SKELETON_CONFIG.erode_radius,
        step=1,
        label="Erosion Radius (separate touching parts)",
    )

    run_button = gr.Button("Run skeletonization", variant="primary")

    components_output = gr.Markdown("")

    with gr.Row():
        mask_preview = gr.Image(label="Stored Mask", type="pil")
        skeleton_overlay_preview = gr.Image(
            label="Skeleton Overlay",
            type="pil",
        )

    status_output = gr.Markdown("")

    run_inputs = [
        existing_selector,
        upload_input,
        smooth_radius_input,
        hole_area_input,
        erode_radius_input,
    ]
    run_outputs = [
        components_output,
        mask_preview,
        skeleton_overlay_preview,
        status_output,
        existing_selector,
    ]

    run_button.click(
        fn=partial(_handle_skeletonization, cfg, browser),
        inputs=run_inputs,
        outputs=run_outputs,
        show_progress=True,
    )
    # file_selector already wires refresh behavior; no extra click handler needed.


def _handle_skeletonization(
    data_paths: DataPaths,
    data_browser: DataBrowser,
    selected_filename: str | None,
    uploaded_file: str | None,
    smooth_radius: float | int,
    hole_area: float | int,
    erode_radius: float | int,
):
    """Persist the mask if needed, run skeletonization, and surface diagnostics."""

    data_paths.ensure_directories()
    try:
        mask_image, source_name, existing_mask_path = resolve_mask_source(
            data_paths,
            uploaded_file,
            selected_filename,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    config = SkeletonizationConfig(
        smooth_radius=int(smooth_radius),
        hole_area_threshold=int(hole_area),
        erode_radius=int(erode_radius),
    )

    persist_mask = uploaded_file is not None
    try:
        result = process_mask(
            mask_image,
            original_name=source_name,
            data_paths=data_paths,
            config=config,
            persist_mask=persist_mask,
            existing_mask_path=existing_mask_path,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    overlay = _render_skeleton_overlay(result.mask_image, result.skeleton_image)
    component_msg = _summarize_components(result.skeleton_image)
    status = (
        f"Saved `{result.mask_path.name}` -> `{result.skeleton_path.name}` "
        f"(closing={config.smooth_radius}, hole<={config.hole_area_threshold}, "
        f"erode={config.erode_radius})."
    )
    dropdown_update = gr.update(
        choices=data_browser.segmented(),
        value=result.mask_path.name,
    )

    return (
        component_msg,
        result.mask_image,
        overlay,
        status,
        dropdown_update,
    )


def _render_skeleton_overlay(mask_image: Image.Image, skeleton_image: Image.Image) -> Image.Image:
    """Overlay a thickened skeleton on the mask for quick QA."""

    base = Image.new("RGBA", mask_image.size, (0, 0, 0, 255))

    mask_l = ImageOps.autocontrast(mask_image)
    mask_alpha = mask_l.point(lambda value: int(180 if value > 0 else 0))
    mask_overlay = Image.new("RGBA", mask_image.size, (180, 180, 180, 0))
    mask_overlay.putalpha(mask_alpha)
    combined = Image.alpha_composite(base, mask_overlay)

    thick = skeleton_image.filter(ImageFilter.MaxFilter(size=5))
    skeleton_alpha = thick.point(lambda value: 255 if value > 0 else 0)
    skeleton_overlay = Image.new("RGBA", mask_image.size, (255, 64, 64, 0))
    skeleton_overlay.putalpha(skeleton_alpha)
    combined = Image.alpha_composite(combined, skeleton_overlay)

    return combined.convert("RGB")


def _summarize_components(skeleton_image: Image.Image) -> str:
    binary = np.asarray(skeleton_image, dtype=np.uint8) > 0
    if not binary.any():
        return "#### Connected components: 0  \n(no skeleton pixels detected)"

    labeled = label(binary, connectivity=2)
    components = int(labeled.max())
    if components <= 1:
        return "#### Connected components: 1  \n(single continuous skeleton)"
    return (
        f"### ⚠️ Connected components: {components}\n"
        "Multiple disjoint branches detected — consider easing preprocessing."
    )


__all__ = ["render"]
