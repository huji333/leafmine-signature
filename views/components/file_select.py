
from __future__ import annotations

from collections.abc import Callable

import gradio as gr


ChoicesProvider = Callable[[], list[str]]


def file_selector(
    *,
    label: str,
    choices_provider: ChoicesProvider,
    allow_custom_value: bool = True,
    refresh_label: str = "Refresh",
) -> tuple[gr.Dropdown, gr.Button]:
    """Render a dropdown + refresh button bound to the provided choices."""

    choices = choices_provider()
    initial_value = choices[0] if choices else None
    dropdown = gr.Dropdown(
        label=label,
        choices=choices,
        value=initial_value,
        allow_custom_value=allow_custom_value,
    )
    refresh_button = gr.Button(refresh_label)

    def _refresh(current_value: str | None) -> gr.Dropdown:
        fresh = choices_provider()
        if current_value and (allow_custom_value or current_value in fresh):
            value = current_value
        else:
            value = fresh[0] if fresh else None
        return gr.update(choices=fresh, value=value)

    refresh_button.click(fn=_refresh, inputs=[dropdown], outputs=[dropdown])
    return dropdown, refresh_button


__all__ = ["file_selector"]
