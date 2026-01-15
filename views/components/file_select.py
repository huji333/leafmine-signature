
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gradio as gr

from controllers.artifact_status import ActionType, ProcessingStatusService

Choice = str | tuple[str, str]
ChoicesProvider = Callable[[], list[Choice]]


@dataclass(slots=True)
class FileSelectorComponents:
    dropdown: gr.Dropdown
    refresh_button: gr.Button
    choices_provider: ChoicesProvider


def file_selector(
    *,
    label: str,
    choices_provider: ChoicesProvider,
    allow_custom_value: bool = True,
    refresh_label: str = "Refresh",
    status_service: ProcessingStatusService | None = None,
    action_type: ActionType | None = None,
    status_badge: str | None = "✅",
) -> FileSelectorComponents:
    """Render a dropdown + refresh button bound to the provided choices."""

    def _materialize_choices() -> list[Choice]:
        choices = choices_provider()
        if status_service and action_type:
            return _annotate_choices(
                choices,
                status_service,
                action_type,
                status_badge=status_badge,
            )
        return choices

    choices = _materialize_choices()
    initial_value = _first_choice_value(choices)
    dropdown = gr.Dropdown(
        label=label,
        choices=choices,
        value=initial_value,
        allow_custom_value=allow_custom_value,
    )
    refresh_button = gr.Button(refresh_label)

    def _refresh(current_value: str | None) -> gr.Dropdown:
        fresh = _materialize_choices()
        fresh_values = _choice_values(fresh)
        if current_value and (allow_custom_value or current_value in fresh_values):
            value = current_value
        else:
            value = _first_choice_value(fresh)
        return gr.update(choices=fresh, value=value)

    refresh_button.click(fn=_refresh, inputs=[dropdown], outputs=[dropdown])
    return FileSelectorComponents(
        dropdown=dropdown,
        refresh_button=refresh_button,
        choices_provider=_materialize_choices,
    )


def _choice_values(choices: list[Choice]) -> list[str]:
    values: list[str] = []
    for choice in choices:
        if isinstance(choice, tuple):
            values.append(choice[1])
        else:
            values.append(choice)
    return values


def _first_choice_value(choices: list[Choice]) -> str | None:
    if not choices:
        return None
    first = choices[0]
    return first[1] if isinstance(first, tuple) else first


def _annotate_choices(
    choices: list[Choice],
    status_service: ProcessingStatusService,
    action_type: ActionType,
    *,
    status_badge: str | None,
) -> list[Choice]:
    status_service.refresh_action(action_type)
    pending: list[tuple[str, str]] = []
    complete: list[tuple[str, str]] = []
    badge = status_badge or ""
    for choice in choices:
        if isinstance(choice, tuple):
            label, value = choice
        else:
            label = value = choice
        is_done = status_service.is_done(value, action_type)
        if is_done and badge:
            rendered_label = f"{label} {badge}".rstrip()
        elif is_done:
            rendered_label = label
        else:
            rendered_label = label
        target = complete if is_done else pending
        target.append((rendered_label, value))
    return pending + complete


__all__ = ["file_selector", "FileSelectorComponents"]
