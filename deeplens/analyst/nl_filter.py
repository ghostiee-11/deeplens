"""Natural Language Data Filter.

Type natural language queries like "show me samples where the model
is confident but wrong" and the LLM converts them to pandas queries.
"""

from __future__ import annotations

import pandas as pd
import param
import panel as pn

from deeplens.analyst.llm import LLMProvider, create_llm, sanitize_expression


_FILTER_SYSTEM = """You are a data filter assistant. The user describes a filter condition
in natural language. Convert it to a valid pandas DataFrame query string.

IMPORTANT RULES:
- Return ONLY the query string, nothing else
- Use pandas query syntax (e.g., "confidence > 0.9 and prediction != label")
- Column names available: {columns}
- If the query cannot be expressed, return "CANNOT_FILTER"
- Do NOT include df.query() wrapper, just the condition string
- Use only column names from the list above
- Do NOT use Python functions, imports, or method calls

Examples:
- "high confidence wrong predictions" → "confidence > 0.8 and prediction != label"
- "class A samples" → "label == 'A'"
- "uncertain predictions" → "confidence < 0.5"
"""


class NLFilter(pn.viewable.Viewer):
    """Natural language data filtering via LLM.

    Features
    --------
    - Type English → LLM converts to pandas query
    - Query history with re-run capability
    - Suggested queries based on current data
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")
    llm = param.ClassSelector(class_=LLMProvider, doc="LLM provider instance")

    def __init__(self, **params):
        if "llm" not in params or params["llm"] is None:
            params["llm"] = create_llm("none")
        super().__init__(**params)
        self._query_input = pn.widgets.TextInput(
            name="Natural Language Filter",
            placeholder="e.g., 'show me confident but wrong predictions'",
            sizing_mode="stretch_width",
        )
        self._apply_btn = pn.widgets.Button(
            name="Apply Filter", button_type="primary",
            sizing_mode="stretch_width",
        )
        self._clear_btn = pn.widgets.Button(
            name="Clear Filter", button_type="warning",
            sizing_mode="stretch_width",
        )
        self._status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._suggestions_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._history: list[dict] = []
        self._history_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Query", "Pandas Expression", "Matches"]),
            sizing_mode="stretch_width",
            height=150,
            show_index=False,
        )
        self._show_history_btn = pn.widgets.Button(
            name="Show History & Suggestions",
            button_type="light",
            icon="history",
            sizing_mode="stretch_width",
        )
        self._detail_panel = None
        self._show_history_btn.on_click(self._toggle_detail)

        self._apply_btn.on_click(self._on_apply)
        self._clear_btn.on_click(self._on_clear)

        # Update suggestions when dataset or model changes
        if self.state is not None:
            self.state.param.watch(self._refresh_suggestions, ["df", "predictions", "cluster_labels"])
            self._refresh_suggestions()

    async def _on_apply(self, event=None):
        """Convert NL query to pandas filter and apply."""
        query_text = self._query_input.value.strip()
        if not query_text or self.state is None or self.state.df is None:
            self._status.object = "*Enter a query and ensure data is loaded.*"
            return

        columns = list(self.state.df.columns)
        system = _FILTER_SYSTEM.format(columns=columns)
        messages = [{"role": "user", "content": query_text}]

        # Get pandas expression from LLM
        expression = ""
        async for chunk in self.llm.stream(messages, system=system):
            expression += chunk

        expression = expression.strip()

        if expression == "CANNOT_FILTER" or not expression:
            self._status.object = f"**Could not convert:** '{query_text}'"
            return

        # Sanitize expression to prevent code injection
        try:
            expression = sanitize_expression(expression, columns)
        except ValueError as e:
            self._status.object = f"**Unsafe expression rejected:** {e}"
            return

        # Apply filter
        try:
            filtered = self.state.df.query(expression)
            n_matches = len(filtered)
            # Convert index to list of ints for consistent state update
            self.state.selected_indices = list(filtered.index)
            self._status.object = (
                f"**Filter:** `{expression}`\n\n"
                f"**Matches:** {n_matches} / {len(self.state.df)} samples"
            )

            self._history.append({
                "Query": query_text,
                "Pandas Expression": expression,
                "Matches": n_matches,
            })
            self._history_table.value = pd.DataFrame(self._history)

        except Exception as e:
            self._status.object = f"**Filter error:** `{expression}`\n\n{e}"

    def _on_clear(self, event=None):
        """Clear the current filter."""
        if self.state is not None:
            self.state.selected_indices = []
        self._status.object = "*Filter cleared.*"
        self._query_input.value = ""

    def _refresh_suggestions(self, event=None):
        """Update suggested queries based on current data."""
        if self.state is None or self.state.df is None:
            self._suggestions_pane.object = ""
            return

        suggestions = ["### Suggested Queries\n"]

        if self.state.predictions is not None and self.state.labels is not None:
            suggestions.append('- "confident but wrong predictions"')
            suggestions.append('- "uncertain predictions"')
            suggestions.append('- "misclassified samples"')

        if self.state.class_names:
            first_class = self.state.class_names[0]
            suggestions.append(f'- "all {first_class} samples"')

        if self.state.cluster_labels is not None:
            suggestions.append('- "cluster 0 samples"')

        self._suggestions_pane.object = "\n".join(suggestions)

    def _toggle_detail(self, event=None):
        """Toggle the detail panel with history and suggestions."""
        if self._detail_panel is None:
            self._detail_panel = pn.layout.FloatPanel(
                pn.Column(
                    self._suggestions_pane,
                    pn.layout.Divider(),
                    "### Query History",
                    self._history_table,
                    sizing_mode="stretch_width",
                ),
                name="NL Filter — History & Suggestions",
                contained=False,
                position="center",
                width=500,
                height=400,
            )
        self._detail_panel.visible = not getattr(self._detail_panel, "visible", False)

    def __panel__(self):
        compact = pn.Column(
            self._query_input,
            pn.Row(self._apply_btn, self._clear_btn),
            self._status,
            self._show_history_btn,
            sizing_mode="stretch_width",
        )
        # Attach the float panel if it exists (it gets created on first click)
        if self._detail_panel is not None:
            return pn.Column(compact, self._detail_panel, sizing_mode="stretch_width")
        return compact
