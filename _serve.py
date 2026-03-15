"""Temporary launcher for stress testing."""
import panel as pn
from deeplens.dashboard.app import DeepLensDashboard

pn.extension("tabulator", sizing_mode="stretch_width")
dashboard = DeepLensDashboard.create(dataset="iris")
template = dashboard.__panel__()
template.servable()
