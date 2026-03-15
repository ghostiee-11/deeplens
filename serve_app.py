"""Serve the DeepLens dashboard for testing."""
import panel as pn
from deeplens.dashboard.app import DeepLensDashboard

dashboard = DeepLensDashboard.create(dataset="iris")
template = dashboard.__panel__()
template.servable()
