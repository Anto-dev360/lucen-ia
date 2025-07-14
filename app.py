"""
app.py

Lightning App for LucenAI: connects a FastAPI backend for sentiment prediction
and a static frontend interface for visualization and interaction.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

from lightning.app import LightningApp, LightningFlow
from lightning.app.components.serve import Serve
from lightning.app.components import StaticWebFrontend


class LucenAIApp(LightningFlow):
    """
    LightningFlow that defines the LucenAI application with:
    - A FastAPI backend component for prediction (`/predict`)
    - A static frontend component serving HTML/JS/CSS
    """

    def __init__(self):
        super().__init__()
        self.backend = Serve(script_path="scripts/serve_api.py", port=8000)
        self.frontend = StaticWebFrontend(source_dir="lucenai/frontend", port=80)

    def run(self):
        """
        Defines execution logic of the app. Both components run concurrently.
        """
        self.backend.run()
        self.frontend.run()


# Entry point for Lightning CLI
app = LightningApp(LucenAIApp())
