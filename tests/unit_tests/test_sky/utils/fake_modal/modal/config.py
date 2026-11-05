"""The part of modal.config that SkyPilot reads: the active token.

Only from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, which is all that the real SDK
reads inside a container (MODAL_IS_REMOTE=1), where it ignores ~/.modal.toml.
"""
import os
from typing import Optional


class _Config:

    def get(self, key: str) -> Optional[str]:
        return os.environ.get(f'MODAL_{key.upper()}')


config = _Config()
