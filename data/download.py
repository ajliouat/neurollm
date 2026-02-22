"""Download helpers for EEG datasets.

Both TUH and BCI-IV require manual steps (credentials / license agreement).
This module provides download instructions and, where possible, automated
retrieval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ── BCI Competition IV Dataset 2a ────────────────────────────────────

BCI_IV_URL = (
    "https://www.bbci.de/competition/iv/download/BCICIV_2a_gdf.tar.gz"
)
BCI_IV_LABELS_URL = (
    "https://www.bbci.de/competition/iv/download/BCICIV_2a_labels.tar.gz"
)


def download_bci_iv(dest: str | Path) -> None:
    """Print instructions for BCI Competition IV Dataset 2a.

    Parameters
    ----------
    dest : path-like
        Destination directory for downloaded files.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(
        "BCI Competition IV Dataset 2a\n"
        "─────────────────────────────\n"
        "1.  Visit https://www.bbci.de/competition/iv/\n"
        "2.  Download 'Dataset 2a' (GDF format).\n"
        f"    Direct link: {BCI_IV_URL}\n"
        f"    Labels:      {BCI_IV_LABELS_URL}\n"
        f"3.  Extract into: {dest}\n"
        "4.  Expected files: A01T.gdf … A09T.gdf, A01E.gdf … A09E.gdf\n"
        "5.  Run `python -m data.preprocess_bci_iv` to create .npz files.\n"
    )


# ── TUH EEG Corpus ──────────────────────────────────────────────────

TUH_BASE_URL = "https://isip.piconepress.com/projects/tuh_eeg/"


def download_tuh(dest: str | Path) -> None:
    """Print instructions for TUH EEG Corpus access.

    Parameters
    ----------
    dest : path-like
        Destination directory.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(
        "Temple University Hospital (TUH) EEG Corpus\n"
        "─────────────────────────────────────────────\n"
        f"1.  Request access at: {TUH_BASE_URL}\n"
        "2.  You will receive credentials via email.\n"
        "3.  Use rsync or the provided download scripts to fetch data.\n"
        f"4.  Place .edf files in: {dest}\n"
        "5.  Run `python -m data.preprocess_tuh` to create .npz epoch files.\n"
    )


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Show download instructions for both datasets."""
    base = Path("data/raw")
    print()
    download_bci_iv(base / "bci_iv")
    print()
    download_tuh(base / "tuh")
    print()


if __name__ == "__main__":
    main()
