# ============================================================================
# TEST PIPELINE — fMRI Data Loader
# ============================================================================
"""Load parcellated fMRI BOLD data from HDF5 and Schaefer 17-network atlas.

Reads the Schaefer2018_1000Parcels_17Networks label file to build
parcel → network/sub-region mappings, then extracts DMN parcels from
the HDF5 BOLD data.

DMN sub-region → cognitive dimension mapping (from project literature):
  DefaultA_PFCm  → vmPFC (situation model / states_context)
  DefaultA_pCunPCC → PCC/precuneus (event boundary / memory)
  DefaultA_PFCd + DefaultB_PFCd → dmPFC (state transitions)
  DefaultB_PFCv + DefaultB_PFCl → amPFC/vlPFC (mentalizing / agents_social)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atlas parsing
# ---------------------------------------------------------------------------

def parse_schaefer_labels(
    label_path: str | Path,
) -> list[dict[str, str | int]]:
    """Parse Schaefer 17-network parcel label file.

    Format: ``index \\t label \\t R \\t G \\t B \\t A``
    where index is 1-based.

    Returns a list of dicts with keys:
      idx_0based, label, hemisphere, network, sub_region
    """

    parcels: list[dict[str, str | int]] = []
    with Path(label_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            idx_1based = int(parts[0])
            label = parts[1]

            # Parse: 17Networks_LH_DefaultA_PFCm_1
            comps = label.split("_")
            hemisphere = comps[1]             # LH or RH
            network = comps[2]                # e.g. DefaultA, VisCent
            sub_parts = comps[3:]             # e.g. ['PFCm', '1']
            # Sub-region is everything except the trailing number
            if sub_parts and sub_parts[-1].isdigit():
                sub_region = "_".join(sub_parts[:-1]) if len(sub_parts) > 1 else ""
            else:
                sub_region = "_".join(sub_parts)

            parcels.append({
                "idx_0based": idx_1based - 1,
                "label": label,
                "hemisphere": hemisphere,
                "network": network,
                "sub_region": sub_region,
            })

    logger.info("Parsed %d parcel labels", len(parcels))
    return parcels


def get_network_indices(
    parcels: list[dict[str, str | int]],
    network_prefix: str,
) -> np.ndarray:
    """Return 0-based parcel indices matching a network prefix.

    Examples: ``"Default"`` matches DefaultA, DefaultB, DefaultC.
    ``"DefaultA"`` matches only DefaultA.
    """

    return np.array([
        p["idx_0based"]
        for p in parcels
        if str(p["network"]).startswith(network_prefix)
    ])


def get_dmn_sub_regions(
    parcels: list[dict[str, str | int]],
) -> dict[str, np.ndarray]:
    """Return DMN sub-region parcel indices grouped by cognitive target.

    Returns
    -------
    dict mapping sub-region name → 0-based parcel indices:
      - ``"vmPFC"`` → DefaultA_PFCm
      - ``"PCC_precuneus"`` → DefaultA_pCunPCC
      - ``"dmPFC"`` → DefaultA_PFCd + DefaultB_PFCd
      - ``"amPFC_vlPFC"`` → DefaultB_PFCv + DefaultB_PFCl
      - ``"IPL"`` → DefaultA_IPL + DefaultB_IPL + DefaultC_IPL
      - ``"temporal"`` → DefaultA_Temp + DefaultB_Temp + DefaultB_AntTemp
      - ``"PHC_Rsp"`` → DefaultC_PHC + DefaultC_Rsp (memory system)
      - ``"all_DMN"`` → all Default parcels
    """

    def _idx(net: str, sub: str) -> list[int]:
        return [
            p["idx_0based"] for p in parcels
            if p["network"] == net and p["sub_region"] == sub
        ]

    regions: dict[str, np.ndarray] = {
        "vmPFC": np.array(_idx("DefaultA", "PFCm")),
        "PCC_precuneus": np.array(_idx("DefaultA", "pCunPCC")),
        "dmPFC": np.array(_idx("DefaultA", "PFCd") + _idx("DefaultB", "PFCd")),
        "amPFC_vlPFC": np.array(_idx("DefaultB", "PFCv") + _idx("DefaultB", "PFCl")),
        "IPL": np.array(
            _idx("DefaultA", "IPL") + _idx("DefaultB", "IPL") + _idx("DefaultC", "IPL")
        ),
        "temporal": np.array(
            _idx("DefaultA", "Temp") + _idx("DefaultB", "Temp") + _idx("DefaultB", "AntTemp")
        ),
        "PHC_Rsp": np.array(_idx("DefaultC", "PHC") + _idx("DefaultC", "Rsp")),
        "all_DMN": get_network_indices(parcels, "Default"),
    }

    for name, idx in regions.items():
        logger.info("  %s: %d parcels", name, len(idx))

    return regions


# ---------------------------------------------------------------------------
# BOLD loading
# ---------------------------------------------------------------------------

def load_bold(
    h5_path: str | Path,
    run_key: str | None = None,
) -> tuple[np.ndarray, str]:
    """Load BOLD time series from an HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    run_key : str, optional
        Dataset key (e.g., ``"ses-003_task-s01e01a"``).
        If *None*, auto-detects ``s01e01a`` from available keys.

    Returns
    -------
    bold : np.ndarray, shape (n_trs, n_parcels)
    run_key : str
    """

    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if run_key is None:
            all_keys = list(f.keys())
            candidates = [k for k in all_keys if "s01e01a" in k]
            if not candidates:
                candidates = all_keys
            if not candidates:
                raise ValueError(f"HDF5 file {h5_path} contains no datasets.")
            if len(candidates) > 1:
                logger.warning(
                    "Multiple BOLD run candidates in %s: %s. "
                    "Auto-selecting '%s'. Use --run-key to specify.",
                    h5_path, candidates, candidates[0],
                )
            else:
                logger.warning(
                    "No --run-key specified. Auto-selected '%s' from %s. "
                    "Available keys: %s",
                    candidates[0], h5_path, all_keys,
                )
            run_key = candidates[0]

        bold = f[run_key][:]

    logger.info(
        "Loaded BOLD: %s, shape=%s, mean=%.3f, std=%.3f",
        run_key, bold.shape, bold.mean(), bold.std(),
    )
    return bold, run_key


def train_test_split_trs(
    n_trs: int,
    train_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple temporal train/test split (first N% train, rest test)."""

    split = int(n_trs * train_ratio)
    return np.arange(split), np.arange(split, n_trs)
