#!/usr/bin/env python3
"""Build the locked ds005385 template EEG forward model.

This is a HARD GATE before EEG signal analysis.
It requires MNE >= 1.13.2 because `fsaverage_1005` was added in MNE 1.13.
The dataset has channel names but no individual electrodes.tsv / coordsystem.json,
so subject-specific coregistration is impossible from ds005385 alone.

Outputs
-------
- fsaverage_DK68_fixed_forward.fif
- leadfield_DK68_64x68.csv
- DK68_regional_surface_area_mm2.csv
- leadfield_U20.npy
- leadfield_build_manifest.json
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from packaging.version import Version
import mne

CHANNELS = [x.strip() for x in (Path(__file__).with_name('channels_64.txt')).read_text().splitlines() if x.strip()]
NETWORK_ORDER = [x.strip() for x in (Path(__file__).with_name('DK68_NETWORK_ORDER.txt')).read_text().splitlines() if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def vertex_areas(rr: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tri = rr[tris]
    a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    out = np.zeros(len(rr), float)
    for k in range(3):
        np.add.at(out, tris[:, k], a / 3.0)
    return out


def canonical_mne_label_from_network(name: str) -> str:
    if name.startswith('L_'):
        return name[2:] + '-lh'
    if name.startswith('R_'):
        return name[2:] + '-rh'
    raise RuntimeError(f'Invalid frozen DK68 network label: {name}')


def clean_dk_labels(labels):
    keep = []
    for lab in labels:
        low = lab.name.lower()
        if 'unknown' in low or 'corpuscallosum' in low:
            continue
        keep.append(lab)
    if len(keep) != 68:
        raise RuntimeError(f'Expected 68 cortical DK labels after exclusions, got {len(keep)}')
    by_name = {}
    for lab in keep:
        if lab.name in by_name:
            raise RuntimeError(f'Duplicate DK label from MNE: {lab.name}')
        by_name[lab.name] = lab
    expected = [canonical_mne_label_from_network(x) for x in NETWORK_ORDER]
    missing = [x for x in expected if x not in by_name]
    extra = sorted(set(by_name) - set(expected))
    if missing or extra:
        raise RuntimeError(f'DK68 MNE/network label mismatch; missing={missing}, extra={extra}')
    return [by_name[x] for x in expected]


def build(out_dir: Path, subjects_dir: Path | None) -> None:
    if Version(mne.__version__) < Version('1.13.2'):
        raise RuntimeError(f'Confirmatory gate requires MNE >= 1.13.2; found {mne.__version__}.')
    if 'fsaverage_1005' not in mne.channels.get_builtin_montages():
        raise RuntimeError('fsaverage_1005 montage unavailable; do not substitute another montage in confirmatory mode.')

    out_dir.mkdir(parents=True, exist_ok=True)
    if subjects_dir is None:
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=True))
        subjects_dir = fs_dir.parent
    else:
        subjects_dir = subjects_dir.expanduser().resolve()
        fs_dir = subjects_dir / 'fsaverage'

    src_fname = fs_dir / 'bem' / 'fsaverage-ico-5-src.fif'
    bem_fname = fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    for required in (src_fname, bem_fname):
        if not required.exists():
            raise FileNotFoundError(f'Missing required official fsaverage asset: {required}')

    info = mne.create_info(CHANNELS, sfreq=250.0, ch_types='eeg')
    montage = mne.channels.make_standard_montage('fsaverage_1005')
    missing = sorted(set(CHANNELS) - set(montage.ch_names))
    if missing:
        raise RuntimeError(f'Locked dataset channels absent from fsaverage_1005: {missing}')
    info.set_montage(montage, on_missing='raise')

    fwd = mne.make_forward_solution(
        info,
        trans='fsaverage',
        src=str(src_fname),
        bem=str(bem_fname),
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=True,
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=True)
    fwd_fname = out_dir / 'fsaverage_DK68_fixed_forward.fif'
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    G = np.asarray(fwd['sol']['data'], float)
    if G.shape[0] != 64:
        raise RuntimeError(f'Expected 64 EEG channels in forward solution, got {G.shape[0]}')

    P = np.eye(64) - np.ones((64, 64)) / 64.0
    G = P @ G

    labels = clean_dk_labels(mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir))
    src = fwd['src']
    lh_vert = np.asarray(src[0]['vertno'], int)
    rh_vert = np.asarray(src[1]['vertno'], int)
    lh_map = {int(v): i for i, v in enumerate(lh_vert)}
    rh_map = {int(v): len(lh_vert) + i for i, v in enumerate(rh_vert)}

    rr_lh, tri_lh = mne.read_surface(fs_dir / 'surf' / 'lh.white')
    rr_rh, tri_rh = mne.read_surface(fs_dir / 'surf' / 'rh.white')
    area_lh = vertex_areas(rr_lh, tri_lh)
    area_rh = vertex_areas(rr_rh, tri_rh)

    columns = []
    areas_mm2 = []
    names = []
    for lab in labels:
        if lab.hemi == 'lh':
            mapping, areas = lh_map, area_lh
        elif lab.hemi == 'rh':
            mapping, areas = rh_map, area_rh
        else:
            raise RuntimeError(f'Unexpected hemisphere for {lab.name}: {lab.hemi}')
        verts = [int(v) for v in lab.vertices if int(v) in mapping]
        if not verts:
            raise RuntimeError(f'No source-space vertices found for {lab.name}')
        idx = np.array([mapping[v] for v in verts], int)
        w = np.array([areas[v] for v in verts], float)
        col = G[:, idx] @ w
        columns.append(col)
        areas_mm2.append(float(w.sum()))
        names.append(lab.name)

    L = np.column_stack(columns)
    areas_mm2 = np.asarray(areas_mm2)
    L /= np.median(areas_mm2)
    L = P @ L
    if np.linalg.matrix_rank(L) > 63:
        raise RuntimeError('Average-referenced leadfield rank unexpectedly exceeds 63.')

    lead_csv = out_dir / 'leadfield_DK68_64x68.csv'
    pd.DataFrame(L, index=CHANNELS, columns=names).to_csv(lead_csv)
    area_csv = out_dir / 'DK68_regional_surface_area_mm2.csv'
    pd.DataFrame({'label': names, 'surface_area_mm2': areas_mm2}).to_csv(area_csv, index=False)

    U, s, _ = np.linalg.svd(L, full_matrices=False)
    U20 = U[:, :20]
    u20_fname = out_dir / 'leadfield_U20.npy'
    np.save(u20_fname, U20)

    manifest = {
        'mne_version': mne.__version__,
        'montage': 'fsaverage_1005',
        'n_channels': 64,
        'n_regions': 68,
        'network_order': NETWORK_ORDER,
        'mne_aparc_label_order': names,
        'leadfield_rank': int(np.linalg.matrix_rank(L)),
        'U20_orthogonality_max_abs_error': float(np.max(np.abs(U20.T @ U20 - np.eye(20)))),
        'singular_values': s.tolist(),
        'files_sha256': {
            fwd_fname.name: sha256(fwd_fname),
            lead_csv.name: sha256(lead_csv),
            area_csv.name: sha256(area_csv),
            u20_fname.name: sha256(u20_fname),
            'official_fsaverage_src': sha256(src_fname),
            'official_fsaverage_bem': sha256(bem_fname),
            'DK68_NETWORK_ORDER.txt': sha256(Path(__file__).with_name('DK68_NETWORK_ORDER.txt')),
        },
        'critical_note': 'Dataset lacks individual electrode digitization. This is a template-MRI/template-montage forward model, not subject-specific source localization.'
    }
    (out_dir / 'leadfield_build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--subjects-dir', type=Path, default=None)
    args = ap.parse_args()
    build(args.out_dir, args.subjects_dir)
