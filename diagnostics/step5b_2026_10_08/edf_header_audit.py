#!/usr/bin/env python3
"""Independent read-only EDF / BIDS audit for the Step 5B incident.

This tool neither executes models nor generates QC status or eligibility decisions.
It reads public pinned input data and records physical/digital EDF metadata and
raw digital counts without rescaling, reordering, filtering, or editing the EEG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _as_number(value, converter):
    return converter(value.strip())


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def edf_headers_and_counts(path: Path):
    """Parse EDF fixed/signal headers; inspect int16 data without modifying it."""
    with path.open('rb') as stream:
        head = stream.read(256)
        if len(head) != 256:
            raise ValueError('Short fixed EDF header')
        header_bytes = _as_number(head[184:192], int)
        record_count = _as_number(head[236:244], int)
        record_seconds = _as_number(head[244:252], float)
        signal_count = _as_number(head[252:256], int)
        if not (1 <= signal_count <= 256 and record_count > 0 and record_seconds > 0):
            raise ValueError('Invalid EDF signal count or recording length')
        if header_bytes != 256 * (signal_count + 1):
            raise ValueError('Unexpected EDF header size')
        raw_header = stream.read(256 * signal_count)
        if len(raw_header) != 256 * signal_count:
            raise ValueError('Short per-signal EDF header')
        position = 0

        def read_field(width):
            nonlocal position
            start = position
            position += width * signal_count
            return [raw_header[start+i*width:start+(i+1)*width].decode('latin-1').strip()
                    for i in range(signal_count)]

        names = read_field(16)
        transducers = read_field(80)
        units = read_field(8)
        physical_min = list(map(float, read_field(8)))
        physical_max = list(map(float, read_field(8)))
        digital_min = list(map(int, read_field(8)))
        digital_max = list(map(int, read_field(8)))
        prefilters = read_field(80)
        samples_per_record = list(map(int, read_field(8)))
        read_field(32)
        if position != 256 * signal_count or min(samples_per_record) <= 0:
            raise ValueError('Invalid per-signal EDF header')
        bytes_expected = header_bytes + record_count * sum(samples_per_record) * 2
        bytes_actual = path.stat().st_size
        if bytes_actual != bytes_expected:
            raise ValueError(f'EDF file length disagrees with header: {bytes_actual} vs {bytes_expected}')
        payload = stream.read()
    counts = np.frombuffer(payload, dtype='<i2').reshape(record_count, sum(samples_per_record))
    offsets = np.cumsum([0] + samples_per_record)
    outputs = []
    for i in range(signal_count):
        block = counts[:, offsets[i]:offsets[i+1]]
        digital_values = block.reshape(-1).astype(np.float64)
        denominator = digital_max[i] - digital_min[i]
        gain = (physical_max[i] - physical_min[i]) / denominator if denominator else None
        physical_values = None if gain is None else physical_min[i] + (digital_values - digital_min[i]) * gain
        units_norm = units[i].lower().replace('micro', 'u').replace('\u00b5', 'u').replace('\u03bc', 'u')
        outputs.append({
            'name': names[i], 'physical_dimension': units[i],
            'transducer': transducers[i], 'prefilter': prefilters[i],
            'physical_min': physical_min[i], 'physical_max': physical_max[i],
            'digital_min': digital_min[i], 'digital_max': digital_max[i],
            'header_gain_physical_units_per_count': gain,
            'sampling_frequency_hz': samples_per_record[i] / record_seconds,
            'n_samples': digital_values.size,
            'observed_digital_min': int(np.min(block)),
            'observed_digital_max': int(np.max(block)),
            'observed_digital_rms': float(np.sqrt(np.mean(digital_values ** 2))),
            'fraction_at_either_digital_limit': float(np.mean((block == digital_min[i]) | (block == digital_max[i]))),
            'fraction_outside_declared_digital_range': float(np.mean((block < digital_min[i]) | (block > digital_max[i]))),
            'raw_physical_median': None if physical_values is None else float(np.median(physical_values)),
            'raw_physical_rms': None if physical_values is None else float(np.sqrt(np.mean(physical_values ** 2))),
            'raw_physical_abs_p999': None if physical_values is None else float(np.percentile(np.abs(physical_values), 99.9)),
            'dimension_is_microvolts': units_norm in ('uv', 'uvolt', 'uvolts'),
        })
    return {
        'file_bytes': bytes_actual, 'header_bytes': header_bytes,
        'record_count': record_count, 'seconds_per_record': record_seconds,
        'duration_seconds': record_count * record_seconds,
        'signal_count': signal_count, 'signals': outputs,
    }


def run(args):
    actual_hash = _hash(args.edf)
    if actual_hash != args.sha256.lower() or args.edf.stat().st_size != args.byte_size:
        raise ValueError('EDF does not match pinned git-annex SHA-256/byte size')
    edf = edf_headers_and_counts(args.edf)
    bids_json = json.loads(args.bids_json.read_text(encoding='utf-8'))
    with args.bids_channels.open('r', encoding='utf-8-sig', newline='') as stream:
        bids_eeg = [r['name'] for r in csv.DictReader(stream, delimiter='\t') if r['type'].upper() == 'EEG']
    frozen = [line.strip() for line in args.frozen_channels.read_text().splitlines() if line.strip()]
    signal_names = [s['name'] for s in edf['signals']]
    eeg_names = [x for x in signal_names if x != 'Status']
    expected = set(frozen)
    matches_as_sets = (len(eeg_names) == len(frozen) == len(expected)
                       and set(eeg_names) == expected)
    differences = [{'position_1_indexed': i+1, 'frozen': a, 'edf': b}
                   for i, (a, b) in enumerate(zip(frozen, eeg_names)) if a != b]
    eeg = [s for s in edf['signals'] if s['name'] != 'Status']
    gains = [s['header_gain_physical_units_per_count'] for s in eeg if s['dimension_is_microvolts'] and s['header_gain_physical_units_per_count'] is not None]
    data = {
        'subject': args.subject,
        'audit_type': 'blind read-only EDF/BIDS forensic diagnostics, no fit or inference',
        'dataset_snapshot_commit': args.dataset_commit,
        'input_sha256': actual_hash,
        'pinned_digest_and_byte_size_pass': True,
        'bids_declared_duration_seconds': bids_json.get('RecordingDuration'),
        'bids_declared_eeg_channel_count': bids_json.get('EEGChannelCount'),
        'bids_declared_sampling_frequency_hz': bids_json.get('SamplingFrequency'),
        'edf_duration_seconds': edf['duration_seconds'],
        'edf_signal_names': signal_names,
        'edf_eeg_names_equal_bids_eeg_names': eeg_names == bids_eeg,
        'edf_eeg_channel_names': eeg_names,
        'bids_eeg_channel_names': bids_eeg,
        'frozen_channel_names': frozen,
        'edf_eeg_names_exact_frozen_order': eeg_names == frozen,
        'edf_eeg_names_same_frozen_set': matches_as_sets,
        'edf_frozen_order_mismatches': differences,
        'only_fp1_fp2_swapped': (matches_as_sets and len(differences) == 2
                                 and [r['position_1_indexed'] for r in differences] == [1, 2]
                                 and eeg_names[:2] == frozen[1::-1]),
        'header_gain_median_microvolts_per_count': float(np.median(gains)) if gains else None,
        'header_gain_min_microvolts_per_count': float(np.min(gains)) if gains else None,
        'header_gain_max_microvolts_per_count': float(np.max(gains)) if gains else None,
        'max_fraction_at_digital_limit': max(s['fraction_at_either_digital_limit'] for s in eeg),
        'max_fraction_outside_declared_digital_range': max(s['fraction_outside_declared_digital_range'] for s in eeg),
        'edf_header': edf,
        'nonintervention_statement': 'No signal reordering, amplitude correction, exclusion, model fitting, or inferential test performed.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    summary = {k: data[k] for k in (
        'subject', 'input_sha256', 'bids_declared_duration_seconds', 'edf_duration_seconds',
        'edf_eeg_names_equal_bids_eeg_names', 'edf_eeg_names_exact_frozen_order',
        'edf_eeg_names_same_frozen_set', 'edf_frozen_order_mismatches',
        'header_gain_median_microvolts_per_count', 'header_gain_min_microvolts_per_count',
        'header_gain_max_microvolts_per_count', 'max_fraction_at_digital_limit',
        'max_fraction_outside_declared_digital_range')}
    print('SMM_STEP5B_BLIND_EDF_AUDIT ' + json.dumps(summary, ensure_ascii=True))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--edf', required=True, type=Path)
    p.add_argument('--bids-json', required=True, type=Path)
    p.add_argument('--bids-channels', required=True, type=Path)
    p.add_argument('--frozen-channels', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--byte-size', required=True, type=int)
    p.add_argument('--dataset-commit', required=True)
    p.add_argument('--output', required=True, type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
