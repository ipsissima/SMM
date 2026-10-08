#!/usr/bin/env python3
"""Self-contained synthetic tests for the read-only Step 5B EDF incident auditor."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent


def padded(value, width):
    b = str(value).encode('latin-1')
    assert len(b) <= width
    return b.ljust(width, b' ')


with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    labels = ['Fp2', 'Fp1', 'Status']
    samples = [
        np.array([[-100, -100], [100, 100]], dtype='<i2'),
        np.array([[3, 4], [5, 6]], dtype='<i2'),
        np.array([[0, 0], [1, 1]], dtype='<i2'),
    ]
    fixed_header = b''.join([
        padded('0', 8), padded('', 80), padded('', 80),
        padded('01.01.01', 8), padded('01.02.03', 8),
        padded(256 * (len(labels) + 1), 8), padded('', 44),
        padded(2, 8), padded(10, 8), padded(len(labels), 4),
    ])
    assert len(fixed_header) == 256
    fields = [
        (16, labels), (80, [''] * 3), (8, ['uV'] * 3),
        (8, [-10] * 3), (8, [10] * 3),
        (8, [-100] * 3), (8, [100] * 3),
        (80, [''] * 3), (8, [2] * 3), (32, [''] * 3),
    ]
    signal_header = b''.join(padded(v, w) for w, vals in fields for v in vals)
    assert len(signal_header) == 256 * len(labels)
    data = b''.join(np.concatenate([channel[record] for channel in samples]).tobytes()
                    for record in range(2))
    edf = d / 'test.edf'
    edf.write_bytes(fixed_header + signal_header + data)
    (d / 'bids.json').write_text(json.dumps({
        'RecordingDuration': 20, 'EEGChannelCount': 2, 'SamplingFrequency': 0.2
    }))
    (d / 'bids.tsv').write_text('name\ttype\tunits\nFp2\tEEG\tuV\nFp1\tEEG\tuV\n')
    (d / 'frozen.txt').write_text('Fp1\nFp2\n')
    command = [
        sys.executable, str(ROOT / 'edf_header_audit.py'),
        '--subject', 'sub-test',
        '--edf', str(edf),
        '--bids-json', str(d / 'bids.json'),
        '--bids-channels', str(d / 'bids.tsv'),
        '--frozen-channels', str(d / 'frozen.txt'),
        '--sha256', hashlib.sha256(edf.read_bytes()).hexdigest(),
        '--byte-size', str(edf.stat().st_size),
        '--dataset-commit', 'synthetic-test',
        '--output', str(d / 'report.json'),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    report = json.loads((d / 'report.json').read_text())
    assert report['edf_duration_seconds'] == 20
    assert report['edf_eeg_names_same_frozen_set'] is True
    assert report['only_fp1_fp2_swapped'] is True
    assert report['edf_eeg_names_equal_bids_eeg_names'] is True
    assert report['max_fraction_outside_declared_digital_range'] == 0
    first = next(ch for ch in report['edf_header']['signals'] if ch['name'] == 'Fp2')
    assert first['fraction_at_digital_min'] == 0.5
    assert first['fraction_at_digital_max'] == 0.5
    assert first['longest_digital_min_run_samples'] == 2
    assert first['longest_digital_max_run_samples'] == 2
    assert report['pinned_digest_and_byte_size_pass'] is True
    command[command.index('--sha256') + 1] = '0' * 64
    tampered = subprocess.run(command, capture_output=True, text=True)
    assert tampered.returncode != 0 and 'does not match' in tampered.stderr
    print('PASS: independent EDF parsing, channel permutation, input-integrity rejection')
