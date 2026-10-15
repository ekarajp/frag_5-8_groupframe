"""Portable construction-only Grouped Frame loader; never runs an analysis."""
import gzip
import hashlib
import json
from pathlib import Path

CONSTRUCTION_COMMANDS = frozenset({
    'wipe', 'model', 'node', 'fix', 'mass', 'equalDOF', 'geomTransf',
    'uniaxialMaterial', 'section', 'patch', 'layer', 'fiber',
    'beamIntegration', 'element', 'timeSeries', 'pattern', 'load', 'eleLoad',
})

def read_model(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Model file checksum mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        model = json.load(handle)
    if model.get('schema') != 'portable-grouped-construction-v1':
        raise ValueError('Unsupported model schema')
    if model.get('analysis_performed') is not False:
        raise ValueError('Expected an unanalysed construction definition')
    if set(model['directions']) != {'X', 'Y'}:
        raise ValueError('Both directional definitions are required')
    for unit in model['directions'].values():
        commands = unit['commands']
        if not commands or commands[0] != ['wipe', []]:
            raise ValueError('Definition must start with a fresh domain')
        for name, args in commands:
            if name not in CONSTRUCTION_COMMANDS or not isinstance(args, list):
                raise ValueError('Forbidden or malformed construction command: ' + name)
    return model

def build_model(path, direction, ops, expected_sha256=None):
    """Populate one fresh OpenSees domain; gravity/damping/analysis are caller work.

    Each call wipes the previous domain. X/Y must be analysed separately using
    the selected recipe; this function does not initialise gravity or damping.
    """
    model = read_model(path, expected_sha256)
    unit = model['directions'][direction.upper()]
    for name, args in unit['commands']:
        getattr(ops, name)(*args)
    return unit['model_info'], unit['audit']
