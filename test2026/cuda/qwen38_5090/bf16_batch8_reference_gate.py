"""Select actual B8 rows for the existing strict independent B4 reference gate.

No geometry/parser/kernel is widened. Selecting four complete row witnesses is
not an independent model-math result; actual fresh B4 reference bytes are needed.
"""
import math
import struct
from .bf16_batch8_admission import one_row
from .bf16_same_precision_gate import same_logical_state


def require_rows(rows):
    if (not isinstance(rows, (list, tuple)) or len(rows) != 4
            or any(type(v) is not int or not 0 <= v < 8 for v in rows) or len(set(rows)) != 4):
        raise ValueError('Four distinct exact B8 physical row indices required')


def four_complete_rows(witness, rows):
    require_rows(rows)
    # Existing B8 validator checks every row/component, including unselected
    # rows, before returning any selected witness. Per-row widths stay actual.
    one_row(witness, 0)
    result = dict(witness)
    for name in ('slots', 'physical_row_sha256', 'logical_row_sha256'):
        result[name] = [witness[name][row] for row in rows]
    result['components'] = [dict(component, row_sha256=[component['row_sha256'][row] for row in rows])
                            for component in witness['components']]
    return result


def same_reference_state(b8, b4_reference, rows):
    # Strict existing B4 gate still requires full FP32-cache representation,
    # BF16 roundtrip attestation, all128logical components/slots and all hashes.
    return same_logical_state(four_complete_rows(b8, rows), b4_reference)


def same_reference_vectors(b8, b4_reference, rows):
    require_rows(rows)
    if not isinstance(b8, (bytes, bytearray)) or not isinstance(b4_reference, (bytes, bytearray)):
        raise ValueError('Actual raw FP32 vector bytes required')
    if len(b8) != 8*5120*4 or len(b4_reference) != 4*5120*4:
        raise ValueError('Complete B8 and B4 hidden/normalized rows required')
    if any(not math.isfinite(value[0]) for raw in (b8, b4_reference) for value in struct.iter_unpack('<f', raw)):
        raise ValueError('Finite complete hidden/normalized vectors required')
    width = 5120*4
    return b''.join(b8[row*width:(row+1)*width] for row in rows) == b4_reference
