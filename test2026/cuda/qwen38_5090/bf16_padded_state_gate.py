"""Compare explicitly selected rows; private padding is never a real request."""
import copy
if __package__:
    from .bf16_same_precision_gate import same_logical_state
else:
    from bf16_same_precision_gate import same_logical_state


def same_logical_row(packed, row, reference, reference_row):
    if any(type(n) is not int or not 0 <= n < 4 for n in (row, reference_row)):
        raise ValueError('State row must be an integer in the physical B4 profile')
    # Validate the complete original witnesses before selecting a row. An
    # unrelated row may differ, but malformed metadata must never be hidden.
    same_logical_state(packed, reference)

    def repeated(value, index):
        result = copy.deepcopy(value)
        for key in ('slots', 'physical_row_sha256', 'logical_row_sha256'):
            result[key] = [value[key][index]] * 4
        for component in result['components']:
            component['row_sha256'] = [component['row_sha256'][index]] * 4
        return result

    return same_logical_state(repeated(packed, row), repeated(reference, reference_row))
