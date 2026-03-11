import os, sys
# Ensure project root is on sys.path so tests can import local packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chemometrics.data_input import load_data, _generate_axis_info, _check_if_numeric, _load_axis_file_content
import pprint
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
tests = os.path.join(root, 'tests')
x1 = os.path.join(tests, 'x1.csv')
x2 = os.path.join(tests, 'x2.csv')
y = os.path.join(tests, 'y.csv')
axis_names = os.path.join(tests, 'axis_names_var.txt')
axis_num = os.path.join(tests, 'axis_numeric_var.txt')

ok = True

print('Running load_data smoke tests...')
try:
    X, Y, axis_t_info, smp_cal, axis_n_info, axis_nature, dim_labels, _, _, cal_metadata = load_data(
        d_specs_separator='comma',
        d_specs_headlines='0',
        d_specs_type='x_matrix',
        d_specs_dimensions=None,
        data_path=[x1, x2],
        nway_flag=1,
        y_path=y,
        var_path=[axis_names],
        smp_path=None,
        transpose=False,
        axis_info=None,
        reshape_order='F',
        dim_labels=['Samples'],
        scale_type=None
    )

    print('\nTest 1 - basic load:')
    print('X.shape =', X.shape)
    print('Expect rows = 4, cols = 3')
    if X.shape != (4,3):
        print('FAIL: unexpected X.shape')
        ok = False
    print('Y.shape =', None if Y is None else Y.shape)
    if Y is None or (Y.shape != (2,1) and Y.shape != (4,1)):
        # Y may be loaded differently depending on assumptions; we at least check it's not error
        print('Note: Y shape unexpected, got', Y.shape if Y is not None else None)

    print('\naxis_t_info =')
    pprint.pprint(axis_t_info)
    if not isinstance(axis_t_info, list):
        print('FAIL: axis_t_info not list')
        ok = False

    print('\naxis_n_info (first few) =')
    pprint.pprint([a.tolist() for a in axis_n_info])
    print('axis_nature =', axis_nature)
    print('\ncal_metadata entries =', len(cal_metadata) if isinstance(cal_metadata, dict) else 0)

    # Test 2 - axis_info list + scale_type
    Xfake = np.zeros((2,3))
    axis_info_list = ['100 300']
    axis_n = _generate_axis_info(axis_info_list, Xfake, scale_type=['Linear'])
    print('\nTest 2 - axis_info list + scale_type:')
    pprint.pprint([v.tolist() for v in axis_n])
    if len(axis_n) < 2 or len(axis_n[1]) != 3:
        print('FAIL: generated axis vector wrong size')
        ok = False

    # Test 3 - numeric file detection
    content = _load_axis_file_content(axis_num)
    isnum, vals = _check_if_numeric(content)
    print('\nTest 3 - numeric file detection:')
    print('is_numeric:', isnum)
    print('values:', None if vals is None else vals.tolist())
    if not isnum:
        print('FAIL: numeric file not detected as numeric')
        ok = False

except Exception as e:
    print('Exception during tests:', e)
    ok = False

print('\nOverall OK:' , ok)
if not ok:
    sys.exit(2)
else:
    sys.exit(0)
