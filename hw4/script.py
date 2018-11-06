from numpy import array
from os import environ
from os.path import join
from sys import argv


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

def save_csv(facepoints, filename):
    with open(filename, 'w') as fhandle:
        print('filename,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14',
                file=fhandle)
        for filename in sorted(facepoints.keys()):
            points_str = ','.join(map(str, facepoints[filename]))
            print('%s,%s' % (filename, points_str), file=fhandle)


def check_test(output_dir, gt_dir):

    def read_img_shapes(gt_dir):
        img_shapes = {}
        with open(join(gt_dir, 'img_shapes.csv')) as fhandle:
            next(fhandle)
            for line in fhandle:
                parts = line.rstrip('\n').split(',')
                filename = parts[0]
                n_rows, n_cols = map(int, parts[1:])
                img_shapes[filename] = (n_rows, n_cols)
        return img_shapes

    def compute_metric(detected, gt, img_shapes):
        res = 0.0
        for filename, coords in detected.items():
            n_rows, n_cols = img_shapes[filename]
            diff = (coords - gt[filename])
            diff[::2] /= n_cols
            diff[1::2] /= n_rows
            diff *= 100
            res += (diff ** 2).mean()
        return res / len(detected.keys())

    detected = read_csv(join(output_dir, 'output.csv'))
    gt = read_csv(join(gt_dir, 'gt.csv'))
    img_shapes = read_img_shapes(gt_dir)
    error = compute_metric(detected, gt, img_shapes)

    return 'Ok, error %.4f' % error


def grade(results_list):
    test_data_result = results_list[-1]

    result = test_data_result['result']
    if not result.startswith('Ok'):
        return '', 0

    error_str = result[10:]
    error = float(error_str)

    if error <= 9:
        mark = 10
    elif error <= 11:
        mark = 9
    elif error <= 13:
        mark = 8
    elif error <= 15:
        mark = 7
    elif error <= 17:
        mark = 6
    elif error <= 20:
        mark = 5
    elif error <= 50:
        mark = 2
    else:
        mark = 0

    return error_str, mark


def run_single_test(data_dir, output_dir):
    from detection import train_detector, detect
    from keras import backend as K
    from keras.models import load_model
    from os import environ
    from os.path import abspath, dirname, join

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    train_detector(train_gt, train_img_dir, fast_train=True)

    code_dir = dirname(abspath(__file__))
    model = load_model(join(code_dir, 'facepoints_model.hdf5'))
    test_img_dir = join(test_dir, 'images')
    detected_points = detect(model, test_img_dir)
    save_csv(detected_points, join(output_dir, 'output.csv'))

    if environ.get('KERAS_BACKEND') == 'tensorflow':
        K.clear_session()


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system, run on single input
        if len(argv) != 3:
            print('Usage: %s data_dir output_dir' % argv[0])
            exit(0)

        run_single_test(argv[1], argv[2])
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_input'))):
            output_dir = sub('input$', 'output', input_dir)
            makedirs(output_dir, exist_ok=True)
            gt_dir = sub('input$', 'gt', input_dir)

            try:
                start = time()
                run_single_test(input_dir, output_dir)
                end = time()
                running_time = end - start
            except:
                result = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    result = check_test(output_dir, gt_dir)
                except:
                    result = 'Checker error'
                    traceback = format_exc()

            test_num = input_dir[-8:-6]
            if result == 'Runtime error' or result == 'Checker error':
                print(test_num, result, '\n', traceback)
                results.append({'result': result})
            else:
                print(test_num, '%.2fs' % running_time, result)
                results.append({
                    'time': running_time,
                    'result': result})

        description, mark = grade(results)
        print('Mark:', mark, description)