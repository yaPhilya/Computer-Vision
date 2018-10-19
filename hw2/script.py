from math import floor
from os import environ
from os.path import join
from pickle import load
from sys import argv


def check_test(output_dir, gt_dir):
    correct = 0
    with open(join(output_dir, 'output_seams'), 'rb') as fout, \
         open(join(gt_dir, 'seams'), 'rb') as fgt:
        for i in range(8):
            if load(fout) == load(fgt):
                correct += 1
    return 'Ok %d/8' % correct


def grade(results_list):
    ok_count = 0
    for result in results_list:
        r = result['result']
        if r.startswith('Ok'):
            ok_count += int(r[3:4])
    total_count = len(results_list) * 8
    mark = floor(ok_count / total_count / 0.1)
    description = '%02d / %02d' % (ok_count, total_count)
    return description, mark


def run_single_test(data_dir, output_dir):
    from numpy import where
    from os.path import join
    from pickle import dump
    from seam_carve import seam_carve
    from skimage.io import imread

    def get_seam_coords(seam_mask):
        coords = where(seam_mask)
        t = [i for i in zip(coords[0], coords[1])]
        t.sort(key=lambda i: i[0])
        return tuple(t)

    def convert_img_to_mask(img):
        return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype('int8')

    img = imread(join(data_dir, 'img.png'))
    mask = convert_img_to_mask(imread(join(data_dir, 'mask.png')))

    with open(join(output_dir, 'output_seams'), 'wb') as fhandle:
        for m in (None, mask):
            for direction in ('shrink', 'expand'):
                for orientation in ('horizontal', 'vertical'):
                    seam = seam_carve(img, orientation + ' ' + direction,
                                      mask=m)[2]
                    dump(get_seam_coords(seam), fhandle)


if __name__ == '__main__':
    print('bjhvjhvhjvhjvhjvhjvjhvjhvjvhgv')
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