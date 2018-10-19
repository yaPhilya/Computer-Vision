from os import environ
from os.path import join
from sys import argv


def check_test(output_dir, gt_dir):
    with open(join(output_dir, 'output.csv')) as fout:
        lines = fout.readlines()
        output = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            output[filename] = class_id

    with open(join(gt_dir, 'gt.csv')) as fgt:
        next(fgt)
        lines = fgt.readlines()
        gt = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            gt[filename] = class_id

    correct = 0
    total = len(gt)
    for k, v in gt.items():
        if output[k] == v:
            correct += 1

    accuracy = correct / total

    return 'Ok, accuracy %.4f' % accuracy


def grade(results_list):
    test_data_result = results_list[-1]

    result = test_data_result['result']
    if not result.startswith('Ok'):
        return '', 0

    accuracy_str = result[13:]
    accuracy = float(accuracy_str)

    if accuracy >= 0.93:
        mark = 10
    elif accuracy >= 0.90:
        mark = 8
    elif accuracy >= 0.85:
        mark = 6
    elif accuracy >= 0.80:
        mark = 4
    elif accuracy >= 0.75:
        mark = 2
    elif accuracy > 0:
        mark = 1
    else:
        mark = 0

    return accuracy_str, mark


def run_single_test(data_dir, output_dir):
    from fit_and_classify import fit_and_classify, extract_hog
    from glob import glob
    from numpy import zeros
    from os.path import basename, join
    from skimage.io import imread

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    def read_gt(gt_dir):
        fgt = open(join(gt_dir, 'gt.csv'))
        next(fgt)
        lines = fgt.readlines()

        filenames = []
        labels = zeros(len(lines))
        for i, line in enumerate(lines):
            filename, label = line.rstrip('\n').split(',')
            filenames.append(filename)
            labels[i] = int(label)

        return filenames, labels

    def extract_features(path, filenames):
        hog_length = len(extract_hog(imread(join(path, filenames[0]))))
        data = zeros((len(filenames), hog_length))
        for i in range(0, len(filenames)):
            filename = join(path, filenames[i])
            data[i, :] = extract_hog(imread(filename))
        return data

    train_filenames, train_labels = read_gt(train_dir)
    test_filenames = []
    for path in sorted(glob(join(test_dir, '*png'))):
        test_filenames.append(basename(path))

    train_features = extract_features(train_dir, train_filenames)
    test_features = extract_features(test_dir, test_filenames)

    y = fit_and_classify(train_features, train_labels, test_features)

    with open(join(output_dir, 'output.csv'), 'w') as fout:
        for i, filename in enumerate(test_filenames):
            print('%s,%d' % (filename, y[i]), file=fout)


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
