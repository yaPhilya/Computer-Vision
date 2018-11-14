from os import environ
from os.path import join
from sys import argv


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res


def save_csv(img_classes, filename):
    with open(filename, 'w') as fhandle:
        print('filename,class_id', file=fhandle)
        for filename in sorted(img_classes.keys()):
            print('%s,%d' % (filename, img_classes[filename]), file=fhandle)


def check_test(output_dir, gt_dir):
    output = read_csv(join(output_dir, 'output.csv'))
    gt = read_csv(join(gt_dir, 'gt.csv'))

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

    if accuracy >= 0.80:
        mark = 10
    elif accuracy >= 0.77:
        mark = 9
    elif accuracy >= 0.75:
        mark = 8
    elif accuracy >= 0.70:
        mark = 6
    elif accuracy >= 0.65:
        mark = 5
    elif accuracy >= 0.60:
        mark = 4
    elif accuracy >= 0.40:
        mark = 2
    else:
        mark = 0

    return accuracy_str, mark


def run_single_test(data_dir, output_dir):
    from classification import train_classifier, classify
    from keras import backend as K
    from keras.models import load_model
    from os import environ
    from os.path import abspath, dirname, join

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    train_classifier(train_gt, train_img_dir, fast_train=True)

    code_dir = dirname(abspath(__file__))
    model = load_model(join(code_dir, 'birds_model.hdf5'))
    test_img_dir = join(test_dir, 'images')
    img_classes = classify(model, test_img_dir)
    save_csv(img_classes, join(output_dir, 'output.csv'))

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