# This is just a data viewing to see there are how many templates, training data and so on.
import argparse
import sys
if __name__ == '__main__':
    hdfs_train = []
    hdfs_test_normal = []
    hdfs_test_abnormal = []
    h1 = set()
    h2 = set()
    h3 = set()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='OpenStack', type=str)
    try:
        args = parser.parse_args()
        dataset_folder = args.dataset
        dataset_folders = [
            'OpenStack',
            'HDFS'
        ]
        if dataset_folder not in dataset_folders:
            raise ValueError("Inserted the wrong dataset.\nDatasets that can be analyzed are: {} ".format(dataset_folders))
    except ValueError as e:
        print(e)
        sys.exit(2)

    dataset_folder = dataset_folder+'/'
    dataset_train = 'train'
    dataset_test_normal = 'test_normal'
    dataset_test_abnormal = 'test_abnormal'
    with open('data_parsed/'+dataset_folder+dataset_train, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_train.append(line)
    for line in hdfs_train:
        for c in line:
            h1.add(c)

    with open('data_parsed/'+dataset_folder+dataset_test_normal, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_normal.append(line)
    for line in hdfs_test_normal:
        for c in line:
            h2.add(c)

    with open('data_parsed/'+dataset_folder+dataset_test_abnormal, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_abnormal.append(line)
    for line in hdfs_test_abnormal:
        for c in line:
            h3.add(c)
    print('train length: %d, template length: %d, template: %s' % (len(hdfs_train), len(h1), h1))
    print('test_normal length: %d, template length: %d, template: %s' % (len(hdfs_test_normal), len(h2), h2))
    print('test_abnormal length: %d, template length: %d, template: %s' % (len(hdfs_test_abnormal), len(h3), h3))
