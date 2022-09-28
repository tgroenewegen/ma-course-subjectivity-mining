import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline import pipelines
from ml_pipeline.cnn import CNN, evaluate
from json import dumps
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def plot(x, y, title, yline=None):
    plt.plot(x, y, label='Augmented data')
    plt.title(f'Impact of percentage added data on {title}')
    plt.xlabel('Percentage hate of added data')
    plt.ylabel(title)
    if yline:
        plt.plot(np.array([min(x), max(x)]), np.array([yline, yline]), color='gold', label='Unaugmented data')
        plt.legend()
    plt.savefig(f'./results/figs/{title}.png')
    plt.clf()


def get_sample(xs: np.array, ys: np.array, percentage: float, n: int):
    """
    Here we wish a sample from the data of size n without replacement with the given
    percentage of hate and 1-percentage of nothate. 

    Returns subset as tuple of (Xs, ys)
    """
    ones = ys[ys == 'hate']
    zeros = ys[ys == 'nothate']
    ones_i_sample = ones.sample(int(n*percentage), replace=False).index
    zeros_i_sample = zeros.sample(int(n*(1-percentage)), replace=False).index

    x_ones = xs[ones_i_sample]
    x_zeros = xs[zeros_i_sample]

    y_ones = ys[ones_i_sample]
    y_zeros = ys[zeros_i_sample]

    retx = pd.concat([x_ones, x_zeros])
    rety = pd.concat([y_ones, y_zeros])

    return retx, rety

def p_runner(pipeline_name, init_train_X, init_train_y, etrain_X, etrain_y, test_X, test_y, percentage):
    """
    Function to run the underlying experiment runs in parallel
    """
    num_to_add = max(len(init_train_y), int(percentage*len(init_train_y))) # add constant 50% of original data size
    p = min(percentage, 1)
    print(f'>> Running training with percentage: {p}, n: {num_to_add}')
    etrain_X_to_add, etrain_y_to_add = get_sample(etrain_X, etrain_y, p, num_to_add)
    train_y = pd.concat([init_train_y, etrain_y_to_add])
    train_X = pd.concat([init_train_X, etrain_X_to_add])
    results = fit_and_eval(pipeline_name, train_X, train_y, test_X, test_y)
    with open(f'./results/tables/res_{percentage}.json', 'w') as f:
        f.write(dumps(results))
    
    return (results['accuracy'], 
        results['macro avg']['precision'], 
        results['macro avg']['recall'], 
        results['macro avg']['f1-score'])

def my_run(task_name, data_dir, pipeline_name, print_predictions):
    """
    Here we are going to add extra data to the dataset in proportions of 0, .25, .5, .75, and 1
    in hate v. no hate to see if different proportions yield better or worse results.
    """
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)

    percentages = np.arange(0, 1.75, .25)
    extra_vua = vf.VuaFormat()
    extra_vua.load(f'/Users/ryansaeta/Desktop/Vrije Universiteit/Y2/S1P1/Subjectivity Mining/ma-course-subjectivity-mining/pynlp/data/extra/')
    etrain_X, etrain_y = extra_vua.train_instances()

    print('>> Running initial baseline with no added data')
    logger.info('>> retrieving train/data instances...')
    init_train_X, init_train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    if 'trac2018' in data_dir:
        # remap the y's to just 2 levels so that extra data can be incorporated
        init_train_y = init_train_y.map({'OAG': 'hate', 'CAG': 'hate', 'NAG': 'nothate'})
        test_y = test_y.map({'OAG': 'hate', 'CAG': 'hate', 'NAG': 'nothate'})
    init_results = fit_and_eval(pipeline_name, init_train_X, init_train_y, test_X, test_y)
    pargs = [(pipeline_name, init_train_X, init_train_y, etrain_X, etrain_y, test_X, test_y, p) for p in percentages]
    with Pool(cpu_count()) as p:
        expresults = p.starmap(p_runner, pargs)
    breakpoint()
    accuracies, precisions, recalls, f1s = zip(*expresults)
    plot(percentages, accuracies, 'accuracy', yline=init_results['accuracy'])
    plot(percentages, precisions, 'Macro Avg precision', yline=init_results['macro avg']['precision'])
    plot(percentages, recalls, 'Macro Avg recall', yline=init_results['macro avg']['recall'])
    plot(percentages, f1s, 'Macro Avg F1', yline=init_results['macro avg']['f1-score'])


def run(task_name, data_dir, pipeline_name, print_predictions, add_extra_training=False):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    if 'trac2018' in data_dir:
        # remap the y's to just 2 levels so that extra data can be incorporated
        train_y = train_y.map({'OAG': 'hate', 'CAG': 'hate', 'NAG': 'nothate'})
        test_y = test_y.map({'OAG': 'hate', 'CAG': 'hate', 'NAG': 'nothate'})
    if add_extra_training:
        extra_vua = vf.VuaFormat()
        extra_vua.load(f'/Users/ryansaeta/Desktop/Vrije Universiteit/Y2/S1P1/Subjectivity Mining/ma-course-subjectivity-mining/pynlp/data/extra/')
        etrain_X, etrain_y = extra_vua.train_instances()

        train_y = pd.concat([train_y, etrain_y])
        train_X = pd.concat([train_X, etrain_X])
    test_X_ref = test_X

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)
  
    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info("   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))

    if print_predictions:
        logger.info('>> predictions')
        utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)
    return utils.eval(test_y, sys_y, output_dict=True)


def fit_and_eval(pipeline_name, train_X, train_y, test_X, test_y):
    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)
    logger.info(f'>> training pipeline {pipeline_name}')
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info("   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation')
    results = utils.eval(test_y, sys_y, output_dict=True)
    logger.info(utils.eval(test_y, sys_y))

    return results

def task(name):
    if name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    else:
        raise ValueError("pipeline name is unknown.")


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
    elif name == 'naive_bayes_counts_lex':
        return pipeline_with_lexicon.naive_bayes_counts_lex()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")


if __name__ == '__main__':
    my_run('vua_format', './data/trac2018_VUA/', 'svm_libsvc_counts', False)

