import logging
import sys
import numpy as np
from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline import pipelines
from ml_pipeline.cnn import CNN, evaluate
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def plot(x, y, title, yline=None):
    plt.plot(x, y)
    plt.title(f'Impact of max_ngram size on {title}')
    plt.xlabel('Max ngram size')
    plt.ylabel(title)
    if yline:
        plt.plot(np.array([min(x), max(x)]), np.array([yline, yline]), color='gold', label='Unaugmented data')
        plt.legend()
    plt.savefig(f'./results2/figs/{title}.png')
    plt.clf()

def prunner(train_X, train_y, test_X, test_y, max_ngram=None):
    pipe = pipelines.svm_libsvc_counts((1, max_ngram))
    logger.info('>> training pipeline svm_libsvc_counts')
    pipe.fit(train_X, train_y)

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    results = utils.eval(test_y, sys_y, output_dict=True)
    # logger.info(utils.eval(test_y, sys_y))
    test_X_ref = test_X
    logger.info('>> predictions')
    utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)

    df=open(f'Balanc {max_ngram}.csv','w', encoding='utf-8')
    for i in range(0, len(sys_y)):
        df.write("{}\t{}\t{}".format(sys_y[i], test_y.values[i], test_X[i]))
        df.write('\n')
    df.close()

    return (results['accuracy'], 
        results['macro avg']['precision'], 
        results['macro avg']['recall'], 
        results['macro avg']['f1-score'])


def my_run2(task_name, data_dir, print_predictions):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    max_ngrams = np.arange(1, 5)
    pargs = [(train_X, train_y, test_X, test_y, max_ngram) for max_ngram in max_ngrams]
    with Pool(cpu_count()) as p:
        expresults = p.starmap(prunner, pargs)
    accuracies, precisions, recalls, f1s = zip(*expresults)


    plot(max_ngrams, accuracies, 'accuracy')
    plot(max_ngrams, precisions, 'Macro Avg precision')
    plot(max_ngrams, recalls, 'Macro Avg recall')
    plot(max_ngrams, f1s, 'Macro Avg F1')


def run(task_name, data_dir, pipeline_name, print_predictions):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
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




