import sklearn.svm

from sylph.learners.sklearn import SklearnClassifierLearner


class SVMLearner(SklearnClassifierLearner):
    model = sklearn.svm.SVC()
