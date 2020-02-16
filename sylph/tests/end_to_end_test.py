from sylph.tests.pipelines import vggish_svm_learner_pipeline
from sylph.tests.datasets import TwoPureTonesDataSet


def test_two_pure_tones():
    pipeline = vggish_svm_learner_pipeline
    dataset = TwoPureTonesDataSet.generate(n=20, duration=10, training_proportion=0.8)
    output = pipeline.run(dataset)
    metrics = pipeline.get_metrics(dataset, output)
    assert metrics["transformed_testing_accuracy"] == 1.0
    assert metrics["testing_accuracy"] == 1.0
