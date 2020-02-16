import os
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from sylph.dataset import Dataset
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_input
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_params
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_slim


tf.disable_v2_behavior()


class Audio2Spectrogram:
    def __call__(self, dataset: Dataset) -> Dataset:
        spectrograms = [
            vggish_input.waveform_to_examples(audio.time_series, audio.sampling_rate)
            for audio in dataset.observations
        ]
        n_examples_per_audio = [s.shape[0] for s in spectrograms]
        observations = np.concatenate(spectrograms, axis=0)
        labels = np.repeat(dataset.labels, n_examples_per_audio)
        dataset = Dataset(observations=observations, labels=labels)
        dataset.n_examples_per_audio = n_examples_per_audio  # Hack
        return dataset


class Spectrogram2VGGishEmbeddings:
    def __call__(self, dataset: Dataset) -> Dataset:
        vggish_checkpoint_path = os.getenv("SYLPH_VGGISH_MODEL_CHECKPOINT_FILE")
        if not vggish_checkpoint_path:
            raise AssertionError(
                "Environment variable SYLPH_VGGISH_MODEL_CHECKPOINT_FILE is not set."
            )
        if not Path(vggish_checkpoint_path).exists():
            raise AssertionError(f"File does not exist: {vggish_checkpoint_path}")

        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)
            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run(
                [embedding_tensor], feed_dict={features_tensor: dataset.observations}
            )
        # Hack
        n_examples_per_audio = dataset.n_examples_per_audio
        dataset = Dataset(observations=embedding_batch, labels=dataset.labels)
        dataset.n_examples_per_audio = n_examples_per_audio
        return dataset
