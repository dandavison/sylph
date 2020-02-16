import os
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from sylph.pipeline import Transform
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_input
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_params
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_slim


tf.disable_v2_behavior()


class Audio2Spectrogram(Transform):
    def transform_observations(self, audios: np.ndarray):
        """
        Each audio file is broken into non-overlapping time windows, and a spectrogram is computed
        for each window. All time windows, across all audio files, have the same duration.
        """
        # The audioset VGGish code refers to a spectrogram from a window as an "example".
        window_spectrograms = [
            vggish_input.waveform_to_examples(audio.time_series, audio.sampling_rate)
            for audio in audios
        ]
        n_windows_per_audio = [s.shape[0] for s in window_spectrograms]
        window_spectrograms = np.concatenate(window_spectrograms, axis=0)
        index_map = np.repeat(range(len(audios)), n_windows_per_audio)
        return window_spectrograms, index_map


class Spectrogram2VGGishEmbeddings(Transform):
    def transform_observations(self, spectrograms: np.ndarray):
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
            [embeddings] = sess.run([embedding_tensor], feed_dict={features_tensor: spectrograms})
        return embeddings, None
