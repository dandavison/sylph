<p align="center">
    <img width=200px src="https://user-images.githubusercontent.com/52205/74552822-b0f85f00-4f1b-11ea-908d-48d4f301b6a3.png" alt="Violet-tailed Sylph" />
</p>

Sylph is a library supplying data pipeline components for audio machine learning. For examples of usage, see [elaenia](https://github.com/dandavison/elaenia).

The following Sylph code defines a pipeline which performs preliminary transformations of the raw audio data, computes the spectrogram, computes [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) embeddings from the spectrogram, trains a classifier, and computes metrics describing the classification success. The API follows that of [`torchvision`](https://github.com/pytorch/vision) and [`sklearn`](https://github.com/scikit-learn/scikit-learn).

```python
from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.audio import Audio2Audio16Bit
from sylph.transforms.pca import PCA
from sylph.transforms.vggish import Audio2Spectrogram
from sylph.transforms.vggish import Spectrogram2VGGishEmbeddings


pipeline = TrainingPipeline(
    transform=Compose(
        [
            Audio2Audio16Bit(normalize_amplitude=True),
            Audio2Spectrogram(),
            Spectrogram2VGGishEmbeddings(),
            PCA(whiten=True),
        ]
    ),
    learn=SVMLearner(),
)
output = pipeline.run(dataset)
metrics = pipeline.get_metrics(dataset, output)
```

<sub>Violet-tailed Sylph (_Aglaiocercus coelestis_) by [ASAV Photography](https://www.flickr.com/photos/asavphotography/40102720883).</sub>
