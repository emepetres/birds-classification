# birds-classification

Train model for birds classification and gradio app

Training is done using fastai, deployment mimics its transforms to publish a gradio app that has no fastai dependencies.

## Train

```bash
conda env create -f environment.yml
```

```bash
conda activate fastai
cd training
python -m birds.train
```
