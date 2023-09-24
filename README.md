# birds-classification

Model training for birds classification and deployment on hugging face gradio app.

Training is done using fastai, deployment mimics its transforms to publish a gradio app that has no fastai dependencies.

It automatically publish to Hugging Face after a pull request is succesfully merged in main branch.

<https://huggingface.co/spaces/jcarnero/birds-classification>

## Train

```bash
conda env create -f environment.yml
```

```bash
conda activate fastai
cd training
python -m birds.train
```

## Run

```bash
conda activate fastai
cd deployment
python app.py
```

And then go to the local URL that appears in the terminal.
