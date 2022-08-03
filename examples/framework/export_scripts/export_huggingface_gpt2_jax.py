from pathlib import Path

import tensorflow
from transformers import FlaxGPT2Model, GPT2Tokenizer

import model_navigator as nav

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = FlaxGPT2Model.from_pretrained("gpt2")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="np")
dataloader = [encoded_input]

desc = nav.jax.export(
    model=model.__call__,
    model_params=model._params,
    dataloader=dataloader,
    override_workdir=True,
    target_formats=(nav.Format.TF_SAVEDMODEL,),
)

desc.save(Path.cwd() / "gpt2.nav")
