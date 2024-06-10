### Property Estimation Baselines
We include a variety of supervised benchmarks to perform physical property estimation from either images, spectra, or photometry. Baseline training can be run with
```python
python trainer.py [modality] [model name] [properties]
```
This automatically trains and evaluates the model on the held-out test set, reporting R-squared metrics on the properties of interest.
