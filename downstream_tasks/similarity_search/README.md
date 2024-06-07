## In-Modal and Cross-Modal Retrieval
AstroCLIP enables researchers to easily find similar galaxies to a query galaxy by simply exploiting the cosine similarity between galaxy embeddings in embedding space. Because AstroCLIP's embedding space is shared between both galaxy images and optical spectra, retrieval can be performed for both in-modal and cross-modal similarity searches. 

### Embedding the dataset
To perform retrieval on the held-out validation set, it is important to first generate AstroCLIP embeddings of the galaxy images and spectra. We provide the already-embedded held-out validation set here:

TODO

For reproducibility, or to embed another dataset, we also include the scripts that can be used to perform this embedding, which can be executed with
```python
python embed_astroclip.py [save_path]
```

### Similarity Search
Once embedded, the ```similarity_search.ipynb``` jupyter notebook contains a brief tutorial that demonstrates the retrieval abilities of the model.