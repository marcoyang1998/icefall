import os

import numpy as np
from lhotse.features.io import NumpyHdf5Writer

def create_dummy_embeddings():
    dummy_embedding_folder = "data/dummy_embeddings"

    # create dummy whisper embeddings
    T = 1510
    whisper_dim = 1280
    dummy_whisper_embedding = np.random.rand(T, whisper_dim) # (T, whisper_dim)
    dummy_whisper_path = f"{dummy_embedding_folder}/dummy_whisper_embedding_{T}.h5"
    if not os.path.exists(dummy_whisper_path):
        with NumpyHdf5Writer(dummy_whisper_path) as writer:
            dummy_whisper_embedding = writer.store_array(
                key=f"dummy_whisper_embedding_{T}",
                value=dummy_whisper_embedding,
                temporal_dim=0,
                frame_shift=0.02,
                start=0,
            )
        print(f"Saved to {dummy_whisper_path}")
            
    T = 1510
    num_cb = 16
    dummy_whisper_cb_index = np.random.randint(0, num_cb, size=(T, num_cb)) # (T, whisper_dim)
    dummy_whisper_cb_path = f"{dummy_embedding_folder}/dummy_whisper_codebook_indexes_{T}.h5"
    import pdb; pdb.set_trace()
    if not os.path.exists(dummy_whisper_cb_path):
        with NumpyHdf5Writer(dummy_whisper_cb_path) as writer:
            dummy_whisper_cb_index = writer.store_array(
                key=f"dummy_whisper_codebook_indexes_{T}",
                value=dummy_whisper_cb_index,
                temporal_dim=0,
                frame_shift=0.02,
                start=0,
            )
        print(f"Saved to {dummy_whisper_cb_path}")

    # dummy ecapa-tdnn embedding
    dummy_ecapa_embedding = np.random.rand(1, 192) # (1, 192)
    dummy_ecapa_path = f"{dummy_embedding_folder}/dummy_ecapa_embedding.h5"
    if not os.path.exists(dummy_ecapa_path):
        with NumpyHdf5Writer(dummy_ecapa_path) as writer:
            dummy_ecapa_embedding = writer.store_array(
                key="dummy_ecapa_embedding",
                value=dummy_ecapa_embedding,
            )

    # dummy beats embedding
    dummy_beats_embedding = np.random.rand(527) # (527)
    dummy_beats_path = f"{dummy_embedding_folder}/dummy_beats_embedding.h5"
    if not os.path.exists(dummy_beats_path):
        with NumpyHdf5Writer(dummy_beats_path) as writer:
            dummy_beats_embedding = writer.store_array(
                key="dummy_beats_embedding",
                value=dummy_beats_embedding,
            )
        print(f"Saved to {dummy_beats_path}")

    # dummy mert embedding
    T = 2260
    mert_dim=1024
    dummy_mert_embedding = np.random.rand(T, mert_dim) # (T, mert_dim)
    dummy_mert_path = f"{dummy_embedding_folder}/dummy_mert_embedding_{T}.h5"
    if not os.path.exists(dummy_mert_path):
        with NumpyHdf5Writer(dummy_mert_path) as writer:
            dummy_mert_embedding = writer.store_array(
                key=f"dummy_mert_embedding_{T}",
                value=dummy_mert_embedding,
                temporal_dim=0,
                frame_shift=1/75,
                start=0,
            )
        print(f"Saved to {dummy_mert_path}")
    

if __name__=="__main__":
    create_dummy_embeddings()