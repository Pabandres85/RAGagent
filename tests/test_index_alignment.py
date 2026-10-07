"""
Pruebas de correspondencia entre indices FAISS y metadatos.

El retriever toma la posicion devuelta por FAISS y la usa como indice en la lista
de metadatos. Si ambos estan en ordenes distintos, el texto y la cita no
corresponden al vector recuperado. Estas pruebas verifican el orden, no solo
que los totales coincidan.
"""
import faiss
import numpy as np
import pytest

from core.config import settings
from core.embeddings import embed_texts
from core.metadata_store import MODULES, MetadataStore

INDEX_DIR = settings.faiss_index_dir
SAMPLE_PER_MODULE = 5


def _require_index(name: str) -> faiss.Index:
    path = INDEX_DIR / f"{name}.faiss"
    if not path.exists():
        pytest.skip(f"Falta {path}; ejecuta scripts/ingest.py")
    return faiss.read_index(str(path))


@pytest.fixture(scope="module")
def store() -> MetadataStore:
    return MetadataStore()


@pytest.mark.parametrize("module", MODULES)
def test_module_index_size_matches_metadata(module, store):
    index = _require_index(module)
    assert index.ntotal == len(store.load(module))


def test_global_index_size_matches_load_all(store):
    index = _require_index("global")
    assert index.ntotal == len(store.load_all())


def test_global_vectors_follow_module_order():
    """Cada bloque del indice global es exactamente el indice de su modulo, en orden."""
    global_index = _require_index("global")
    offset = 0
    for module in MODULES:
        module_index = _require_index(module)
        n = module_index.ntotal
        block = global_index.reconstruct_n(offset, n)
        expected = module_index.reconstruct_n(0, n)
        assert np.allclose(block, expected, atol=1e-6), (
            f"El bloque global de '{module}' (posiciones {offset}-{offset + n - 1}) "
            "no coincide con su indice por modulo."
        )
        offset += n


def test_global_position_matches_metadata_text(store):
    """El vector global en la posicion i es el embedding del texto de load_all()[i]."""
    global_index = _require_index("global")
    chunks = store.load_all()

    positions: list[int] = []
    offset = 0
    for module in MODULES:
        n = len(store.load(module))
        step = max(n // SAMPLE_PER_MODULE, 1)
        positions.extend(range(offset, offset + n, step)[:SAMPLE_PER_MODULE])
        offset += n

    try:
        expected = embed_texts([chunks[i].text for i in positions])
    except (OSError, ConnectionError) as exc:  # modelo no cacheado y sin red
        pytest.skip(f"Modelo de embeddings no disponible: {exc}")
    for k, pos in enumerate(positions):
        vector = global_index.reconstruct(pos)
        assert float(np.dot(vector, expected[k])) > 0.999, (
            f"La posicion global {pos} no corresponde al texto de su metadato "
            f"(modulo '{chunks[pos].module}')."
        )
