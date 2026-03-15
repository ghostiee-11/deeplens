from deeplens.embeddings.compute import EmbeddingComputer
from deeplens.embeddings.reduce import DimensionalityReducer

try:
    from deeplens.embeddings.explorer import EmbeddingExplorer
except ImportError:
    EmbeddingExplorer = None  # datashader not installed
