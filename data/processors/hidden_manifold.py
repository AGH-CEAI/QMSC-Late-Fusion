import numpy.typing as npt


def split_to_multisource(
    x: npt.NDArray, n_sources: int = 2, n_feat_per_source: int = 4
) -> tuple[npt.NDArray, ...]:
    return tuple(
        x[:, i * n_feat_per_source : (i + 1) * n_feat_per_source]
        for i in range(n_sources)
    )
