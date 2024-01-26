try:
    from opex.normalization import FusedLayerNorm
    print("apex found")
except ImportError as e:
    try:
        from xformers.triton import FusedLayerNorm
        print("xformer found")

    except ImportError as e:
        FusedLayerNorm = None
        print("not found")