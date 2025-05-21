def patch_within_bounds(x, y, half, image):
    return (
        x - half > 0
        and x + half < image.shape[1]
        and y - half > 0
        and y + half < image.shape[0]
    )
