import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def split_into_subtiles(tile, grid_size=3):
    """
    The tile is cut in subtiles of size grid_size x grid_size 
    """
    H, W, C = tile.shape
    assert H % grid_size == 0 and W % grid_size == 0
    h_step, w_step = H // grid_size, W // grid_size

    subtiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step
            sub_tile = tile[y1:y2, x1:x2, :]
            subtiles.append(sub_tile)
    return subtiles  # This will return 9 subtiles (H/3, W/3, C)



def extract_tile_fixed(image, x, y, tile_size=336, pad_mode='edge'):
    """
    Extract the tile centred at (x, y), If it exceeds the boundary, padding it and return the position of the padding.
    
    Args:
        image: Original Image (H, W, C)
        x, y: Central coordinates
        tile_size: The size of the captured tile 
        pad_mode: The mode used by np.pad (default is 'edge')

    Returns:
        tile: The captured tile (tile_size, tile_size, C)
        padded_coords: If padding exists,it is [x, y];Otherwise it is None
    """
    H, W, C = image.shape
    half = tile_size // 2

    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half

    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)

    padded = any([pad_left, pad_right, pad_top, pad_bottom])
    if padded:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=pad_mode
        )
        # Adjust coordinates
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    tile = image[y1:y2, x1:x2, :]
    return (tile, [x, y]) if padded else (tile, None)



def get_spots_in_tile(df, center_x, center_y, tile_size):
    half = tile_size // 2
    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    df_in_tile = df[
        (df['x'] >= x_min) & (df['x'] <= x_max) &
        (df['y'] >= y_min) & (df['y'] <= y_max)
    ].copy()

    df_in_tile = df_in_tile[
        ~((df_in_tile['x'] == center_x) & (df_in_tile['y'] == center_y))
    ]

    return df_in_tile




def plot_tile_with_spots(
    slide_image, spot_df, center_x, center_y,
    tile_size=78, stride=None, grid_size=None,
    spot_radius_px=15
):
    """
    Visualize a tile including the centre spot neighbouring spots and optional sub grid lines

    Parameters:
    - slide_image: The image of this slice (numpy array, H x W x 3)
    - spot_df: Includes all spots that fall within the tile (excluding the center spot)
    - center_x, center_y: slide coordinates of the center spot
    - tile_size: Size of a single tile (in pixels)
    - stride: Tile_spacing (default equal to tile_size,indicating no weight)
    - grid_size: The size of the tile（If None,only a single tile will be drawn）
    - spot_radius_px: The radius of the circle drawn from the center spot
    """

    if stride is None:
        stride = tile_size

    if grid_size:
        full_tile_size = tile_size + stride * (grid_size - 1)
    else:
        full_tile_size = tile_size

    center_color = 'red'
    neighbor_color = 'orange'
    show_legend = True
    title = f"{len(spot_df)} neighbors in tile"

    # Get tile
    def extract_tile(slide_img, x, y, total_size):
        x, y = int(x), int(y)
        half = total_size // 2
        x1, x2 = x - half, x + half
        y1, y2 = y - half, y + half
        tile = slide_img[y1:y2, x1:x2]
        return tile

    tile = extract_tile(slide_image, center_x, center_y, full_tile_size)

    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tile)

    # Grid lines
    if grid_size:
        for row in range(grid_size):
            for col in range(grid_size):
                x0 = col * stride
                y0 = row * stride
                rect = patches.Rectangle(
                    (x0, y0),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor='blue',
                    facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)


    # Center spot（fixed in the center of the tile ）
    center_px = full_tile_size // 2
    ax.scatter(center_px, center_px, c=center_color, s=40, label='Center Spot')

    # Center spot circle
    circle = patches.Circle(
        (center_px, center_px),
        spot_radius_px,
        linewidth=1.5,
        edgecolor='yellow',
        facecolor='none'
    )
    ax.add_patch(circle)

    # Neighbour spots
    for _, neighbor in spot_df.iterrows():
        dx = int(neighbor['x']) - center_x
        dy = int(neighbor['y']) - center_y
        tile_x = center_px + dx
        tile_y = center_px + dy
        ax.scatter(tile_x, tile_y, c=neighbor_color, s=20)

    ax.axis('off')
    if show_legend:
        ax.legend(loc='upper right')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()