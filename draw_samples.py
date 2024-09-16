from matplotlib import pyplot as plt
import matplotlib.patches as patch
import image_func as im

def draw_samples(dataset_path: str,
                 n_rows: int = 3,
                 n_cols: int = 4,
                 bbox_show: bool = True,
                 img_size_pixels: tuple[int, int] = (640, 640)) -> None:
    """
    Draw samples from dataset
    Number of images is n_rows * n_cols

    Parameters
    ----------
    dataset_path str:
        Path to dataset folder

    n_rows int:
        Number rows of images on plot

    n_cols int:
        Number columns of images on plot

    bbox_show bool:
        Show or don't show bounding boxes

    img_size_pixels tuple[int, int]:
        Dimensions of all images in pixels

    Returns
    -------
    None
    """

    # Create a figure
    fig, tmp_axs = plt.subplots(figsize=(12,6), nrows=n_rows, ncols=n_cols)

    # Flatten axs list
    axs = [i for j in tmp_axs for i in j]

    del tmp_axs

    # Iterate through axs
    for ax in axs:
        # Get all data related with image
        img_name, x_centre, y_centre, width, height, image = next(im.random(dataset_path, n_imgs= n_cols * n_rows, show_labels=True))
        ax.imshow(image)

        # Optionally show bboxes
        if bbox_show:
            x_centre = int(x_centre * img_size_pixels[0])
            y_centre = int(y_centre * img_size_pixels[1])
            width = int(width * img_size_pixels[0])
            height = int(height * img_size_pixels[1])
            bbox = patch.Rectangle(xy=(x_centre - width // 2, y_centre - height // 2),
                                   width=width,
                                   height=height,
                                   edgecolor='g',
                                   linewidth=1,
                                   facecolor='none')
            ax.add_patch(bbox)
    plt.show()

draw_samples('datasets\\dataset\\test')


    