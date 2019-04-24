import argparse
import concurrent.futures
import os
from tqdm import tqdm
import numpy as np
from PIL import Image


def generate_palette_masks(indices, arr, colors):
    output_palette = np.zeros((arr.shape[0], arr.shape[1]))
    output_mask = None
    for i in indices:
        color = colors[i]
        masks = [arr[:, :, j] == color_j for j, color_j in enumerate(color)]
        mask = np.prod(masks, axis=0)
        output_palette[mask > 0] = i
        if output_mask is None:
            output_mask = mask
        else:
            output_mask = np.add(output_mask, mask)
    return output_palette, output_mask


def space_out_sprites(image_path: str, sprites_per_row=5, buffer_pixels=10):
    with Image.open(image_path) as img:
        pixel_array = np.array(img)
    print('Counting colors:')
    pixels_flattened = np.reshape(pixel_array, (img.width * img.height, len(img.getbands())))
    unique_colors, color_counts = np.unique(pixels_flattened, axis=0, return_counts=True)
    colors = np.column_stack((color_counts, unique_colors))
    colors = colors[(-color_counts).argsort()]
    background_color = colors[0, 1:]
    print('Detecting background color:', background_color)
    pixel_array_palette = np.zeros((img.height, img.width))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        chunked_colors = np.array_split(list(range(len(colors))), 16)
        try:
            futures = [executor.submit(generate_palette_masks, color_set, np.copy(pixel_array), colors[:, 1:]) for
                       color_set
                       in chunked_colors]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(chunked_colors),
                          desc='Computing color palettes'):
                if f.exception() is not None:
                    raise f.exception()
        except Exception as ex:
            executor.shutdown()
            raise ex
    for f in tqdm(futures, desc='Merging palettes'):
        palette, mask = f.result()
        pixel_array_palette[mask != 0] = palette[mask != 0]
    image_ids = np.zeros((img.height, img.width))
    bounding_boxes = []
    bounding_box_width = 0
    bounding_box_height = 0
    for y in tqdm(range(0, img.height, buffer_pixels), desc='Identifying sprites by row'):
        for x in range(0, img.width, buffer_pixels):
            if pixel_array_palette[y, x] != 0 and image_ids[y, x] == 0:
                start_y = img.height
                start_x = img.width
                end_y = y
                end_x = x
                image_id = len(bounding_boxes) + 1
                image_ids[y, x] = image_id
                queue = [(y, x, buffer_pixels)]
                while len(queue) > 0:
                    queued_y, queued_x, buffer_left = queue.pop()
                    if 0 <= queued_y < img.height and 0 <= queued_x < img.width:
                        start_y = min(queued_y, start_y)
                        end_y = max(queued_y + 1, end_y)
                        start_x = min(queued_x, start_x)
                        end_x = max(queued_x + 1, end_x)
                        if pixel_array_palette[queued_y, queued_x] != 0 or buffer_left > 0:
                            if pixel_array_palette[queued_y, queued_x] == 0:
                                buffer_left -= 1
                            for delta_y, delta_x in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
                                new_y, new_x = delta_y + queued_y, delta_x + queued_x
                                if 0 <= new_y < img.height and 0 <= new_x < img.width:
                                    if image_ids[new_y, new_x] == 0:
                                        image_ids[new_y, new_x] = image_id
                                        queue.append((new_y, new_x, buffer_left))
                bounding_boxes.append((start_y, start_x, end_y, end_x))
                bounding_box_width = max(bounding_box_width, end_x - start_x + buffer_pixels)
                bounding_box_height = max(bounding_box_height, end_y - start_y + buffer_pixels)

    print('\nDetermined sprite image sizes: ({},{}).\n'.format(bounding_box_width, bounding_box_height))
    sprite_sheet_dimensions = (bounding_box_height * int((len(bounding_boxes) + sprites_per_row - 1) / sprites_per_row),
                               bounding_box_width * sprites_per_row, len(background_color))
    sprite_sheet = np.zeros(sprite_sheet_dimensions, dtype=pixel_array.dtype)
    sprite_sheet_y = 0
    sprite_sheet_x = 0
    for i, bounding_box in tqdm(enumerate(bounding_boxes), total=len(bounding_boxes), desc='Generating sprites'):
        start_y, start_x, end_y, end_x = bounding_box
        width, height = end_x - start_x, end_y - start_y
        image_id = i + 1
        delta_width = int((bounding_box_width - width) / 2 + 0.5)
        delta_height = int((bounding_box_height - height) / 2 + 0.5)
        img = np.zeros((bounding_box_height, bounding_box_width, len(background_color)), dtype=np.uint8)
        sprite = pixel_array_palette[start_y:start_y + height, start_x: start_x + width]
        sprite_mask = image_ids[start_y:start_y + height, start_x: start_x + width]
        sprite[np.where(sprite_mask != image_id)] = 0
        img[delta_height:delta_height + sprite.shape[0], delta_width: delta_width + sprite.shape[1], 0] = sprite
        for y, row in enumerate(img):
            for x, palette_index in enumerate(row[:, 0]):
                img[y, x] = colors[palette_index, 1:]
        sprite_sheet[sprite_sheet_y: sprite_sheet_y + img.shape[0], sprite_sheet_x: sprite_sheet_x + img.shape[1]] = img
        sprite_sheet_x += img.shape[1]
        if sprite_sheet_x >= sprite_sheet.shape[1]:
            sprite_sheet_y += img.shape[0]
            sprite_sheet_x = 0
    sprite_sheet = Image.fromarray(sprite_sheet, mode='RGBA')
    sprite_sheet.save('{}_separated.png'.format(os.path.splitext(image_path)[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='The path to the sprite sheet you want to normalize/space out.')
    parser.add_argument('--sprites_per_row', '--columns', '-c', type=int, default=5, help='The number of sprites to have on each row.')
    parser.add_argument('--buffer_pixels', '--buffer', '-b', type=int, default=10, help='The padding to add to each side of the sprite.')
    args = parser.parse_args()
    space_out_sprites(args.image_path, args.sprites_per_row, args.buffer_pixels)
