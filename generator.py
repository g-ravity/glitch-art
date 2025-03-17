import random
import time
import numpy as np
from PIL import Image, ImageColor, ImageFilter, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import concurrent.futures

# ----------------------
# Global Configuration
# ----------------------

CONFIG = {
    'width': 2480,
    'height': 1320,
    'target_width': 1920,
    'target_height': 1080,

    'shape_count_min': 3000,
    'shape_count_max': 5000,
    'shape_weights': {
        'fractal_tree': 0.30,
        'spiral': 0.15,
        'hexagon': 0.15,
        'pentagon': 0.08,
        'triangle': 0.08,
        'circle': 0.08,
        'rectangle': 0.08,
        'ellipse': 0.04,
        'rhombus': 0.04
    },

    'sort_probability': 0.5,

    'gaussian_blur_sigma_1': (0.2, 0.3),
    'gaussian_blur_sigma_2': (0.1, 0.2),
    'gaussian_blur_sigma_3': 0.25,

    'segment_size_min': 45,
    'segment_size_max': 80,

    'rotation_angle_min': -60,
    'rotation_angle_max': 90,

    'num_frames': 30,
    'num_waves_min': 1,
    'num_waves_max': 2,

    'animation_type': random.choice(['wave', 'zoom', 'pulse', 'glitch']),
    'zoom_range': (0.8, 1.2),
    'pulse_intensity': 0.3,
    'glitch_intensity': 0.1,

    'wave_amplitude_min': 3,
    'wave_amplitude_max': 8,
    'wave_frequency_min': 0.04,
    'wave_frequency_max': 0.06,
    'wave_speed': 2 * np.pi,
    # 1.0 = current speed, 2.0 = half speed, 0.5 = double speed
    'gif_speed': 1.5,

    'hue_shift_amount': 60,
    'hue_shift_enabled': True,

    'base_palettes': [
        # Original
        ["#9B5523", "#EAB530", "#657B3F", "#584D9B", "#5692C4", "#9B109B"],
        # Coastal
        ["#365663", "#3AAD9F", "#F9D47A", "#F4B271", "#F77F61", "#9B109B"],
        # Forest
        ["#707C48", "#384628", "#FFFAF0", "#EDB16E", "#CC7C35", "#654565"],
        # Muted
        ["#32324B", "#5A5E79", "#AAA9A8", "#D9BDB7", "#FFF9F4", "#AA9F8A"],
        # Ocean
        ["#002229", "#006F83", "#1AA3A6", "#A4E2CD", "#F9E8B6", "#FE9B10"],
        # Ruby
        ["#690D32", "#901F3F", "#B4234C", "#D9285A", "#FF5D7D", "#FF859F"],
        # Sunset
        ["#FF8B10", "#FF9810", "#FFA510", "#FFB210", "#FFBA10", "#FFC710"],
        # Electric
        ["#3D10FF", "#7A10FF", "#9910FF", "#B110FF", "#C110FF", "#CC10FF"],
        # Forest (another variant)
        ["#105B33", "#107400", "#108200", "#109000", "#48C010", "#80F010"],
        # Ocean (another variant)
        ["#001029", "#001044", "#00105F", "#00107A", "#001095", "#0010B0"],

    ]
}

# ----------------------
# Helper Functions
# ----------------------


def draw_polygon(draw, x, y, sides, size, rotation, color):
    points = []
    for i in range(sides):
        angle = rotation + (2 * np.pi * i / sides)
        px = x + size * np.cos(angle)
        py = y + size * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=color)


def draw_spiral(draw, x, y, size, color):
    points = []
    for i in range(360):
        angle = i * np.pi / 180
        r = (size * i) / 360
        px = x + r * np.cos(angle)
        py = y + r * np.sin(angle)
        points.append((int(px), int(py)))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)


def get_fractal_params():
    seed = int(time.time() * 1000) % 10000
    random.seed(seed)

    return {
        'branch_ratio': random.uniform(0.6, 0.8),
        'angle_spread': random.uniform(0.3, 0.7),
        'width_decay': random.uniform(0.7, 0.9),
        'branch_twist': random.uniform(-0.2, 0.2),
        'style': random.choice(['symmetric', 'asymmetric', 'twisted'])
    }


def draw_fractal_tree(draw, x, y, length, angle, depth, color):
    if depth > 0:
        params = get_fractal_params()
        x2 = x + int(length * np.cos(angle))
        y2 = y - int(length * np.sin(angle))
        draw.line([(x, y), (x2, y2)], fill=color, width=max(1, depth))

        if params['style'] == 'symmetric':
            # Symmetric branching
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'],
                              angle + params['angle_spread'], depth - 1, color)
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'],
                              angle - params['angle_spread'], depth - 1, color)
        elif params['style'] == 'asymmetric':
            # Asymmetric branching
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'] * 0.9,
                              angle + params['angle_spread'], depth - 1, color)
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'] * 1.1,
                              angle - params['angle_spread'] * 0.8, depth - 1, color)
        else:
            # Twisted branching
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'],
                              angle + params['angle_spread'] +
                              params['branch_twist'] * depth,
                              depth - 1, color)
            draw_fractal_tree(draw, x2, y2, length * params['branch_ratio'],
                              angle - params['angle_spread'] +
                              params['branch_twist'] * depth,
                              depth - 1, color)

        random.seed()


def draw_shapes_chunk(args):
    """
    Draws shapes for a chunk of shape indices in parallel.
    Returns an np.array representing the drawn chunk.
    """
    start_idx, end_idx, width, height, palette, cfg = args
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    shapes = list(cfg['shape_weights'].keys())
    weights = list(cfg['shape_weights'].values())

    for _ in range(start_idx, end_idx):
        x = random.randint(0, width - 1)
        y = int(random.triangular(0, height - 1, height // 2))

        color = random.choice(palette)
        color = tuple(
            min(255, max(0, c + random.randint(-20, 20)))
            for c in color
        )

        shape_type = random.choices(shapes, weights=weights, k=1)[0]

        if shape_type == "spiral":
            size = random.randint(15, 40)
            draw_spiral(draw, x, y, size, color)
        elif shape_type == "fractal_tree":
            length = random.randint(15, 25)
            angle = random.uniform(0, 2 * np.pi)
            draw_fractal_tree(draw, x, y, length, angle, 4, color)
        elif shape_type in ["triangle", "pentagon", "hexagon"]:
            size = random.randint(10, 30)
            sides = {"triangle": 3, "pentagon": 5, "hexagon": 6}[shape_type]
            rotation = random.uniform(0, 2 * np.pi)
            draw_polygon(draw, x, y, sides, size, rotation, color)
        elif shape_type == "rhombus":
            size = random.randint(10, 25)
            points = [
                (x, y - size),
                (x + size, y),
                (x, y + size),
                (x - size, y)
            ]
            draw.polygon(points, fill=color)
        else:
            if shape_type == "circle":
                size = random.randint(5, 20)
                draw.ellipse([x, y, x + size, y + size], fill=color)
            else:
                w = random.randint(8, 25)
                h = random.randint(8, 25)
                x2 = min(x + w, width)
                y2 = min(y + h, height)
                if shape_type == "rectangle":
                    draw.rectangle([x, y, x2, y2], fill=color)
                else:
                    draw.ellipse([x, y, x2, y2], fill=color)

    return np.array(img)


def create_gradient_color(color1, color2, ratio):
    return tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))


def process_column(args):
    """
    Sorts each column by brightness with some probability to create
    vertical glitch-like effects.
    """
    data, j, sort_probability = args
    column = data[:, j].copy()

    if random.random() > sort_probability:
        brightness = np.mean(column.reshape(-1, 3), axis=1)
        sorted_indices = np.argsort(brightness)
        column = column.reshape(-1, 3)[sorted_indices].reshape(column.shape)

    return j, column


def generate_base_image(cfg, palette, background_color):
    """
    Draws shapes on an empty image in parallel chunks,
    applies blur and column-sorting for glitch effects,
    returns a single combined PIL image.
    """
    width = cfg['width']
    height = cfg['height']
    sort_probability = cfg['sort_probability']

    num_shapes = random.randint(cfg['shape_count_min'], cfg['shape_count_max'])
    num_processes = 8
    chunk_size = num_shapes // num_processes

    chunks = [
        (i * chunk_size, (i + 1) * chunk_size, width, height, palette, cfg)
        for i in range(num_processes)
    ]

    with Pool() as pool:
        results = pool.map(draw_shapes_chunk, chunks)

    data = np.zeros((height, width, 3), dtype=np.uint8)
    for result in results:
        data = np.maximum(data, result)

    sigma1 = random.uniform(*cfg['gaussian_blur_sigma_1'])
    data = gaussian_filter(data, sigma=sigma1)

    column_args = [(data, j, sort_probability) for j in range(width)]
    with Pool() as pool:
        col_results = pool.map(process_column, column_args)

    for j, column in col_results:
        data[:, j] = column

    sigma2 = random.uniform(*cfg['gaussian_blur_sigma_2'])
    data = gaussian_filter(data, sigma=sigma2)

    data = gaussian_filter(data, sigma=cfg['gaussian_blur_sigma_3'])

    return Image.fromarray(data.astype(np.uint8))


def create_rotated_version(cfg, base_image, second_image, background_color):
    """
    Creates a rotated version of second_image, then crops it 
    to the same size as base_image.
    """
    width = cfg['width']
    height = cfg['height']

    diagonal = int(np.sqrt(width**2 + height**2))
    padding_width = (diagonal - width) // 2
    padding_height = (diagonal - height) // 2

    padded_image = Image.new('RGB', (diagonal, diagonal), background_color)
    padded_image.paste(second_image, (padding_width, padding_height))

    angle = random.uniform(cfg['rotation_angle_min'],
                           cfg['rotation_angle_max'])
    rotated_padded = padded_image.rotate(angle, Image.Resampling.BILINEAR)

    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
    rotated_image = rotated_padded.crop(
        (left, top, left + width, top + height))

    return rotated_image


def apply_animation(cfg, final_image):
    """
    Creates various types of animations from final_image.
    """
    width = cfg['target_width']
    height = cfg['target_height']
    num_frames = cfg['num_frames']
    frames = []

    if cfg['animation_type'] == 'wave':
        frames = create_wave_frames(cfg, final_image)
    elif cfg['animation_type'] == 'zoom':
        frames = create_zoom_frames(cfg, final_image)
    elif cfg['animation_type'] == 'pulse':
        frames = create_pulse_frames(cfg, final_image)
    elif cfg['animation_type'] == 'glitch':
        frames = create_glitch_frames(cfg, final_image)

    if not frames:
        print(
            f"Warning: No frames generated for animation type '{cfg['animation_type']}'")
        return

    base_duration = 30
    frame_duration = int(base_duration * cfg['gif_speed'])

    frames[0].save(
        "infinite_art.gif",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )


def create_zoom_frames(cfg, image):
    frames = []
    num_frames = cfg['num_frames']
    min_zoom, max_zoom = cfg['zoom_range']

    bg_size = (int(image.width * 1.5), int(image.height * 1.5))
    background = image.resize(bg_size, Image.Resampling.LANCZOS)
    background = background.filter(
        ImageFilter.GaussianBlur(radius=1))

    for i in range(num_frames):
        progress = (1 + np.cos(2 * np.pi * i / num_frames)) / 2
        zoom_factor = min_zoom + (max_zoom - min_zoom) * progress

        new_size = (
            int(image.width * zoom_factor),
            int(image.height * zoom_factor)
        )

        zoomed = image.resize(new_size, Image.Resampling.LANCZOS)
        bg_zoom_factor = min_zoom * 1.5 + \
            (max_zoom - min_zoom) * progress * 0.5
        bg_new_size = (
            int(image.width * bg_zoom_factor),
            int(image.height * bg_zoom_factor)
        )
        zoomed_bg = background.resize(bg_new_size, Image.Resampling.LANCZOS)

        frame = Image.new('RGB', (image.width, image.height), (0, 0, 0))

        bg_left = (frame.width - zoomed_bg.width) // 2
        bg_top = (frame.height - zoomed_bg.height) // 2
        frame.paste(zoomed_bg, (bg_left, bg_top))

        left = (frame.width - zoomed.width) // 2
        top = (frame.height - zoomed.height) // 2
        frame.paste(zoomed, (left, top))

        frame = apply_hue_shift(cfg, frame, i)
        frames.append(frame)

    return frames


def create_pulse_frames(cfg, image):
    frames = []
    num_frames = cfg['num_frames']
    intensity = cfg['pulse_intensity']

    for i in range(num_frames):
        progress1 = (1 + np.sin(2 * np.pi * i / num_frames)) / 2
        progress2 = (1 + np.cos(2 * np.pi * i / num_frames)) / 2
        progress3 = (1 + np.sin(4 * np.pi * i / num_frames)) / 2

        frame = image.copy()
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(1 + intensity * progress1)

        enhancer = ImageEnhance.Contrast(frame)
        frame = enhancer.enhance(1 + (intensity * 0.5) * progress2)

        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(1 + (intensity * 0.7) * progress3)

        frame = apply_hue_shift(cfg, frame, i)
        frames.append(frame)

    return frames


def create_wave_frames(cfg, final_image):
    """
    Creates a flowing wave GIF from final_image.
    """
    width = cfg['target_width']
    height = cfg['target_height']
    num_frames = cfg['num_frames']
    frames = []

    num_waves = random.randint(cfg['num_waves_min'], cfg['num_waves_max'])
    wave_params = []
    for _ in range(num_waves):
        wave_params.append({
            'amplitude': random.uniform(cfg['wave_amplitude_min'], cfg['wave_amplitude_max']),
            'frequency': random.uniform(cfg['wave_frequency_min'], cfg['wave_frequency_max']),
            'direction': random.uniform(0, 2 * np.pi),
            'speed': cfg['wave_speed']
        })

    x_grid, y_grid = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64)
    )

    frames = []
    source_array = np.array(final_image)

    for frame_idx in range(num_frames):
        phase = (frame_idx / num_frames) * (2 * np.pi)
        distorted_x = x_grid.copy()
        distorted_y = y_grid.copy()

        for wave in wave_params:
            angle = wave['direction']
            dx = wave['amplitude'] * np.sin(
                wave['frequency'] *
                (x_grid * np.cos(angle) + y_grid * np.sin(angle)) + phase
            )
            dy = wave['amplitude'] * np.cos(
                wave['frequency'] *
                (x_grid * np.sin(angle) - y_grid * np.cos(angle)) + phase
            )
            distorted_x += dx
            distorted_y += dy

        distorted_x = np.clip(distorted_x, 0, width - 1).astype(np.int32)
        distorted_y = np.clip(distorted_y, 0, height - 1).astype(np.int32)

        distorted_frame = np.zeros_like(source_array)
        distorted_frame[y_grid.astype(np.int32), x_grid.astype(np.int32)] = \
            source_array[distorted_y, distorted_x]

        frame_image = Image.fromarray(distorted_frame)
        frame_image = frame_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        frame_image = apply_hue_shift(cfg, frame_image, frame_idx)
        frames.append(frame_image)

    return frames


def create_glitch_frames(cfg, image):
    frames = []
    num_frames = cfg['num_frames']
    intensity = cfg['glitch_intensity']

    for i in range(num_frames):
        frame = image.copy()
        data = np.array(frame)

        if random.random() < intensity * 1.5:
            for _ in range(random.randint(1, 3)):
                channel = random.randint(0, 2)
                shift = random.randint(-15, 15)
                data[:, :, channel] = np.roll(
                    data[:, :, channel], shift, axis=1)

        if random.random() < intensity * 1.2:
            num_scanlines = random.randint(3, 8)
            for _ in range(num_scanlines):
                y = random.randint(0, frame.height - 10)
                height = random.randint(2, 8)
                data[y:y+height, :] = data[y:y+height, :] * 1.8

        if random.random() < intensity:
            num_blocks = random.randint(1, 3)
            for _ in range(num_blocks):
                x = random.randint(0, frame.width - 50)
                y = random.randint(0, frame.height - 20)
                w = random.randint(30, 100)
                h = random.randint(5, 15)
                block = data[y:y+h, x:x+w].copy()
                shift = random.randint(-10, 10)
                data[y:y+h, x:x+w] = np.roll(block, shift, axis=1)

        if random.random() < intensity * 0.8:
            noise_mask = np.random.random(data.shape) > 0.99
            data[noise_mask] = np.random.randint(0, 255, size=(3,))

        frame = Image.fromarray(data)
        frame = apply_hue_shift(cfg, frame, i)
        frames.append(frame)

    return frames


def apply_hue_shift(cfg, image, frame_idx):
    """Helper function to apply hue shift to a frame"""
    if cfg['hue_shift_enabled']:
        hue_shift = (cfg['hue_shift_amount'] *
                     (1 + np.cos(2 * np.pi * frame_idx / cfg['num_frames']))) / 2
        image = image.convert('HSV')
        h, s, v = image.split()
        h = h.point(lambda x: (x + int(hue_shift)) % 256)
        image = Image.merge('HSV', (h, s, v)).convert('RGB')
    return image


def main():
    cfg = CONFIG

    palette = random.choice(cfg['base_palettes'])
    palette = [
        tuple(int(c * random.uniform(0.8, 1.2))
              for c in ImageColor.getrgb(color))
        for color in palette
    ]

    background_color = tuple(random.randint(60, 120) for _ in range(3))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(
            generate_base_image, cfg, palette, background_color)

        palette2 = random.choice(cfg['base_palettes'])
        palette2 = [
            tuple(int(c * random.uniform(0.8, 1.2))
                  for c in ImageColor.getrgb(color))
            for color in palette2
        ]
        future2 = executor.submit(
            generate_base_image, cfg, palette2, background_color)

        first_image = future1.result()
        second_image = future2.result()

    rotated_image = create_rotated_version(
        cfg, first_image, second_image, background_color)

    image_choice = random.choice(['first', 'rotated', 'blend'])
    if image_choice == 'first':
        final_image = first_image
    elif image_choice == 'rotated':
        final_image = rotated_image
    else:
        blend_value = random.uniform(0.1, 0.5)
        final_image = Image.blend(first_image, rotated_image, blend_value)

    width = cfg['width']
    height = cfg['height']
    tw = cfg['target_width']
    th = cfg['target_height']

    left = (width - tw) // 2
    top = (height - th) // 2
    final_image = final_image.crop((left, top, left + tw, top + th))

    final_image.save("pixel_art.png")
    apply_animation(cfg, final_image)
    final_image.show()


if __name__ == '__main__':
    main()
