from carb import Float3
from typing import Tuple, List
import numpy as np

_COLOR_T = Tuple[float, float, float, float]

# alpha value for the walls
_ALPHA_1 = 0.95
_ALPHA_2 = 0.5

# color palette from https://colorbrewer2.org/
_COLOR_ACCENT = [(240 / 255., 240 / 255., 240 / 255., 1.),
                (127 / 255., 201 / 255., 127 / 255., 1.),
				(190 / 255., 174 / 255., 212 / 255., 1.),
				(253 / 255., 192 / 255., 134 / 255., 1.),
				(255 / 255., 255 / 255., 153 / 255., 1.),
				(56 / 255., 108 / 255., 176 / 255., 1.),
				(240 / 255., 2 / 255., 127 / 255., 1.),
				(191 / 255., 91 / 255., 23 / 255., 1.),
				(102 / 255., 102 / 255., 102 / 255., 1.)]

def _carb_float3_add(a: Float3, b: Float3) -> Float3:
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z)

def _colors_add_alphas(colors: List[_COLOR_T], alphas: List[float]) -> List[_COLOR_T]:
    for i in range(len(colors)):
        assert len(colors[i]) == 4
        colors[i] = list(colors[i])
        colors[i][-1] = alphas[i]
        colors[i] = tuple(colors[i])
    return colors

def draw_edge_circle(
    R: float, H: float, color: _COLOR_T = (1.0, 1.0, 1.0, 1.0), line_size: float = 10.0, num_points: int = 100
):
    point_list_1 = []
    point_list_2 = []
    for i in range(num_points):
        theta0 = 2 * np.pi * i / num_points
        theta1 = 2 * np.pi * (i + 1) / num_points
        point_list_1.append(Float3(R * np.cos(theta0), R * np.sin(theta0), 0))
        point_list_1.append(Float3(R * np.cos(theta0), R * np.sin(theta0), H))
        point_list_2.append(Float3(R * np.cos(theta1), R * np.sin(theta1), 0))
        point_list_2.append(Float3(R * np.cos(theta1), R * np.sin(theta1), H))
    
    n = len(point_list_1)
    assert n == len(point_list_2)
    colors = [color for _ in range(n)]
    sizes = [line_size for _ in range(n)]

    return point_list_1, point_list_2, colors, sizes

def draw_wall_circle(
    R: float, H: float, color = _COLOR_ACCENT[0],
    alpha = 0.1, line_size: float = 10.0, num_points: int = 100
):
    point_list_1 = []
    point_list_2 = []
    colors = []
    for i in range(num_points, 0, -1):
        theta = np.pi * i / num_points
        point_list_1.append(Float3(R * np.cos(theta), R * np.sin(theta), 0))
        point_list_1.append(Float3(R * np.cos(theta), - R * np.sin(theta), 0))
        point_list_2.append(Float3(R * np.cos(theta), R * np.sin(theta), H))
        point_list_2.append(Float3(R * np.cos(theta), - R * np.sin(theta), H))

        color_accent = _colors_add_alphas([color], [alpha * (1 - np.cos(theta)) / 2])
        colors.extend([color_accent[0], color_accent[0]]) 
    
    sizes = [line_size for _ in range(2 * num_points)]

    return point_list_1, point_list_2, colors, sizes

def _draw_lines_args_merger(*args):
    buf = [[] for _ in range(4)]
    for arg in args:
        buf[0].extend(arg[0])
        buf[1].extend(arg[1])
        buf[2].extend(arg[2])
        buf[3].extend(arg[3])

    return (
        buf[0],
        buf[1],
        buf[2],
        buf[3],
    )

def draw_court_circle(
    R: float, H: float, color_edge: _COLOR_T = (1.0, 1.0, 1.0, 1.0), 
    color_wall = _COLOR_ACCENT[0], line_size: float = 10.0, num_points: int = 200
):
    return _draw_lines_args_merger(draw_wall_circle(R, H, color=color_wall, line_size=line_size, num_points=num_points), 
                                   draw_edge_circle(R, H, color=color_edge, line_size=line_size, num_points=num_points))