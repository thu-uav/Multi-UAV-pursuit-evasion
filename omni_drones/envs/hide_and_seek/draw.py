from carb import Float3
from typing import Tuple, List
import numpy as np

_COLOR_T = Tuple[float, float, float, float]

# alpha value for the walls
_ALPHA_1 = 0.95
_ALPHA_2 = 0.5

# color palette from https://colorbrewer2.org/
_COLOR_ACCENT = [
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

def draw_edge(
    W: float, L: float, H: float, color: _COLOR_T = (1.0, 1.0, 1.0, 1.0), line_size: float = 10.0
):
    point_list_1 = [
        Float3(-L / 2, -W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(-L / 2, -W / 2, 0),
        Float3(L / 2, -W / 2, 0),

        Float3(-L / 2, -W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        
        Float3(-L / 2, -W / 2, H),
        Float3(-L / 2, W / 2, H),
        Float3(-L / 2, -W / 2, H),
        Float3(L / 2, -W / 2, H),
    ]
    point_list_2 = [
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, W / 2, 0),
        
        Float3(-L / 2, -W / 2, H),
        Float3(-L / 2, W / 2, H),
        Float3(L / 2, -W / 2, H),
        Float3(L / 2, W / 2, H),
        
        Float3(L / 2, -W / 2, H),
        Float3(L / 2, W / 2, H),
        Float3(-L / 2, W / 2, H),
        Float3(L / 2, W / 2, H),
    ]

    n = len(point_list_1)
    assert n == len(point_list_2)
    colors = [color for _ in range(n)]
    sizes = [line_size for _ in range(n)]

    return point_list_1, point_list_2, colors, sizes

def draw_wall(
    W: float, L: float, H: float, color = _COLOR_ACCENT,
    alpha = [_ALPHA_1, _ALPHA_2, _ALPHA_2], line_size: float = 10.0
):
    n = round(40 * W)
    n_wall = 3

    point_list_1 = []
    point_list_2 = []

    for i in range(n):
        point_list_1.append(Float3(-L / 2, -W / 2 + i * W / n, 0))
        point_list_2.append(Float3(-L / 2, -W / 2 + i * W / n, H))
        point_list_1.append(Float3(-L / 2 + i * L / n, -W / 2, 0))
        point_list_2.append(Float3(-L / 2 + i * L / n, -W / 2, H))
        point_list_1.append(Float3(-L / 2 + i * L / n, W / 2, 0))
        point_list_2.append(Float3(-L / 2 + i * L / n, W / 2, H))
        # point_list_1.append(Float3(L / 2, -W / 2 + i * W / n, 0))
        # point_list_2.append(Float3(L / 2, -W / 2 + i * W / n, H))

    color_accent = _colors_add_alphas(color[:n_wall], alpha[:n_wall])
    colors = [col_a for _ in range(n) for col_a in color_accent]
    sizes = [line_size for _ in range(n_wall*n)]
     
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

def draw_court(
    W: float, L: float, H: float, color_edge: _COLOR_T = (1.0, 1.0, 1.0, 1.0), 
    color_wall = _COLOR_ACCENT, line_size: float = 10.0
):
    return _draw_lines_args_merger(draw_wall(W, L, H, color=color_wall, line_size=line_size), 
                                   draw_edge(W, L, H, color=color_edge, line_size=line_size))

def draw_traj(
    drone_pos, drone_vel, dt: float = 0.01, color: _COLOR_T = _COLOR_ACCENT[4], size: float = 10.0
):
    drone_pos_dt = drone_pos + drone_vel * dt
    point_list1 = []
    point_list2 = []
    for i in range(drone_pos.shape[0]):
        point_list1.append(Float3(drone_pos[i, 0], drone_pos[i, 1], drone_pos[i, 2]))
        point_list2.append(Float3(drone_pos_dt[i, 0], drone_pos_dt[i, 1], drone_pos_dt[i, 2]))
    colors = [color for _ in range(drone_pos.shape[0])]
    sizes = [size for _ in range(drone_pos.shape[0])]
    return point_list1, point_list2, colors, sizes

def draw_range(
    pos, xaxis, yaxis, zaxis, drange: float,
    color: _COLOR_T = (1.0, 1.0, 1.0, 0.05), size: float = 5.0, num: int = 100
):
    point_list = []
    for i in range(pos.shape[0]):
        posi = pos[i, :]
        for j in range(1, num):
            num_phi = round(num / 2 - abs(j - num / 2)) * 4
            for k in range(num_phi):
                theta = np.pi * j / num
                phi = 2 * np.pi * k / num_phi
                point1 = posi + drange * (np.sin(theta) * np.cos(phi) * xaxis + 
                                          np.sin(theta) * np.sin(phi) * yaxis + 
                                          np.cos(theta) * zaxis)
                # point2 = posi + drange * (np.sin(theta) * np.cos(phi) * xaxis +
                #                           np.sin(theta) * np.sin(phi) * yaxis + 
                #                           np.cos(theta) * zaxis)
                # breakpoint()
                point_list.append(Float3(point1[0], point1[1], point1[2]))
                # point_list.append(Float3(point2[0], point2[1], point2[2]))
    n = len(point_list)
    colors = [color for _ in range(n)]
    sizes = [size for _ in range(n)]

    return point_list, colors, sizes

def draw_axis(
    pos, xaxis, yaxis, zaxis, drange: float, size: float = 5.0, num: int = 50
):
    point_list = []
    for i in range(xaxis.shape[0]):
        posi = pos[i, :]
        for j in range(num):
            pointx = posi + 0.5 * drange * j / num * xaxis
            pointy = posi + 0.5 * drange * j / num * yaxis
            pointz = posi + 0.5 * drange * j / num * zaxis
            point_list.extend([Float3(pointx[0], pointx[1], pointx[2]),
                               Float3(pointy[0], pointy[1], pointy[2]),
                               Float3(pointz[0], pointz[1], pointz[2])])
    n = len(point_list)
    colors_axis = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    colors = [color for _ in range(num) for color in colors_axis]
    sizes = [size for _ in range(n)]

    return point_list, colors, sizes

def _draw_points_args_merger(*args):
    buf = [[] for _ in range(4)]
    for arg in args:
        buf[0].extend(arg[0])
        buf[1].extend(arg[1])
        buf[2].extend(arg[2])

    return (
        buf[0],
        buf[1],
        buf[2],
    )

def draw_catch(
    pos, xaxis, yaxis, zaxis, drange: float,
    color: _COLOR_T = (1.0, 1.0, 1.0, 0.1), size_range: float = 20.0, num_range: int = 20,
    size_axis: float = 5.0, num_axis: int = 20
):
    return _draw_points_args_merger(
        draw_range(pos, xaxis, yaxis, zaxis, drange, color, size_range, num_range), 
        # draw_axis(pos, xaxis, yaxis, zaxis, drange, size_axis, num_axis)
    )

def draw_detection(
    pos, xaxis, yaxis, zaxis, drange: float,
    color: _COLOR_T = _COLOR_ACCENT[0], size_range: float = 20.0, num_range: int = 20,
    size_axis: float = 5.0, num_axis: int = 20
):
    return _draw_points_args_merger(
        draw_range(pos, xaxis, yaxis, zaxis, drange, color, size_range, num_range), 
        # draw_axis(pos, xaxis, yaxis, zaxis, drange, size_axis, num_axis)
    )
