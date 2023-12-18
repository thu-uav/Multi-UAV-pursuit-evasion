from carb import Float3
from typing import Tuple, List

_COLOR_T = Tuple[float, float, float, float]

# alpha value for the walls
_ALPHA_1 = 0.95
_ALPHA_2 = 0.5

# color palette from https://colorbrewer2.org/
_COLOR_ACCENT = [(127 / 255., 201 / 255., 127 / 255., 1.),
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

def draw_point(
    drone_states, color: _COLOR_T = _COLOR_ACCENT[4], size: float = 10
):
    point_list = []
    for i in range(drone_states.shape[0]):
        point_list.append(Float3(drone_states[i, 0], drone_states[i, 1], drone_states[i, 2]))
    colors = [color for _ in range(drone_states.shape[0])]
    sizes = [size for _ in range(drone_states.shape[0])]
    return point_list, colors, sizes

