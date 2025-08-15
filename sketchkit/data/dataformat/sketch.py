class Vertex():
    def __init__(self, pos):
        self.pos = pos


class BaseStroke:
    def __init__(self, p_start: Vertex, p_end: Vertex, thickness, pressure, color, opacity, texture):
        self.p_start = p_start
        self.p_end = p_end
        self.thickness = thickness
        self.pressure = pressure
        self.color = color
        self.opacity = opacity
        self.texture = texture


class Line(BaseStroke):
    def __init__(self, p_start: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(Line, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)


class QuadraticBezierCurve(BaseStroke):
    def __init__(self, p_start: Vertex, p_crtl: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(QuadraticBezierCurve, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)
        self.p_crtl = p_crtl


class CubicBezierCurve(BaseStroke):
    def __init__(self, p_start: Vertex, p_crtl1: Vertex, p_crtl2: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(CubicBezierCurve, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)
        self.p_crtl1 = p_crtl1
        self.p_crtl2 = p_crtl2


class Path():
    def __init__(self, stroke_type: str, stroke_list: list,
                 thickness_list=None, pressure_list=None,
                 color_list=None, opacity_list=None, texture_list=None):
        """
        The class for a path or stroke chain.
        :param stroke_type:
            "line", "qbc", or "cbc".
        :param stroke_list:
            A list of strokes, each stroke in shape (N_points, 2),
            with N_points = 2/3/4 for "line", "qbc", and "cbc", respectively.
        :param thickness_list:
            In size (N_strokes). Set to None by default.
        :param pressure_list:
            In size (N_strokes). Set to None by default.
        :param color_list:
            In shape (N_strokes, 3). Set to None by default.
        :param opacity_list:
            In size (N_strokes). Set to None by default.
        :param texture_list:
            In size (N_strokes). Set to None by default.
        """
        self.stroke_type = stroke_type
        self.stroke_list = []
        stroke_num = len(stroke_list)
        for stroke_i in range(stroke_num):
            stroke = stroke_list[stroke_i]
            stroke_thickness = thickness_list[stroke_i] if thickness_list is not None else None
            stroke_pressure = pressure_list[stroke_i] if pressure_list is not None else None
            stroke_color = color_list[stroke_i] if color_list is not None else None
            stroke_opacity = opacity_list[stroke_i] if opacity_list is not None else None
            stroke_texture = texture_list[stroke_i] if texture_list is not None else None

            if stroke_type == "line":
                p_start = Vertex(stroke[0])
                p_end = Vertex(stroke[1])
                self.stroke_list.append(Line(p_start, p_end,
                                             stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            elif stroke_type == "qbc":
                p_start = Vertex(stroke[0])
                p_crtl = Vertex(stroke[1])
                p_end = Vertex(stroke[2])
                self.stroke_list.append(QuadraticBezierCurve(p_start, p_crtl, p_end,
                                                             stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            elif stroke_type == "cbc":
                p_start = Vertex(stroke[0])
                p_crtl1 = Vertex(stroke[1])
                p_crtl2 = Vertex(stroke[2])
                p_end = Vertex(stroke[3])
                self.stroke_list.append(CubicBezierCurve(p_start, p_crtl1, p_crtl2, p_end,
                                                         stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            else:
                raise Exception('Unknown stroke_type:', stroke_type)

    def get_length(self):
        return len(self.stroke_list)


class SketchVG():
    def __init__(self, stroke_type: str, stroke_chain_list: list,
                 stroke_thickness_list=None, stroke_pressure_list=None,
                 stroke_color_list=None, stroke_opacity_list=None, stroke_texture_list=None):
        """
        The universal sketch class.
        :param stroke_type:
            "line", "qbc", or "cbc". For Line, QuadraticBezierCurve, and CubicBezierCurve, respectively.
        :param stroke_chain_list:
            A list of stroke chains, each chain in shape (N_strokes, N_points, 2),
            with N_points = 2/3/4 for "line", "qbc", and "cbc", respectively.
        :param stroke_thickness_list:
            A list of stroke thickness, each in size (N_strokes). Set to None by default.
        :param stroke_pressure_list:
            A list of stroke pressure, each in size (N_strokes). Set to None by default.
        :param stroke_color_list:
            A list of stroke colors, each in shape (N_strokes, 3). Set to None by default.
        :param stroke_opacity_list:
            A list of stroke opacity, each in size (N_strokes). Set to None by default.
        :param stroke_texture_list:
            A list of stroke texture, each in size (N_strokes). Set to None by default.
        """
        self.stroke_type = stroke_type
        self.path_list = []
        for chain_i in range(len(stroke_chain_list)):
            stroke_chain = stroke_chain_list[chain_i]
            stroke_chain_thickness = stroke_thickness_list[chain_i] if stroke_thickness_list is not None else None
            stroke_chain_pressure = stroke_pressure_list[chain_i] if stroke_pressure_list is not None else None
            stroke_chain_color = stroke_color_list[chain_i] if stroke_color_list is not None else None
            stroke_chain_opacity = stroke_opacity_list[chain_i] if stroke_opacity_list is not None else None
            stroke_chain_texture = stroke_texture_list[chain_i] if stroke_texture_list is not None else None

            self.path_list.append(Path(stroke_type, stroke_chain,
                                       stroke_chain_thickness, stroke_chain_pressure,
                                       stroke_chain_color, stroke_chain_opacity, stroke_chain_texture))

    def get_path_num(self):
        return len(self.path_list)

    def get_stroke_num(self):
        path_num = self.get_path_num()
        stroke_num = 0
        for path_i in range(path_num):
            stroke_num += self.path_list[path_i].get_length()
        return stroke_num

    def rasterize(self):
        # TODO:
        print()

    def vectorize(self):
        # TODO:
        print()

    def save(self, save_path):
        # TODO:
        print()
