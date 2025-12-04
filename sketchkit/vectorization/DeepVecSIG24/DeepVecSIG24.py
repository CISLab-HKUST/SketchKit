import os
import torch
import numpy as np
import cv2
from PIL import Image

from types import SimpleNamespace
from svgpathtools import wsvg

from sketchkit.core.sketch import Sketch, Path
from sketchkit.utils.transform import line_to_cubic

from sketchkit.vectorization.DeepVecSIG24.network.udc import CNN_2d_resnet
from sketchkit.vectorization.DeepVecSIG24.network.keypoint import PyramidalNet

from sketchkit.vectorization.DeepVecSIG24.utils.ndc_tools import refine_topology, lines_to_svg, map_to_lines, pre_to_map, init_base_coord
from sketchkit.vectorization.DeepVecSIG24.utils.keypt_tools import plot_udf_coord_numpy, udf_filter
from sketchkit.vectorization.DeepVecSIG24.utils.svg_tools import simplify_graph, open_svg_flatten, ramerDouglas, flatten_paths_to_pts_array


class DeepVecSIG24:
    """
    A class for image vectorization using the method of paper "Deep Sketch Vectorization via Implicit Surface Extraction"
    in SIGGRAPH 2024.

    Attributes:
        device (str): The device that is used.
    """
    def __init__(self, device: str="cuda"):
        """
        Initializes the DeepVecSIG24 vectorizer.

        Args:
            device (str, optional): The device that is used.
        """
        self.method = "DeepVecSIG24"
        self.device = device

        self.resize_to = 512  # downsacle image to its long edge equals resize_to, this option will not work if device is set to cpu
        self.refine = True  # enable dual contouring refinement
        self.rdp = True  # enable Ramer–Douglas–Peucker similification
        self.eps = 0.4
        self.down_rate = 8  # ratio for dual contouring downsampling
        self.skip_len = 4  # drop lines if it contains less than skip_len segements

        self._download_models()
        self.model_udf, self.args_udf, self.model_ndc, self.args_ndc = self._load_model(device)

    def _download_models(self):
        """Download the pretrained models
        """
        root = os.path.join(
            os.environ.get("HOME", "/"),
            ".sketchkit/models/vectorization/",
            self.method,
        )
        self.model_udf_path = os.path.join(root, "udf_full.pth")
        self.model_ndc_path = os.path.join(root, "ndc_full.pth")
        if not os.path.exists(self.model_udf_path) or not os.path.exists(self.model_ndc_path):
            cmd = "git clone https://huggingface.co/waterheater/deepsketch " + root
            os.system(cmd)
            os.remove(os.path.join(root, "line_thin.pth.tar"))
            os.remove(os.path.join(root, "ndc_light.pth"))
            os.remove(os.path.join(root, "netG.pth"))
            os.remove(os.path.join(root, "udf_light.pth"))

    def _load_model(self, device):
        # load pretrained model
        assert (os.path.exists(self.model_udf_path))
        assert (os.path.exists(self.model_ndc_path))

        model_udf, args_udf = self._load_checkpoint_udf(self.model_udf_path)
        model_udf = model_udf.to(device)
        model_ndc, args_ndc = self._load_checkpoint_ndc(self.model_ndc_path)
        model_ndc = model_ndc.to(device)
        print('Loaded pretrained models!')
        return model_udf, args_udf, model_ndc, args_ndc

    def _load_checkpoint_udf(self, filepath):
        """
        Loads model from a checkpoint
        we will need the distance threshold to accruately recover the UDF for the next NDC model
        """
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location="cpu")
        try:
            args_udf = SimpleNamespace(**checkpoint['param'])
        except BaseException:
            args_udf = checkpoint['param']
        args_udf = vars(args_udf)
        if "hourglass_input_channels" not in args_udf:
            args_udf["hourglass_input_channels"] = 64
        if "hourglass_channels" not in args_udf:
            args_udf["hourglass_channels"] = 96
        if "downsample_depth" not in args_udf:
            args_udf["downsample_depth"] = 3
        if "cardinality" not in args_udf:
            args_udf["cardinality"] = False
        args_udf = SimpleNamespace(**args_udf)
        loadmodel = PyramidalNet(
            n_classes=6,
            up_scale=args_udf.up_scale,
            hourglass_channels=args_udf.hourglass_channels,
            hourglass_input_channels=args_udf.hourglass_input_channels,
            downsample_depth=args_udf.downsample_depth,
            cardinality=args_udf.cardinality)
        loadmodel.load_state_dict(checkpoint['model_state_dict'])
        return loadmodel, args_udf

    def _load_checkpoint_ndc(self, filepath):
        """
        Loads model from a checkpoint
        we will need the distance threshold to accruately recover the UDF for the next NDC model
        """
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location='cpu')
        args_ndc = checkpoint['param']
        if isinstance(args_ndc, dict) == False:
            args_ndc = vars(args_ndc)
        if "noisy" not in args_ndc:
            args_ndc['noisy'] = False
        if "resnext_f" not in args_ndc:
            args_ndc['resnext_f'] = False
        if "resnext_i" not in args_ndc:
            args_ndc['resnext_i'] = False
        args_ndc = SimpleNamespace(**args_ndc)

        loadmodel = CNN_2d_resnet('cpu', channels=args_ndc.dims,
                                  layers=args_ndc.layers,
                                  multi_scale=args_ndc.msb,
                                  batch_norm=args_ndc.bn,
                                  drop_out=args_ndc.do,
                                  coord_conv=args_ndc.coord,
                                  noisy=args_ndc.noisy,
                                  resnext_feature=args_ndc.resnext_f,
                                  resnext_input=args_ndc.resnext_i)
        loadmodel.load_state_dict(checkpoint['model_state_dict'])
        return loadmodel, args_ndc

    def _load_img(self, img_input, device, up_scale,
                  resize=False,
                  path_to_out=None,
                  resize_to=1024):
        _, png = os.path.split(img_input)
        name, _ = os.path.splitext(png)

        try:
            img_np = np.array(Image.open(img_input).convert("L"))
        except BaseException:
            img_np = np.array(img_input.convert("L"))
        h, w = img_np.shape[0], img_np.shape[1]
        h_raw = h
        w_raw = w
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:
                background = np.ones((h, w, 3)) * 255
                img_np_rgb = img_np[:, :, 0:3]
                img_alpha = np.expand_dims(img_np[:, :, 3], -1).astype(float) / 255
                img_np = (img_np_rgb * img_alpha + background *
                          (1 - img_alpha)).mean(axis=-1)
            else:
                img_np = img_np.mean(axis=-1)
        if resize:
            long_side = h if h > w else w
            ratio = resize_to / long_side
            h = int(h * ratio)
            w = int(w * ratio)
            img_np = cv2.resize(img_np.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_AREA)
        h = h // 8 * 8
        w = w // 8 * 8
        img_np = img_np[:h, :w]
        img = torch.FloatTensor(img_np / 255).unsqueeze(0).unsqueeze(0)
        # change image color to blue
        img_np = np.repeat(img_np[..., np.newaxis], 3, axis=-1)
        # save updated input image if possible
        if path_to_out is not None:
            if up_scale:
                img_np = cv2.resize(
                    img_np, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
            if not os.path.exists(path_to_out):
                os.makedirs(path_to_out)
            Image.fromarray(img_np).save(os.path.join(path_to_out, name + '.png'))
        return img.to(device), img_np, (h_raw, w_raw)

    def _D(self, t):
        try:
            n = t.detach().cpu().numpy().squeeze()
            return n
        except BaseException:
            return t

    def _pre_udf(self, net, img):
        net.eval()
        with torch.no_grad():
            udf_topo_pre, udfs_pre = net.forward_with_downscale_outs(img)
            udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, udf_usm_pre = udfs_pre

            # udf_topo_pre = D(udf_topo_pre)
            udf_all_pre = self._D(udf_all_pre)
            udf_end_pre = self._D(udf_end_pre)
            udf_sharp_pre = self._D(udf_sharp_pre)
            udf_junc_pre = self._D(udf_junc_pre)
            usm_pre = self._D(udf_usm_pre)

        return udf_topo_pre, udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, usm_pre

    def _pre_svg(self, net, udf, gsize):
        base_coord = init_base_coord(udf[..., :-1, :-1], udf.device, gsize)
        net.eval()
        with torch.no_grad():
            edge_maps_pre, pt_map_pre = net.forward(udf, gsize)
        return edge_maps_pre, pt_map_pre + base_coord

    def _predict_UDF(
            self,
            img,
            img_np,
            model_udf,
            usm_thr=0.5):
        udf_topo_pre, udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, usm_pre = \
            self._pre_udf(model_udf, img)
        usm_pre_ = (usm_pre.clip(0, 1) < usm_thr) | (udf_all_pre.clip(0, 1) < usm_thr / 3)
        # extract keypoints from the prediction
        udf_all_pre = udf_filter(udf_all_pre)
        udf_end_pre = udf_filter(udf_end_pre)
        udf_sharp_pre = udf_filter(udf_sharp_pre)
        udf_junc_pre = udf_filter(udf_junc_pre)
        udfs_pre = [udf_end_pre, udf_sharp_pre, udf_junc_pre]
        h, w = udf_all_pre.shape
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
        keypt_pre_np, keypt_pre_dict = plot_udf_coord_numpy(
            img_np, udfs_pre, udf_all_pre)
        for k in keypt_pre_dict:
            keypt_pre_dict[k] = keypt_pre_dict[k] / 2

        return udf_topo_pre, usm_pre_, keypt_pre_dict, keypt_pre_np

    def _predict_SVG(
            self,
            udf_topo_pre,
            model_ndc,
            args_udf,
            args_ndc,
            refine=True):
        # predict SVG from UDF
        if args_udf.dist_clip != args_ndc.dist_clip:
            udf_topo_pre = (udf_topo_pre * args_udf.dist_clip).clip(0, args_ndc.dist_clip) / args_ndc.dist_clip
        else:
            udf_topo_pre = udf_topo_pre.clip(0, 1)
        gsizes = [0.5] if args_udf.up_scale else [1]
        edge_maps_pre, pt_map_pre = self._pre_svg(model_ndc, udf_topo_pre, gsizes)
        pt_map_pre = self._D(pt_map_pre.squeeze().permute(1, 2, 0))

        # reconstruct vector image from prediction
        edge_maps_pre = edge_maps_pre.permute(0, 2, 3, 1)[..., :-1].squeeze()
        edge_maps_pre_x, edge_maps_pre_y, _ = pre_to_map(edge_maps_pre)
        edge_maps_pre_xy = torch.stack((edge_maps_pre_x, edge_maps_pre_y), dim=-1)
        lines_pre, linemap_pre_x, linemap_pre_y, edge_maps_pre_xy = map_to_lines(
            self._D(edge_maps_pre_xy), pt_map_pre, refine=refine)

        return linemap_pre_x, linemap_pre_y, pt_map_pre, edge_maps_pre_xy

    def _simplify_SVG(
            self,
            path_to_svg,
            keypt_pre_dict,
            rdp_simplify=False,
            epsilon=0.4,
            skip_len=4):
        p, name = os.path.split(path_to_svg)
        name, _ = os.path.splitext(name)
        name = "_".join(name.split("_")[:-1])
        paths, (h, w) = open_svg_flatten(path_to_svg)
        strokes, nodes = simplify_graph(
            paths, keypt_pre_dict, mode="hybird", skip_len=skip_len)

        # strokes, _ = refine_topology_2nd(strokes, keypt_pre_dict, nodes[-1])
        if rdp_simplify:
            strokes = ramerDouglas(strokes, epsilon)
        attributes = [{"fill": 'none',
                       "stroke": "#000000",
                       "stroke-width": '1',
                       "stroke-linecap": "round"}] * len(strokes)
        wsvg(strokes,
             stroke_widths=[0.5] * len(strokes),
             dimensions=(w, h),
             filename=os.path.join(p, name + "_final.svg"),
             attributes=attributes)
        # also save another version for visulization
        self._vis_stroke(strokes, (h, w), os.path.join(p, name + "_vis.svg"))
        print("Results are saved to dir:", p)
        return strokes

    def _vis_stroke(self, strokes, canvas_size, save_path):
        h, w = canvas_size
        # get random colors for each stroke
        colors = np.random.randint(0, 255, size=(len(strokes), 3))
        colors = [tuple(c) for c in colors]
        # save to file
        # attributes = [{"fill":'none', "stroke-width":'1', "stroke-linecap":"round"}] * len(strokes)
        wsvg(strokes,
             colors=colors,
             stroke_widths=[2] * len(strokes),
             dimensions=(w, h),
             filename=save_path)

    def _paths_to_Sketch(self, paths):
        path_list = []  # list of (N_strokes, 4, 2)
        for path in paths:
            pts_array = flatten_paths_to_pts_array(path)  # (N, 2)
            stroke_list = []
            for i in range(1, len(pts_array)):
                p_start = pts_array[i - 1]  # (2)
                p_end = pts_array[i]  # (2)
                line = np.stack([p_start, p_end], axis=0)  # (2, 2)
                cubic = line_to_cubic(line)  # (4, 2)
                stroke_list.append(cubic)
            path_list.append(Path(stroke_list))
        return path_list

    def run(self, image_path: str, out_folder: str = "svg"):
        """
        Vectorize the input raster image or images in a folder.

        Args:
            image_path (str): a path to an input raster image or a folder with multiple images.
            out_folder (str, optional): an output folder to store the output svg file.

        Returns:
            sketch (Sketch): a Sketch instance.
        """
        img_base, img_name = os.path.split(image_path)
        name, _ = os.path.splitext(img_name)

        # load input data
        if self.device == 'cuda':
            ww, hh = Image.open(image_path).size
            longer = ww if ww > hh else hh
            resize = True if longer > self.resize_to else False
        else:
            resize = False
        img, img_np, canvas_size = self._load_img(image_path, self.device, self.args_udf.up_scale,
                                                  resize=resize, resize_to=self.resize_to)
        tensor_h, tensor_w = img.shape[2], img.shape[3]

        # predict UDF from sketch
        udf_topo_pre, usm_pre_, keypt_pre_dict, keypt_pre_np = self._predict_UDF(
            img, img_np, self.model_udf)

        # predict SVG from UDF
        linemap_pre_x, linemap_pre_y, pt_map_pre, edge_maps_pre_xy = self._predict_SVG(
            udf_topo_pre, self.model_ndc, self.args_udf, self.args_ndc, refine=self.refine)

        # USM surgery
        lines_pre, _, usm_applied, usm_uncertain = refine_topology(
            self._D(edge_maps_pre_xy),
            pt_map_pre,
            usm_pre_,
            linemap_pre_x,
            linemap_pre_y,
            keypt_pre_dict,
            down_rate=self.down_rate,
            downsample=True,
            full_auto_mode=True)
        lines_to_svg(
            lines_pre * 2,
            tensor_w * 2,
            tensor_h * 2,
            os.path.join(img_base, out_folder, name + "_refine.svg"))

        # stroke traversal
        paths_poly = self._simplify_SVG(
            os.path.join(img_base, out_folder, name + "_refine.svg"),
            keypt_pre_dict,
            rdp_simplify=self.rdp,
            epsilon=self.eps,
            skip_len=self.skip_len)
        os.remove(os.path.join(img_base, out_folder, name + "_refine.svg"))

        # convert to Sketch format
        path_list = self._paths_to_Sketch(paths_poly)
        sketch = Sketch(paths=path_list)
        return sketch
