# PhotoSketch Backend

The PhotoSketch integration wraps the Pix2Pix generator from\
[Photo-Sketching: Inferring Contour Drawings from Images](https://arxiv.org/abs/1901.00542)\
so it can be used through `sketchkit.image2sketch`.

The pretrained weights are available at\
<https://drive.google.com/file/d/1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj/view>.\
Place `latest_net_G.pth` inside `~/.sketchkit/image2sketch/photosketch/pretrained/`\
or under `sketchkit/image2sketch/photosketch/assets/` to skip downloading.\
Otherwise the wrapper downloads and caches the checkpoint automatically (requires `gdown`).
