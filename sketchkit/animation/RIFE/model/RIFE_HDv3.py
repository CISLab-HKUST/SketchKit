from sketchkit.animation.RIFE.model.IFNet_HDv3 import *

class Model:
    def __init__(self, model_dir, device):
        self.flownet = IFNet()
        self.load_model(model_dir, -1)
        self.eval()
        self.device = device
        self.flownet.to(device)

    def eval(self):
        self.flownet.eval()

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), False)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list, device=self.device)
        return merged[-1]
