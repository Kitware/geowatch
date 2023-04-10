import os
import torch

from PIL import Image

from ..misc.imutils import save_image
from .networks import define_G


class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder

        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        print('check point is :', os.path.join(self.checkpoint_dir, checkpoint_name))

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G

    def _visualize_pred(self, size=None):

        # pdb.set_trace()

        # This is original
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # pred_vis = pred * 255

        before = Image.fromarray(self.G_pred[0, 0, :, :].cpu().data.numpy())
        after = Image.fromarray(self.G_pred[0, 1, :, :].cpu().data.numpy())

        if size is not None:
            before = before.resize(size)
            after = after.resize(size)

        # img.imsave('/media/FastData/yguo/samples/predict/before.png', np.array(before))
        # img.imsave('/media/FastData/yguo/samples/predict/after.png', np.array(after))

        # This is my mod
        pred = -self.G_pred[0, 0, :, :] + self.G_pred[0, 1, :, :]

        pred_min = pred.min()
        pred_max = pred.max()

        pred_vis = (pred - pred_min) / (pred_max - pred_min) * 255
        pred_vis = pred_vis[None, None, :].detach()

        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]

        depth_in1 = batch['depth_A'].to(self.device)
        depth_in2 = batch['depth_B'].to(self.device)

        img_in1 = torch.cat((img_in1, depth_in1), 1)
        img_in2 = torch.cat((img_in2, depth_in2), 1)

        self.G_pred = self.net_G(img_in1, img_in2)
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self, size=None):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):

            # pdb.set_trace()

            file_name = os.path.join(self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name, size)
