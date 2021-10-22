import torch.nn as nn
from mmdet.ops.build import build_op
import torch.nn.functional as F
from mmcv.cnn import Scale
from ..builder import NECKS
from mmcv.cnn import ConvModule, xavier_init

@NECKS.register_module()
class YLFPNv2(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, out_channels,
                 norm_cfg=dict(type="BN"),
                 act_cfg=dict(type="ReLU"),
                 num_stack=1,
                 num_output=5,
                 start_level=0,
                 conv_cfg=dict(type="Conv2d"),
                 depth_wise=False,):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """

        super(YLFPNv2, self).__init__()

        in_channels = in_channels[start_level:]
        self.start_level = start_level
        self.num_extra = None

        num_input = len(in_channels)
        if num_input != num_output:
           self.extra_convs = nn.ModuleList()
           assert len(in_channels) <= num_output, "in_channels must smaller than out_channels"
           self.num_extra = num_output-num_input
           for i in range(self.num_extra):
               self.extra_convs.append(ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=dict(type='BN', requires_grad=True)))
               in_channels.append(in_channels[-1])


        # Conv layers
        self.lateral_P3 = ConvModule(in_channels[0], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P4 = ConvModule(in_channels[1], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P5 = ConvModule(in_channels[2], out_channels, kernel_size=1, norm_cfg=norm_cfg)

        if self.num_extra:
            self.lateral_P6 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)
            self.lateral_P7 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)
        else:
            self.lateral_P6 = ConvModule(in_channels[3], out_channels, kernel_size=1, norm_cfg=norm_cfg)
            self.lateral_P7 = ConvModule(in_channels[4], out_channels, kernel_size=1, norm_cfg=norm_cfg)

        self.conv7_1 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv6_1 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv5_1 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv4_1 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3_1 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv7_2 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv6_2 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv5_2 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv4_2 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3_2 = ConvModule(out_channels, out_channels, kernel_size=3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dowm_con6_7 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dowm_con5_6 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dowm_con3_4 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dowm_con4_5 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.upsample7_6 = F.interpolate
        self.upsample6_5 = F.interpolate
        self.upsample5_4 = F.interpolate
        self.upsample4_3 = F.interpolate

        self.shortcut6_7 = F.interpolate
        self.shortcut3_2 = F.interpolate

        self.scale7_6 = Scale()
        self.scale6_5 = Scale()
        self.scale5_4 = Scale()
        self.scale4_3 = Scale()

        self.scale6_7 = Scale()
        self.scale5_6 = Scale()

        self.scale3_4 = Scale()

        self.shortcut_scale4_5 = Scale()

    def forward(self, x):
        '''
        :param x:
        :return:
        '''
        extra= []
        if self.num_extra:
            P3, P4, P5 = x[1:]

            P3 = self.lateral_P3(P3)
            P4 = self.lateral_P4(P4)
            P5 = self.lateral_P5(P5)

            P6 = self.extra_convs[0](P5)
            P7 = self.extra_convs[1](P6)

            P6 = self.lateral_P6(P6)
            P7 = self.lateral_P7(P7)

        else:
            P3, P4, P5, P6, P7 = x

            P3 = self.lateral_P3(P3)
            P4 = self.lateral_P4(P4)
            P5 = self.lateral_P5(P5)
            P6 = self.lateral_P6(P6)
            P7 = self.lateral_P7(P7)

        P3_shape = P3.shape[2:]
        P4_shape = P4.shape[2:]
        P5_shape = P5.shape[2:]
        P6_shape = P6.shape[2:]
        P7_shape = P7.shape[2:]

        P3_1 = self.conv3_1(P3)
        P4_1 = self.conv4_1(self.scale3_4(self.dowm_con3_4(P3)) + self.scale5_4(self.upsample5_4(P5, P4_shape, mode="nearest")) + P4)
        P5_1 = self.conv5_1(P5)
        P6_1 = self.conv6_1(self.scale7_6(self.upsample7_6(P7, P6_shape, mode="nearest")) + self.scale5_6(self.dowm_con5_6(P5)) + P6)
        P7_1 = self.conv7_1(P7)



        P4_2 = self.conv4_2(P4_1)
        P3_2 = self.scale4_3(self.upsample4_3(P4_2, P3_shape, mode="nearest")) + self.conv3_2(P3_1)
        P5_2 = self.conv5_2(P5_1) + self.scale6_5(self.upsample6_5(P6, P5_shape, mode="nearest")) + self.shortcut_scale4_5(self.dowm_con4_5(P4))
        P6_2 = self.conv6_2(P6_1)
        P7_2 = self.conv7_2(P7_1) + self.scale6_7(self.dowm_con6_7(P6_2))


        return tuple([P3_2, P4_2, P5_2, P6_2, P7_2])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')