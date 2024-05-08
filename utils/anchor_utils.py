# import torch
# import torch.nn as nn
# from typing import List

# def anchor_to_box_transform(anchors, deltas):
#     """
#     shape:
#         anchors: anchor_num, 2(W, H)
#         deltas: N, anchor_num, 2(dw, dh)
#     """

#     boxes = torch.exp(deltas) * anchors
#     return boxes
