import torch
import torch.nn.functional as F


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                       dilation_w, group, group_channels, offset_scale):
    # Ensure padding is applied correctly
    input = F.pad(input, [pad_w, pad_w, pad_h, pad_h])

    # Extract dimensions
    N, C, H_in, W_in = input.shape
    P_ = kernel_h * kernel_w

    # Assuming _get_reference_points and _generate_dilation_grids are correctly defined elsewhere
    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w,
                                stride_h, stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)

    # Calculate spatial normalization (this might need adjustment based on your specific needs)
    spatial_norm = torch.tensor([W_in, H_in], dtype=input.dtype, device=input.device).reshape(1, 1, 1, 2).repeat(1, 1,
                                                                                                                 1,
                                                                                                                 group * P_)

    # Calculate sampling locations (this might also need adjustment)
    sampling_locations = (ref + grid * offset_scale).repeat(N, 1, 1, 1, 1).reshape(N, H_in, W_in, group, P_, 2) + \
                         offset.reshape(N, -1, group, P_, 2) * offset_scale / spatial_norm
    sampling_locations = 2 * sampling_locations - 1  # Scale to [-1, 1]

    # Reshape input for sampling
    input_ = input.view(N, group, group_channels, H_in, W_in).transpose(1, 2).contiguous().view(N * group,
                                                                                                group_channels, H_in,
                                                                                                W_in)

    # Prepare sampling grid for grid_sample (this might need further adjustment)
    sampling_grid_ = sampling_locations.view(N * group, H_in * W_in, P_, 2)

    # Perform grid sampling
    sampling_input_ = F.grid_sample(input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Apply mask and reshape output
    mask = mask.view(N, -1, group, P_).transpose(1, 2).reshape(N * group, 1, -1, P_)
    output = (sampling_input_ * mask).sum(-1).view(N, group, group_channels, -1)

    return output.transpose(1, 2).reshape(N, -1, output.shape[2] // group, output.shape[3] // group)

    # Note: You need to define _get_reference_points and _generate_dilation_grids functions.