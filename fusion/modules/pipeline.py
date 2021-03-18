import torch

from modules.extractor import Extractor
from modules.integrator import Integrator


class Pipeline(torch.nn.Module):
    """
        Depth map fusion pipeline which integrates the regularized cost volume
        from MVS pipeline into a global latent feature volume and translates
        the feature volume into a shape offset map. 
    """

    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        self._device = config.MODEL.device
        self._extractor = Extractor(config.MODEL.device, config.MODEL.n_points)
        self._integrator = Integrator(config.MODEL.device)
        


    def train(self, sample, database):
        """
            Learned real-time depth map fusion pipelin.
            :param sample: Data sample with depth map, cost volume and camera calibrations.
            :param database: Dataset used to retrieve and update global states. 
            :return: The output from fusion model, which contains the estimated shape offset.
        """

        # Get current feature volume with the corresponding scene ID.
        device = self._device
        scene_volume = database[sample['scene_id']]
        current_state = scene_volume['current'].to(device) # x-y-z

        # Step 1: Extract the view aligned feature volume from current global state.
        depth_map = sample['depth_map'].to(device) # b-h-w
        extrinsics = sample['extrinsics'].to(device) # b-4-4
        intrinsics = sample['intrinsics'].to(device) # b-3-3
        origin = scene_volume['origin'].to(device) 
        values = self._extractor.forward(depth_map, extrinsics, intrinsics, current_state, origin)

        # Step 2: Run the fusion model on a stack of input.
        cost_volume = sample['cost_volume'].to(device) # b-h-w-c
        extracted_volume = values['interpolated_volume'].to(device) # b-h-w-n
        fusion_output = self._fusion(cost_volume, extracted_volume) # b-h-w-n

        # Step 3: Integrate the estimated volume back into the globle state.
        fusion_output = fusion_output.to(self._device) # b-h-w-n
        indices = values['indices'].to(self._device) # b-h-w-n-3
        updated_volume = self._integrator.forward(fusion_output, indices, current_state) # x-y-z

        # Step 4: Update the database and return values that are needed for loss evaluation.
        database.scenes_est[sample['scene_id']].volume = updated_volume.cpu().detach().numpy()

        output = dict()
        output['est_volume'] = fusion_output

        del scene_volume, current_state, depth_map, extrinsics, \
            intrinsics, cost_volume, extracted_volume, values, \
            indices, fusion_output, updated_volume
        return output


    def _fusion(self, cost_volume, extracted_volume):

        # Prepare input data by stacking extracted feature volume, cost volume, and ray directions.
        # TODO(zgz): Add ray direction in the stack.
        fusion_input = torch.cat([cost_volume, extracted_volume], dim=3)

        # Run fusion model on the stacked input data.
        fusion_output = self._fusion_network.forward(fusion_input)
        fusion_output = fusion_output.permute(0, 2, 3, 1)

        del fusion_input
        return fusion_output
        

