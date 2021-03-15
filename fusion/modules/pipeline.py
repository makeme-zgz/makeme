import torch

from modules.extractor import Extractor
from modules.integrator import Integrator


class Pipeline(torch.nn.Module):
    """
        Depth map fusion pipeline which integrates the regularized cost volume
        from MVS pipeline into a global latent feature volume and translates
        the feature volume into a shape offset map. 
    """

    def __init__(self, config, device):

        super(Pipeline, self).__init__()

        self.config = config

        self._extractor = Extractor(device)
        self._integrator = Integrator(config.MODEL)
        self._device = device


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
        current_state = scene_volume['current'].to(device)

        # Step 1: Extract the view aligned feature volume from current global state.
        depth_map = sample['depth_map'].to(device)
        extrinsics = sample['extrinsics'].to(device)
        intrinsics = sample['intrinsics'].to(device)
        origin = scene_volume['origin'].to(device)
        values = self._extractor.forward(depth_map, extrinsics, intrinsics, current_state, origin)

        # Step 2: Run the fusion model on a stack of input.
        cost_volume = sample['cost_volume'].to(device)
        extracted_volume = values['interpolated_volume'].to(device)
        fusion_output = self._fusion(cost_volume, extracted_volume)

        # Step 3: Integrate the estimated volume back into the globle state.
        updated_volume = self._integrator.forward(
            fusion_output.to(self._device), 
            values['indices'].to(self._device), 
            current_state, 
        )
        database.scenes_est[sample['scene_id']].volume = updated_volume.cpu().detach().numpy()

        # Step 4: Return values that are needed for loss evaluation.
        output = dict()
        output['est_volume'] = fusion_output['est_volume']

        del scene_volume, current_state, depth_map, extrinsics, intrinsics, cost_volume, extracted_volume, values
        return output


    def _fusion(self, cost_volume, extracted_volume):

        # Prepare input data by stacking extracted feature volume, cost volume, and ray directions.
        # TODO(zgz): Add ray direction in the stack.
        b, h, w, _ = cost_volume.shape
        fusion_input = torch.cat([cost_volume, extracted_volume], dim=3)

        # Run fusion model on the stacked input data.
        fusion_output = self._fusion_network.forward(fusion_input)
        fusion_output = fusion_output.permute(0, 2, 3, 1)

        # Reshape the updated local volume.
        b, h, w, c = fusion_output.shape
        fusion_output = fusion_output.view(b, h * w, c)

        del fusion_input
        return fusion_output
        

