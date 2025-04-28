import torch


class Constraint:
    """
    Class to handle constraints and application of adversarial perturbations to images.

    Supports different constraints:
    - Perturbation magnitude
    - Perturbation distribution

    Supports different application modes:
    - Pixel: Apply perturbation to the entire image
    - Patch: Apply perturbation to a specific patch of the image
    - Frame: Apply perturbation around the border (frame) of the image
    """

    def __init__(
        self,
        mode: str = "frame",
        *,
        frame_width: int = 6,
        patch_size: tuple = (40, 40),
        patch_location: tuple = (0, 0),
        ref_size: int = 299,
    ) -> None:
        """
        Initialize the constraint.

        Args:
            mode (str): Mode for perturbation ('pixel', 'patch', 'frame', or 'corner')
            epsilon (float): Maximum perturbation magnitude
            norm_type (str): Type of norm ('linf', 'l2', 'l1')
            patch_size (tuple): (width, height) of the patch
            patch_location (tuple): (x, y) is top-left corner of the patch
            frame_width (int, optional): Width of the frame for frame mode
        """
        self.mode = mode.lower()
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.frame_width = frame_width
        self.ref_size = ref_size
        # Validate inputs
        self._mask: torch.Tensor = self.get_mask((1, 3, ref_size, ref_size))

    def get_mask(self, shape) -> torch.Tensor:
        if self.mode == "pixel":
            # Simply add the perturbation to the entire image
            self._mask = torch.ones(shape)

        elif self.mode == "patch":
            # Create a mask for the patch
            self._mask = torch.zeros(shape)
            x, y = self.patch_location
            w, h = self.patch_size
            if self.ref_size is not None:
                x = x * shape[-1] // self.ref_size
                y = y * shape[-2] // self.ref_size
                w = w * shape[-1] // self.ref_size
                h = h * shape[-2] // self.ref_size
            self._mask[..., y : y + h, x : x + w] = 1

        elif self.mode == "frame":
            # Create a mask for the frame
            self._mask = torch.ones(shape)
            w = self.frame_width
            h = self.frame_width
            if self.ref_size is not None:
                w = w * shape[-1] // self.ref_size
                h = h * shape[-2] // self.ref_size
            self._mask[..., w:-w, h:-h] = 0

        elif self.mode == "corner":
            self._mask = torch.zeros(shape)
            w, h = self.patch_size
            if self.ref_size is not None:
                w = w * shape[-1] // self.ref_size
                h = h * shape[-2] // self.ref_size
            self._mask[..., :w, :h] = 1
            self._mask[..., -w:, -h:] = 1
            self._mask[..., :w, -h:] = 1
            self._mask[..., -w:, :h] = 1
        return self._mask

    def apply_perturbation(self, image: torch.Tensor, perturbation: torch.Tensor):
        """
        Apply the perturbation to the image according to the specified mode.

        Args:
            image (torch.Tensor): Original image(s)
            perturbation (torch.Tensor): Perturbation to apply

        Returns:
            torch.Tensor: Perturbed image(s)
        """
        r = (image.shape[0] // self._mask.shape[0],) + (1,) * (image.ndim - 1)
        # Repeat the perturbation to the same shape of the image
        perturb = perturbation.repeat(r).to(image.device)
        _mask = self._mask.repeat(r).to(image.device)
        # Make a copy of the image to avoid modifying the original
        perturbed_image = image.clone()
        # Apply the perturbation only to the frame area
        if self.mode == 'pixel':
            perturbed_image = image + perturb
        else:
            perturbed_image = perturbed_image * (1 - _mask) + perturb * _mask
        return perturbed_image

    def __call__(self, image, perturbation):
        """
        Apply the perturbation to the image and clip the result.

        Args:
            image (torch.Tensor): Original image(s)
            perturbation (torch.Tensor): Perturbation to apply

        Returns:
            torch.Tensor: Perturbed and clipped image(s)
        """
        return self.apply_perturbation(image, perturbation)
