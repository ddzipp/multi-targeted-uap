import torch

from utils.distance import get_distance


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
        epsilon: float = 1.0,
        norm_type: str = "linf",
        frame_width: int = 6,
        patch_size: tuple = (40, 40),
        patch_location: tuple = (0, 0),
        bound: tuple = (0.0, 1.0),
        ref_size: int | None = None,
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
        self.epsilon = epsilon
        self.norm_type = norm_type
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.frame_width = frame_width
        self.bound = bound
        self.ref_size = ref_size
        # Validate inputs
        self.mask: torch.Tensor = torch.zeros(1)

        if mode != "pixel":
            self.norm_type = "linf"
            self.epsilon = 1.0
        self.distance = get_distance(self.norm_type)

    def clip_perturbation(self, perturbation, original):
        return self.distance.clip_perturbation(
            perturbation, epsilon=self.epsilon, references=original
        )

    def apply_perturbation(self, image: torch.Tensor, perturbation: torch.Tensor):
        """
        Apply the perturbation to the image according to the specified mode.

        Args:
            image (torch.Tensor): Original image(s)
            perturbation (torch.Tensor): Perturbation to apply

        Returns:
            torch.Tensor: Perturbed image(s)
        """
        assert image.shape == perturbation.shape
        # Repeat the perturbation to the same shape of the image
        perturb = perturbation.expand_as(image).to(image.device)
        # Make a copy of the image to avoid modifying the original
        perturbed_image = image.clone()

        if self.mode == "pixel":
            # Simply add the perturbation to the entire image
            self.mask = torch.zeros_like(image)

        elif self.mode == "patch":
            # Create a mask for the patch
            self.mask = torch.zeros_like(image)
            x, y = self.patch_location
            w, h = self.patch_size
            if self.ref_size is not None:
                x = x * image.shape[-1] // self.ref_size
                y = y * image.shape[-2] // self.ref_size
                w = w * image.shape[-1] // self.ref_size
                h = h * image.shape[-2] // self.ref_size
            self.mask[..., y : y + h, x : x + w] = 1

        elif self.mode == "frame":
            # Create a mask for the frame
            self.mask = torch.ones_like(image)
            w = self.frame_width
            if self.ref_size is not None:
                w = w * image.shape[-1] // self.ref_size
                h = h * image.shape[-2] // self.ref_size
            self.mask[..., w:-w, h:-h] = 0

        elif self.mode == "corner":
            self.mask = torch.zeros_like(image)
            w, h = self.patch_size
            if self.ref_size is not None:
                w = w * image.shape[-1] // self.ref_size
                h = h * image.shape[-2] // self.ref_size
            self.mask[..., :w, :h] = 1
            self.mask[..., -w:, -h:] = 1
            self.mask[..., :w, -h:] = 1
            self.mask[..., -w:, :h] = 1

        # Apply the perturbation only to the frame area
        perturbed_image = perturbed_image * (1 - self.mask) + perturb * self.mask
        # Ensure the resulting image has valid pixel values (assuming 0-1 range)
        # perturbed_image = torch.clamp(perturbed_image, self.bound[0], self.bound[1])
        if self.mode == "pixel":
            perturbed_image = self.clip_perturbation(perturbed_image, image)
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
