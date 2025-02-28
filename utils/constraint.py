import torch

from utils.distance import get_distance


class Constraint(torch.nn.Module):
    """
    Class to handle constraints and application of adversarial perturbations to images.

    Supports different constraints:
    - Perturbation magnitude
    - Perturbation distribution

    Supports different application modes:
    - Direct: Apply perturbation to the entire image
    - Patch: Apply perturbation to a specific patch of the image
    - Frame: Apply perturbation around the border (frame) of the image
    """

    def __init__(
        self,
        mode: str = "frame",
        epsilon: float = 1.0,
        norm_type: str = "linf",
        *,
        frame_width: int = 6,
        patch_size: tuple = (40, 40),
        patch_location: tuple = (0, 0),
        bound: tuple = (0.0, 1.0),
    ) -> None:
        """
        Initialize the constraint.

        Args:
            mode (str): Mode for perturbation ('direct', 'patch', or 'frame')
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
        # Validate inputs
        self._validate_inputs()
        self.mask: torch.Tensor = torch.zeros(1)

        if mode != "direct":
            self.norm_type = "linf"
            self.epsilon = 1.0
        self.distance = get_distance(self.norm_type)

    def _validate_inputs(self):
        """Validate class initialization inputs."""
        valid_modes = ["direct", "patch", "frame"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Mode must be one of {valid_modes}, got {self.mode} instead."
            )

        if self.mode == "patch" and (
            self.patch_size is None or self.patch_location is None
        ):
            raise ValueError(
                "Patch parameters must be provided when using 'patch' mode."
            )

        if self.mode == "frame" and self.frame_width is None:
            raise ValueError("Frame width must be provided when using 'frame' mode.")

        if self.mode == "direct" and (self.epsilon is None or self.norm_type is None):
            raise ValueError(
                "Max norm and norm type must be provided when using 'direct' mode."
            )
        valid_norms = ["linf", "l2", "l1", "l0"]
        if self.norm_type not in valid_norms:
            raise ValueError(
                f"Norm type must be one of {valid_norms}, got {self.norm_type} instead."
            )

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
        # Repeat the perturbation to the same shape of the image
        perturb = perturbation.expand_as(image).to(image.device)
        # Make a copy of the image to avoid modifying the original
        perturbed_image = image.clone()

        if self.mode == "direct":
            # Simply add the perturbation to the entire image
            self.mask = torch.zeros_like(image)

        elif self.mode == "patch":
            # Create a mask for the patch
            self.mask = torch.zeros_like(image)
            x, y = self.patch_location
            w, h = self.patch_size
            self.mask[..., y : y + h, x : x + w] = 1

        elif self.mode == "frame":
            # Create a mask for the frame
            self.mask = torch.ones_like(image)
            w = self.frame_width
            self.mask[..., w:-w, w:-w] = 0

        # Apply the perturbation only to the frame area
        perturbed_image = perturbed_image * (1 - self.mask) + perturb * self.mask

        # Ensure the resulting image has valid pixel values (assuming 0-1 range)
        perturbed_image = torch.clamp(perturbed_image, self.bound[0], self.bound[1])
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
