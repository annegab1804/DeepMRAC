import SimpleITK as sitk
import argparse
import os


def register_umap_to_pet(
    fixed_pet_path: str,
    moving_umap_path: str,
    output_path: str,
    dof: int = 6
) -> None:
    """Performs co-registration of an attenuation map (u-map) to a PET image.

    Supports three levels of transformation flexibility controlled by the `dof` argument:
        - 6  DOF: Rigid       — 3 rotations + 3 translations          (Euler3DTransform)
        - 9  DOF: Similarity  — 6 rigid + 1 isotropic scale           (Similarity3DTransform)
        - 12 DOF: Affine      — 9 rigid/scale + 3 shears              (AffineTransform)

    Registration is performed against a noisy PET-no-AC image. Several strategies
    are applied to maximise robustness:
        - Otsu-based brain mask to exclude the empty circular FOV ring
        - Multi-resolution pyramid (4x → 2x → 1x) to escape local minima
        - Normalized Correlation metric, better suited to noisy same-contrast images
        - Conservative learning rate and high iteration count for noisy data

    Args:
        fixed_pet_path (str): Path to the reference PET image (reconstructed without AC).
        moving_umap_path (str): Path to the u-map image to be registered.
        output_path (str): Path where the registered u-map will be saved.
        dof (int): Degrees of freedom for the transform. Must be 6, 9, or 12. Default: 6.
    """
    if dof not in (6, 9, 12):
        raise ValueError(f"Invalid dof={dof}. Must be 6, 9, or 12.")

    # 1. Load images
    fixed_image  = sitk.ReadImage(fixed_pet_path,  sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_umap_path, sitk.sitkFloat32)

    # Clamp negative u-map values before registration
    moving_image = sitk.Threshold(moving_image, lower=0.0, upper=100.0, outsideValue=0.0)

    # 2. Build an Otsu mask on the PET to exclude the empty FOV ring
    # OtsuThreshold returns a mask where voxels BELOW the threshold = 1,
    # so we invert it to keep the bright brain/skull region.
    print("Computing Otsu mask to exclude empty FOV ring...")
    otsu_mask = sitk.OtsuThreshold(fixed_image, 0, 1)
    brain_mask = sitk.InvertIntensity(otsu_mask, maximum=1)  # bright region = 1

    # 3. Registration framework
    registration_method = sitk.ImageRegistrationMethod()

    # Metric: Normalized Correlation
    registration_method.SetMetricAsCorrelation()

    # Restrict metric evaluation to the masked brain/skull region only
    registration_method.SetMetricFixedMask(brain_mask)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.30)

    # Linear interpolation at non-grid positions
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer: conservative learning rate + high iteration count for noisy data.
    # Scales are computed from physical shift so rotation/translation are balanced.
    learning_rate  = {6: 0.5,  9: 0.3,  12: 0.2}[dof]
    num_iterations = {6: 300,  9: 400,  12: 500}[dof]
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=20,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 4. Multi-resolution pyramid  (coarse → fine)
    
    # Starting at 4x downsampling lets the optimizer find the correct
    # basin of attraction before refining at full resolution, which is
    # critical when the PET-no-AC is very noisy.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 5. Transform initialisation
    # All modes start from a geometry-based center alignment (Euler3D).
    # For 9 / 12 DOF the result is promoted to a richer transform.
    euler_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    if dof == 6:
        initial_transform = euler_transform

    elif dof == 9:
        initial_transform = sitk.Similarity3DTransform()
        initial_transform.SetMatrix(euler_transform.GetMatrix())
        initial_transform.SetTranslation(euler_transform.GetTranslation())
        initial_transform.SetFixedParameters(euler_transform.GetFixedParameters())
        initial_transform.SetScale(1.0)

    else:  # dof == 12
        initial_transform = sitk.AffineTransform(3)
        initial_transform.SetMatrix(euler_transform.GetMatrix())
        initial_transform.SetTranslation(euler_transform.GetTranslation())
        initial_transform.SetFixedParameters(euler_transform.GetFixedParameters())

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 6. Execute
    print(f"Starting multi-resolution registration ({dof} DOF, 3 levels: 4x→2x→1x)...")
    final_transform = registration_method.Execute(fixed_image, moving_image)

    print(f"Stop condition : {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric   : {registration_method.GetMetricValue():.6f}")
    print(f"Iterations used: {registration_method.GetOptimizerIteration()}")

    # 7. Resample u-map onto PET grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampler.SetDefaultPixelValue(0)

    output_image = resampler.Execute(moving_image)

    # Remove interpolation artifacts (negative values / out-of-range)
    output_image = sitk.Threshold(output_image, lower=0.0, upper=100.0, outsideValue=0.0)

    sitk.WriteImage(output_image, output_path)
    print(f"Registered u-map saved to: {output_path}")


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='Co-register u-map to PET-no-AC using SimpleITK.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "DOF options:\n"
            "  6  — Rigid       : 3 rotations + 3 translations                [default]\n"
            "  9  — Similarity  : rigid + isotropic scale\n"
            "  12 — Affine      : similarity + anisotropic scale + shear\n\n"
            "Note: 6 DOF is strongly recommended for u-map to PET registration,\n"
            "      as the skull is a rigid structure. Higher DOF can cause the\n"
            "      optimizer to fit scanner FOV geometry rather than anatomy.\n"
        )
    )

    parser.add_argument(
        "--pet_no_att_path",
        help="Path to the NIfTI PET image reconstructed WITHOUT attenuation correction.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--umap_path",
        help="Path to the NIfTI u-map image that needs to be co-registered.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        help="Output path for the final co-registered u-map.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dof",
        help=(
            "Degrees of freedom for the registration transform: "
            "6 (rigid), 9 (similarity), or 12 (affine). Default: 6."
        ),
        type=int,
        choices=[6, 9, 12],
        default=6,
    )

    args = parser.parse_args()

    if os.path.exists(args.pet_no_att_path) and os.path.exists(args.umap_path):
        register_umap_to_pet(
            fixed_pet_path=args.pet_no_att_path,
            moving_umap_path=args.umap_path,
            output_path=args.output_path,
            dof=args.dof
        )
    else:
        print("Error: One or more input files were not found. Check your paths.")


if __name__ == "__main__":
    main()