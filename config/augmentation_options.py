augmentations = {
    "meta": [
        {
            "no": 1,
            "augmentation": """iaa.Sequential([iaa.Affine(translate_px={"x": -40}),
                                        iaa.AdditiveGaussianNoise(scale=0.1*255)])""",
        },
        {
            "no": 2,
            "augmentation": """iaa.Sequential([iaa.Affine(translate_px={"x": -40}),
                                        iaa.AdditiveGaussianNoise(scale=0.1*255)], random_order=True)""",
        },
        {
            "no": 3,
            "augmentation": """iaa.SomeOf(2, [iaa.Affine(rotate=45), iaa.AdditiveGaussianNoise(scale=0.2*255),
                                       iaa.Add(50, per_channel=True), iaa.Sharpen(alpha=0.5)])""",
        },
        {
            "no": 4,
            "augmentation": """iaa.SomeOf((0, None), [iaa.Affine(rotate=45), iaa.AdditiveGaussianNoise(scale=0.2*255),
                                               iaa.Add(50, per_channel=True), iaa.Sharpen(alpha=0.5)])""",
        },
        {
            "no": 5,
            "augmentation": """iaa.SomeOf(2, [iaa.Affine(rotate=45), iaa.AdditiveGaussianNoise(scale=0.2*255),
                                       iaa.Add(50, per_channel=True), iaa.Sharpen(alpha=0.5)],
                                   random_order=True)""",
        },
        {
            "no": 6,
            "augmentation": """iaa.OneOf([iaa.Affine(rotate=45), iaa.AdditiveGaussianNoise(scale=0.2*255),
                                   iaa.Add(50, per_channel=True), iaa.Sharpen(alpha=0.5)])""",
        },
        {
            "no": 7,
            "augmentation": """iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=2.0))""",
        },
        {
            "no": 8,
            "augmentation": """iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=2.0), iaa.Sequential(
            [iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)]))""",
        },
        {
            "no": 9,
            "augmentation": """iaa.WithChannels(0, iaa.Add((10, 100)))""",
        },
        {
            "no": 10,
            "augmentation": """iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))""",
        },
        {"no": 11, "augmentation": """iaa.Identity()"""},
        {"no": 12, "augmentation": """iaa.Noop()"""},
        {
            "no": 13,
            "augmentation": """iaa.Sequential([iaa.AssertShape((None, 32, 32, 3)), iaa.Fliplr(0.5)])""",
        },
        {
            "no": 14,
            "augmentation": """iaa.Sequential([iaa.AssertShape((None, (32, 64), 32, [1, 3])), iaa.Fliplr(0.5)])""",
        },
        {"no": 15, "augmentation": """iaa.ChannelShuffle(0.35)"""},
        {
            "no": 16,
            "augmentation": """iaa.ChannelShuffle(0.35, channels=[0, 1])""",
        },
        {
            "no": 17,
            "augmentation": """iaa.Sequential([iaa.Affine(translate_px={"x": (-100, 100)}),
                                        iaa.RemoveCBAsByOutOfImageFraction(0.5)])""",
        },
        {
            "no": 18,
            "augmentation": """iaa.Sequential([iaa.Affine(translate_px={"x": (-100, 100)}),
                                        iaa.ClipCBAsToImagePlanes()])""",
        },
    ],
    "arithmetic": [
        {"no": 1, "augmentation": """iaa.Add((-40, 40))"""},
        {"no": 2, "augmentation": """iaa.Add((-40, 40), per_channel=0.5)"""},
        {"no": 3, "augmentation": """iaa.AddElementwise((-40, 40))"""},
        {
            "no": 4,
            "augmentation": """iaa.AddElementwise((-40, 40), per_channel=0.5)""",
        },
        {
            "no": 5,
            "augmentation": """iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))""",
        },
        {
            "no": 6,
            "augmentation": """iaa.AdditiveGaussianNoise(scale=0.2 * 255)""",
        },
        {
            "no": 7,
            "augmentation": """iaa.AdditiveGaussianNoise(scale=0.2 * 255, per_channel=True)""",
        },
        {
            "no": 8,
            "augmentation": """iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255))""",
        },
        {
            "no": 9,
            "augmentation": """iaa.AdditiveLaplaceNoise(scale=0.2 * 255)""",
        },
        {
            "no": 10,
            "augmentation": """iaa.AdditiveLaplaceNoise(scale=0.2 * 255, per_channel=True)""",
        },
        {"no": 11, "augmentation": """iaa.AdditivePoissonNoise(40)"""},
        {"no": 12, "augmentation": """iaa.AdditivePoissonNoise(12)"""},
        {"no": 13, "augmentation": """iaa.Multiply((0.5, 1.5))"""},
        {
            "no": 14,
            "augmentation": """iaa.Multiply((0.5, 1.5), per_channel=0.5)""",
        },
        {"no": 15, "augmentation": """iaa.MultiplyElementwise((0.5, 1.5))"""},
        {
            "no": 16,
            "augmentation": """iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)""",
        },
        {"no": 17, "augmentation": """iaa.Cutout(nb_iterations=2)"""},
        {
            "no": 18,
            "augmentation": """iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)""",
        },
        {
            "no": 19,
            "augmentation": """iaa.Cutout(fill_mode="constant", cval=255)""",
        },
        {
            "no": 20,
            "augmentation": """iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=0.5)""",
        },
        {
            "no": 21,
            "augmentation": """iaa.Cutout(fill_mode="gaussian", fill_per_channel=True)""",
        },
        {"no": 22, "augmentation": """iaa.Dropout(p=(0, 0.2))"""},
        {
            "no": 23,
            "augmentation": """iaa.Dropout(p=(0, 0.2), per_channel=0.5)""",
        },
        {
            "no": 24,
            "augmentation": """iaa.CoarseDropout(0.02, size_percent=0.5)""",
        },
        {
            "no": 25,
            "augmentation": """iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))""",
        },
        {
            "no": 26,
            "augmentation": """iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)""",
        },
        {"no": 27, "augmentation": """iaa.Dropout2d(p=0.5)"""},
        {
            "no": 28,
            "augmentation": """iaa.Dropout2d(p=0.5, nb_keep_channels=0)""",
        },
        {
            "no": 29,
            "augmentation": """iaa.ReplaceElementwise(0.1, [0, 255])""",
        },
        {
            "no": 30,
            "augmentation": """iaa.ReplaceElementwise(0.1, [0, 255], per_channel=0.5)""",
        },
        {
            "no": 31,
            "augmentation": """iaa.ReplaceElementwise(0.1, iap.Normal(128, 0.4 * 128), per_channel=0.5)""",
        },
        {
            "no": 32,
            "augmentation": """iaa.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
                             iap.Normal(128, 0.4 * 128), per_channel=0.5)""",
        },
        {"no": 33, "augmentation": """iaa.SaltAndPepper(0.1)"""},
        {
            "no": 34,
            "augmentation": """iaa.SaltAndPepper(0.1, per_channel=True)""",
        },
        {
            "no": 35,
            "augmentation": """iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))""",
        },
        {
            "no": 36,
            "augmentation": """iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1), per_channel=True)""",
        },
        {"no": 37, "augmentation": """iaa.Pepper(0.1)"""},
        {"no": 38, "augmentation": """iaa.Invert(0.5)"""},
        {"no": 39, "augmentation": """iaa.Invert(0.25, per_channel=0.5)"""},
        {
            "no": 40,
            "augmentation": """iaa.Solarize(0.5, threshold=(32, 128))""",
        },
        {
            "no": 41,
            "augmentation": """iaa.JpegCompression(compression=(70, 99))""",
        },
    ],
    "artistic": [
        {"no": 1, "augmentation": """iaa.Cartoon()"""},
        {
            "no": 2,
            "augmentation": """iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
                                    saturation=2.0, edge_prevalence=1.0)""",
        },
    ],
    "blend": [
        {
            "no": 1,
            "augmentation": """iaa.BlendAlpha(0, iaa.Affine(rotate=(-20, 20)))""",
        },
        {
            "no": 2,
            "augmentation": """iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(100), background=iaa.Multiply(0.2))""",
        },
        {
            "no": 3,
            "augmentation": """iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13))""",
        },
        {
            "no": 4,
            "augmentation": """iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.Clouds())""",
        },
        {
            "no": 5,
            "augmentation": """iaa.BlendAlphaElementwise(0.5, iaa.Grayscale(1.0))""",
        },
        {
            "no": 6,
            "augmentation": """iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100))""",
        },
        {
            "no": 7,
            "augmentation": """iaa.BlendAlphaElementwise(
            (0.0, 1.0), iaa.Affine(rotate=(-20, 20)), per_channel=0.5)""",
        },
        {
            "no": 8,
            "augmentation": """iaa.BlendAlphaElementwise(
            (0.0, 1.0), foreground=iaa.Add(100), background=iaa.Multiply(0.2))""",
        },
        {
            "no": 9,
            "augmentation": """iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))""",
        },
        {
            "no": 10,
            "augmentation": """iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0))""",
        },
        {
            "no": 11,
            "augmentation": """iaa.BlendAlphaSimplexNoise(
            iaa.EdgeDetect(1.0), upscale_method="nearest")""",
        },
        {
            "no": 12,
            "augmentation": """iaa.BlendAlphaSimplexNoise(
            iaa.EdgeDetect(1.0), upscale_method="linear")""",
        },
        {
            "no": 13,
            "augmentation": """iaa.BlendAlphaSimplexNoise(
            iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0))""",
        },
        {
            "no": 14,
            "augmentation": """iaa.BlendAlphaFrequencyNoise(
            upscale_method="linear", exponent=-2, sigmoid=False)""",
        },
        {
            "no": 15,
            "augmentation": """iaa.BlendAlphaFrequencyNoise(sigmoid_thresh=iap.Normal(10.0, 5.0))""",
        },
        {
            "no": 16,
            "augmentation": """iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0))""",
        },
        {
            "no": 17,
            "augmentation": """iaa.BlendAlphaSomeColors(iaa.TotalDropout(1.0))""",
        },
        {
            "no": 18,
            "augmentation": """iaa.BlendAlphaSomeColors(
            iaa.MultiplySaturation(0.5), iaa.MultiplySaturation(1.5))""",
        },
        {
            "no": 19,
            "augmentation": """iaa.BlendAlphaSomeColors(
            iaa.AveragePooling(7), alpha=[0.0, 1.0], smoothness=0.0)""",
        },
        {
            "no": 20,
            "augmentation": """iaa.BlendAlphaSomeColors(
            iaa.AveragePooling(7), nb_bins=2, smoothness=0.0)""",
        },
        {
            "no": 21,
            "augmentation": """iaa.BlendAlphaSomeColors(
            iaa.AveragePooling(7), from_colorspace="BGR")""",
        },
        {
            "no": 22,
            "augmentation": """iaa.BlendAlphaHorizontalLinearGradient(
            iaa.AddToHue((-100, 100)))""",
        },
        {
            "no": 23,
            "augmentation": """iaa.BlendAlphaHorizontalLinearGradient(
            iaa.TotalDropout(1.0), min_value=0.2, max_value=0.8)""",
        },
        {
            "no": 24,
            "augmentation": """iaa.BlendAlphaHorizontalLinearGradient(iaa.AveragePooling(11), start_at=(0.0, 1.0),
                                                               end_at=(0.0, 1.0))""",
        },
        {
            "no": 25,
            "augmentation": """iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100)))""",
        },
        {
            "no": 26,
            "augmentation": """a26 = iaa.BlendAlphaVerticalLinearGradient(
            iaa.TotalDropout(1.0), min_value=0.2, max_value=0.8)""",
        },
        {
            "no": 27,
            "augmentation": """iaa.BlendAlphaVerticalLinearGradient(iaa.AveragePooling(11), 
                                           start_at=(0.0, 1.0), end_at=(0.0, 1.0))""",
        },
        {
            "no": 28,
            "augmentation": """iaa.BlendAlphaVerticalLinearGradient(
            iaa.Clouds(), start_at=(0.15, 0.35), end_at=0.0)""",
        },
        {
            "no": 29,
            "augmentation": """iaa.BlendAlphaRegularGrid(
            nb_rows=(4, 6), nb_cols=(1, 4), foreground=iaa.Multiply(0.0))""",
        },
        {
            "no": 30,
            "augmentation": """iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2, foreground=iaa.Multiply(0.0),
                                background=iaa.AveragePooling(8), alpha=[0.0, 0.0, 1.0])""",
        },
        {
            "no": 31,
            "augmentation": """iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4), 
                                 foreground=iaa.AddToHue((-100, 100)))""",
        },
        {
            "no": 32,
            "augmentation": """iaa.BlendAlphaBoundingBoxes("person", foreground=iaa.Grayscale(1.0))""",
        },
        {
            "no": 33,
            "augmentation": """iaa.BlendAlphaBoundingBoxes(["person", "car"], 
        foreground=iaa.AddToHue((-255, 255)))""",
        },
        {
            "no": 34,
            "augmentation": """iaa.BlendAlphaBoundingBoxes(
            ["person", "car"], foreground=iaa.AddToHue((-255, 255)), nb_sample_labels=1)""",
        },
    ],
    "gaussian_blur": [
        {"no": 1, "augmentation": """iaa.AverageBlur(k=(2, 11))"""},
        {"no": 2, "augmentation": """iaa.AverageBlur(k=((5, 11), (1, 3)))"""},
        {"no": 3, "augmentation": """iaa.MedianBlur(k=(3, 11))"""},
        {
            "no": 4,
            "augmentation": """iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), 
                            sigma_space=(10, 250))""",
        },
        {"no": 5, "augmentation": """iaa.MotionBlur(k=15)"""},
        {"no": 6, "augmentation": """iaa.MotionBlur(k=15, angle=[-45, 45])"""},
        {"no": 7, "augmentation": """iaa.MeanShiftBlur()"""},
    ],
    "color": [
        {
            "no": 1,
            "augmentation": """iaa.WithColorspace(
            to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(
                0, iaa.Add((0, 50))))""",
        },
        {
            "no": 2,
            "augmentation": """iaa.WithBrightnessChannels(iaa.Add((-50, 50)))""",
        },
        {
            "no": 3,
            "augmentation": """iaa.WithBrightnessChannels(iaa.Add((-50, 50)), to_colorspace=[
            iaa.CSPACE_Lab, iaa.CSPACE_HSV])""",
        },
        {
            "no": 4,
            "augmentation": """iaa.WithBrightnessChannels(iaa.Add((-50, 50)), from_colorspace=iaa.CSPACE_BGR)""",
        },
        {
            "no": 5,
            "augmentation": """iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))""",
        },
        {"no": 6, "augmentation": """iaa.MultiplyBrightness((0.5, 1.5))"""},
        {"no": 7, "augmentation": """iaa.AddToBrightness((-30, 30))"""},
        {
            "no": 8,
            "augmentation": """iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 50))))""",
        },
        {
            "no": 9,
            "augmentation": """iaa.WithHueAndSaturation(
            [iaa.WithChannels(0, iaa.Add((-30, 10))), iaa.WithChannels(1, [
                iaa.Multiply((0.5, 1.5)), iaa.LinearContrast((0.75, 1.25))])])""",
        },
        {
            "no": 10,
            "augmentation": """iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)""",
        },
        {
            "no": 11,
            "augmentation": """iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))""",
        },
        {
            "no": 12,
            "augmentation": """iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))""",
        },
        {"no": 13, "augmentation": """iaa.MultiplyHue((0.5, 1.5))"""},
        {"no": 14, "augmentation": """iaa.MultiplySaturation((0.5, 1.5))"""},
        {"no": 15, "augmentation": """iaa.RemoveSaturation()"""},
        {"no": 16, "augmentation": """iaa.RemoveSaturation(1.0)"""},
        {
            "no": 17,
            "augmentation": """iaa.RemoveSaturation(from_colorspace=iaa.CSPACE_BGR)""",
        },
        {
            "no": 18,
            "augmentation": """iaa.AddToHueAndSaturation((-50, 50), per_channel=True)""",
        },
        {"no": 19, "augmentation": """iaa.AddToHue((-50, 50))"""},
        {"no": 20, "augmentation": """iaa.AddToSaturation((-50, 50))"""},
        {
            "no": 21,
            "augmentation": """iaa.Sequential([iaa.ChangeColorspace(
            from_colorspace="RGB", to_colorspace="HSV"), iaa.WithChannels(
            0, iaa.Add((50, 100))), iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])""",
        },
        {"no": 22, "augmentation": """iaa.Grayscale(alpha=(0.0, 1.0))"""},
        {
            "no": 23,
            "augmentation": """iaa.ChangeColorTemperature((1100, 10000))""",
        },
        {"no": 24, "augmentation": """iaa.KMeansColorQuantization()"""},
        {
            "no": 25,
            "augmentation": """iaa.KMeansColorQuantization(n_colors=8)""",
        },
        {
            "no": 26,
            "augmentation": """iaa.KMeansColorQuantization(n_colors=(4, 16))""",
        },
        {
            "no": 27,
            "augmentation": """iaa.KMeansColorQuantization(from_colorspace=iaa.ChangeColorspace.BGR)""",
        },
        {
            "no": 28,
            "augmentation": """iaa.KMeansColorQuantization(
            to_colorspace=[iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV])""",
        },
        {"no": 29, "augmentation": """iaa.UniformColorQuantization()"""},
        {
            "no": 30,
            "augmentation": """iaa.UniformColorQuantization(n_colors=8)""",
        },
        {
            "no": 31,
            "augmentation": """iaa.UniformColorQuantization(n_colors=(4, 16))""",
        },
        {
            "no": 32,
            "augmentation": """iaa.UniformColorQuantization(
            from_colorspace=iaa.ChangeColorspace.BGR, to_colorspace=[
                iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV])""",
        },
        {
            "no": 33,
            "augmentation": """iaa.UniformColorQuantizationToNBits()""",
        },
        {
            "no": 34,
            "augmentation": """iaa.UniformColorQuantizationToNBits(nb_bits=(2, 8))""",
        },
        {
            "no": 35,
            "augmentation": """iaa.UniformColorQuantizationToNBits(
            from_colorspace=iaa.CSPACE_BGR, to_colorspace=[
                iaa.CSPACE_RGB, iaa.CSPACE_HSV])""",
        },
    ],
    "contrast": [
        {"no": 1, "augmentation": """iaa.GammaContrast((0.5, 2.0))"""},
        {
            "no": 2,
            "augmentation": """iaa.GammaContrast((0.5, 2.0), per_channel=True)""",
        },
        {
            "no": 3,
            "augmentation": """iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))""",
        },
        {
            "no": 4,
            "augmentation": """iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)""",
        },
        {"no": 5, "augmentation": """iaa.LogContrast(gain=(0.6, 1.4))"""},
        {
            "no": 6,
            "augmentation": """iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)""",
        },
        {"no": 7, "augmentation": """iaa.LinearContrast((0.4, 1.6))"""},
        {
            "no": 8,
            "augmentation": """iaa.LinearContrast((0.4, 1.6), per_channel=True)""",
        },
        {"no": 9, "augmentation": """iaa.AllChannelsCLAHE()"""},
        {
            "no": 10,
            "augmentation": """iaa.AllChannelsCLAHE(clip_limit=(1, 10))""",
        },
        {
            "no": 11,
            "augmentation": """iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)""",
        },
        {"no": 12, "augmentation": """iaa.CLAHE()"""},
        {"no": 13, "augmentation": """iaa.CLAHE(clip_limit=(1, 10))"""},
        {"no": 14, "augmentation": """iaa.CLAHE(tile_grid_size_px=(3, 21))"""},
        {
            "no": 15,
            "augmentation": """iaa.CLAHE(tile_grid_size_px=iap.Discretize(
            iap.Normal(loc=7, scale=2)), tile_grid_size_px_min=3)""",
        },
        {
            "no": 16,
            "augmentation": """iaa.CLAHE(tile_grid_size_px=((3, 21), [3, 5, 7]))""",
        },
        {
            "no": 17,
            "augmentation": """iaa.CLAHE(from_colorspace=iaa.CLAHE.BGR, to_colorspace=iaa.CLAHE.HSV)""",
        },
        {
            "no": 18,
            "augmentation": """iaa.AllChannelsHistogramEqualization()""",
        },
        {
            "no": 19,
            "augmentation": """iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())""",
        },
        {"no": 20, "augmentation": """iaa.HistogramEqualization()"""},
        {
            "no": 21,
            "augmentation": """iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization())""",
        },
        {
            "no": 22,
            "augmentation": """iaa.HistogramEqualization(
            from_colorspace=iaa.HistogramEqualization.BGR, 
            to_colorspace=iaa.HistogramEqualization.HSV)""",
        },
    ],
    "convolution": [
        {
            "no": 1,
            "augmentation": """iaa.Convolve(matrix=np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))""",
        },
        {
            "no": 2,
            "augmentation": """iaa.Convolve(matrix=np.array([[0, 0, 0], [0, -4, 1], [0, 2, 1]]))""",
        },
        {
            "no": 3,
            "augmentation": """iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))""",
        },
        {
            "no": 4,
            "augmentation": """iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))""",
        },
        {"no": 5, "augmentation": """iaa.EdgeDetect(alpha=(0.0, 1.0))"""},
        {
            "no": 6,
            "augmentation": """iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0))""",
        },
    ],
    "edges": [
        {"no": 1, "augmentation": """iaa.Canny()"""},
        {"no": 2, "augmentation": """iaa.Canny(alpha=(0.0, 0.5))"""},
        {
            "no": 3,
            "augmentation": """iaa.Canny(alpha=(0.0, 0.5), colorizer=iaa.RandomColorsBinaryImageColorizer(
            color_true=255, color_false=0))""",
        },
        {
            "no": 4,
            "augmentation": """iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])""",
        },
        {
            "no": 5,
            "augmentation": """iaa.Alpha((0.0, 1.0), iaa.Canny(alpha=1), iaa.MedianBlur(13))""",
        },
    ],
    "flip": [
        {"no": 1, "augmentation": """iaa.Fliplr(0.5)"""},
        {"no": 2, "augmentation": """iaa.Flipud(0.5)"""},
    ],
    "geometric": [
        {"no": 1, "augmentation": """iaa.Affine(scale=(0.5, 1.5))"""},
        {
            "no": 2,
            "augmentation": """iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})""",
        },
        {
            "no": 3,
            "augmentation": """iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})""",
        },
        {
            "no": 4,
            "augmentation": """iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})""",
        },
        {"no": 5, "augmentation": """iaa.Affine(rotate=(-45, 45))"""},
        {"no": 6, "augmentation": """iaa.Affine(shear=(-16, 16))"""},
        {
            "no": 7,
            "augmentation": """iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255))""",
        },
        {"no": 8, "augmentation": """iaa.ScaleX((0.5, 1.5))"""},
        {"no": 9, "augmentation": """iaa.ScaleY((0.5, 1.5))"""},
        {"no": 10, "augmentation": """iaa.TranslateX(px=(-20, 20))"""},
        {"no": 11, "augmentation": """iaa.TranslateX(percent=(-0.1, 0.1))"""},
        {"no": 12, "augmentation": """iaa.TranslateY(px=(-20, 20))"""},
        {"no": 13, "augmentation": """iaa.TranslateY(percent=(-0.1, 0.1))"""},
        {"no": 14, "augmentation": """iaa.Rotate((-45, 45))"""},
        {"no": 15, "augmentation": """iaa.ShearX((-20, 20))"""},
        {"no": 16, "augmentation": """iaa.ShearY((-20, 20))"""},
        {
            "no": 17,
            "augmentation": """iaa.PiecewiseAffine(scale=(0.01, 0.05))""",
        },
        {
            "no": 18,
            "augmentation": """iaa.PerspectiveTransform(scale=(0.01, 0.15))""",
        },
        {
            "no": 19,
            "augmentation": """iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)""",
        },
        {
            "no": 20,
            "augmentation": """iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)""",
        },
        {"no": 21, "augmentation": """iaa.Rot90(1)"""},
        {"no": 22, "augmentation": """iaa.Rot90([1, 3])"""},
        {"no": 23, "augmentation": """iaa.Rot90((1, 3))"""},
        {"no": 24, "augmentation": """iaa.Rot90((1, 3), keep_size=False)"""},
        {
            "no": 25,
            "augmentation": """iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))""",
        },
        {
            "no": 26,
            "augmentation": """iaa.WithPolarWarping(iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}))""",
        },
        {
            "no": 27,
            "augmentation": """iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))""",
        },
        {"no": 28, "augmentation": """iaa.Jigsaw(nb_rows=10, nb_cols=10)"""},
        {
            "no": 29,
            "augmentation": """iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))""",
        },
        {
            "no": 30,
            "augmentation": """iaa.Jigsaw(nb_rows=10, nb_cols=10, max_steps=(1, 5))""",
        },
    ],
    "corrupt_like": [
        {
            "no": 1,
            "augmentation": """iaa.imgcorruptlike.GaussianNoise(severity=2)""",
        },
        {
            "no": 2,
            "augmentation": """iaa.imgcorruptlike.ShotNoise(severity=2)""",
        },
        {
            "no": 3,
            "augmentation": """iaa.imgcorruptlike.ImpulseNoise(severity=2)""",
        },
        {
            "no": 4,
            "augmentation": """iaa.imgcorruptlike.SpeckleNoise(severity=2)""",
        },
        {
            "no": 5,
            "augmentation": """iaa.imgcorruptlike.GaussianBlur(severity=2)""",
        },
        {
            "no": 6,
            "augmentation": """iaa.imgcorruptlike.GlassBlur(severity=2)""",
        },
        {
            "no": 7,
            "augmentation": """iaa.imgcorruptlike.DefocusBlur(severity=2)""",
        },
        {
            "no": 8,
            "augmentation": """iaa.imgcorruptlike.MotionBlur(severity=2)""",
        },
        {
            "no": 9,
            "augmentation": """iaa.imgcorruptlike.ZoomBlur(severity=2)""",
        },
        {"no": 10, "augmentation": """iaa.imgcorruptlike.Fog(severity=2)"""},
        {"no": 11, "augmentation": """iaa.imgcorruptlike.Frost(severity=2)"""},
        {"no": 12, "augmentation": """iaa.imgcorruptlike.Snow(severity=2)"""},
        {
            "no": 13,
            "augmentation": """iaa.imgcorruptlike.Spatter(severity=2)""",
        },
        {
            "no": 14,
            "augmentation": """iaa.imgcorruptlike.Contrast(severity=2)""",
        },
        {
            "no": 15,
            "augmentation": """iaa.imgcorruptlike.Brightness(severity=2)""",
        },
        {
            "no": 16,
            "augmentation": """iaa.imgcorruptlike.Saturate(severity=2)""",
        },
        {
            "no": 17,
            "augmentation": """iaa.imgcorruptlike.JpegCompression(severity=2)""",
        },
        {
            "no": 18,
            "augmentation": """iaa.imgcorruptlike.Pixelate(severity=2)""",
        },
        {
            "no": 19,
            "augmentation": """iaa.imgcorruptlike.ElasticTransform(severity=2)""",
        },
    ],
    "pi_like": [
        {
            "no": 1,
            "augmentation": """iaa.Solarize(0.5, threshold=(32, 128))""",
        },
        {"no": 2, "augmentation": """iaa.pillike.Equalize()"""},
        {"no": 3, "augmentation": """iaa.pillike.Autocontrast()"""},
        {
            "no": 4,
            "augmentation": """iaa.pillike.Autocontrast((10, 20), per_channel=True)""",
        },
        {"no": 5, "augmentation": """iaa.pillike.EnhanceColor()"""},
        {"no": 6, "augmentation": """iaa.pillike.EnhanceContrast()"""},
        {"no": 7, "augmentation": """iaa.pillike.EnhanceBrightness()"""},
        {"no": 8, "augmentation": """iaa.pillike.EnhanceSharpness()"""},
        {"no": 9, "augmentation": """iaa.pillike.FilterBlur()"""},
        {"no": 10, "augmentation": """iaa.pillike.FilterSmooth()"""},
        {"no": 11, "augmentation": """iaa.pillike.FilterSmoothMore()"""},
        {"no": 12, "augmentation": """iaa.pillike.FilterEdgeEnhance()"""},
        {"no": 13, "augmentation": """iaa.pillike.FilterEdgeEnhanceMore()"""},
        {"no": 14, "augmentation": """iaa.pillike.FilterFindEdges()"""},
        {"no": 15, "augmentation": """iaa.pillike.FilterContour()"""},
        {"no": 16, "augmentation": """iaa.pillike.FilterEmboss()"""},
        {"no": 17, "augmentation": """iaa.pillike.FilterSharpen()"""},
        {"no": 18, "augmentation": """iaa.pillike.FilterDetail()"""},
        {
            "no": 19,
            "augmentation": """iaa.pillike.Affine(scale={"x": (0.8, 1.2), "y": (0.5, 1.5)})""",
        },
        {
            "no": 20,
            "augmentation": """iaa.pillike.Affine(translate_px={"x": 0, "y": [-10, 10]}, 
                                           fillcolor=128)""",
        },
        {
            "no": 21,
            "augmentation": """iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256))""",
        },
    ],
    "pooling": [
        {"no": 1, "augmentation": """iaa.AveragePooling(2)"""},
        {
            "no": 2,
            "augmentation": """iaa.AveragePooling(2, keep_size=False)""",
        },
        {"no": 3, "augmentation": """iaa.AveragePooling([2, 8])"""},
        {"no": 4, "augmentation": """iaa.AveragePooling((1, 7))"""},
        {"no": 5, "augmentation": """iaa.AveragePooling(((1, 7), (1, 7)))"""},
        {"no": 6, "augmentation": """iaa.MaxPooling(2)"""},
        {"no": 7, "augmentation": """iaa.MaxPooling(2, keep_size=False)"""},
        {"no": 8, "augmentation": """iaa.MaxPooling([2, 8])"""},
        {"no": 9, "augmentation": """iaa.MaxPooling((1, 7))"""},
        {"no": 10, "augmentation": """iaa.MaxPooling(((1, 7), (1, 7)))"""},
        {"no": 11, "augmentation": """iaa.MinPooling(2)"""},
        {"no": 12, "augmentation": """iaa.MinPooling(2, keep_size=False)"""},
        {"no": 13, "augmentation": """iaa.MinPooling([2, 8])"""},
        {"no": 14, "augmentation": """iaa.MinPooling((1, 7))"""},
        {"no": 15, "augmentation": """iaa.MinPooling(((1, 7), (1, 7)))"""},
        {"no": 16, "augmentation": """iaa.MedianPooling(2)"""},
        {
            "no": 17,
            "augmentation": """iaa.MedianPooling(2, keep_size=False)""",
        },
        {"no": 18, "augmentation": """iaa.MedianPooling([2, 8])"""},
        {"no": 19, "augmentation": """iaa.MedianPooling((1, 7))"""},
        {"no": 20, "augmentation": """iaa.MedianPooling(((1, 7), (1, 7)))"""},
    ],
    "segmentation": [
        {
            "no": 1,
            "augmentation": """iaa.Superpixels(p_replace=0.5, n_segments=64)""",
        },
        {
            "no": 2,
            "augmentation": """iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128))""",
        },
        {"no": 3, "augmentation": """iaa.UniformVoronoi((100, 500))"""},
        {
            "no": 4,
            "augmentation": """iaa.UniformVoronoi(250, p_replace=0.9, max_size=None)""",
        },
        {"no": 5, "augmentation": """iaa.RegularGridVoronoi(10, 20)"""},
        {
            "no": 6,
            "augmentation": """iaa.RegularGridVoronoi((10, 30), 20, p_drop_points=0.0, 
                                               p_replace=0.9, max_size=None)""",
        },
        {
            "no": 7,
            "augmentation": """iaa.RelativeRegularGridVoronoi(0.1, 0.25)""",
        },
        {
            "no": 8,
            "augmentation": """iaa.RelativeRegularGridVoronoi(
            (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=512)""",
        },
    ],
    "size": [
        {
            "no": 1,
            "augmentation": """iaa.Resize({"height": 32, "width": 64})""",
        },
        {
            "no": 2,
            "augmentation": """iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})""",
        },
        {"no": 3, "augmentation": """iaa.Resize((0.5, 1.0))"""},
        {
            "no": 4,
            "augmentation": """iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})""",
        },
        {"no": 5, "augmentation": """iaa.CropAndPad(percent=(-0.25, 0.25))"""},
        {
            "no": 6,
            "augmentation": """iaa.CropAndPad(percent=(0, 0.2), pad_mode=["constant", "edge"], pad_cval=(0, 128))""",
        },
        {
            "no": 7,
            "augmentation": """iaa.CropAndPad(px=((0, 30), (0, 10), (0, 30), (0, 10)), 
                                       pad_mode=ia.ALL, pad_cval=(0, 128))""",
        },
        {
            "no": 8,
            "augmentation": """iaa.CropAndPad(px=(-10, 10),sample_independently=False)""",
        },
        {
            "no": 9,
            "augmentation": """iaa.PadToFixedSize(width=100, height=100)""",
        },
        {
            "no": 10,
            "augmentation": """iaa.PadToFixedSize(width=100, height=100, position="center")""",
        },
        {
            "no": 11,
            "augmentation": """iaa.PadToFixedSize(width=100, height=100, pad_mode=ia.ALL)""",
        },
        {
            "no": 12,
            "augmentation": """iaa.Sequential([iaa.PadToFixedSize(width=100, height=100), 
                                        iaa.CropToFixedSize(width=100, height=100)])""",
        },
        {
            "no": 13,
            "augmentation": """iaa.CropToFixedSize(width=100, height=100)""",
        },
        {
            "no": 14,
            "augmentation": """iaa.CropToFixedSize(width=100, height=100, position="center")""",
        },
        {
            "no": 15,
            "augmentation": """iaa.Sequential([iaa.PadToFixedSize(width=100, height=100), 
                                        iaa.CropToFixedSize(width=100, height=100)])""",
        },
        {
            "no": 16,
            "augmentation": """iaa.PadToMultiplesOf(height_multiple=10, width_multiple=6)""",
        },
        {
            "no": 17,
            "augmentation": """iaa.CropToMultiplesOf(height_multiple=10, width_multiple=6)""",
        },
        {
            "no": 18,
            "augmentation": """iaa.CropToPowersOf(height_base=3, width_base=2)""",
        },
        {
            "no": 19,
            "augmentation": """iaa.PadToPowersOf(height_base=3, width_base=2)""",
        },
        {"no": 20, "augmentation": """iaa.CropToAspectRatio(2.0)"""},
        {"no": 21, "augmentation": """iaa.PadToAspectRatio(2.0)"""},
        {"no": 22, "augmentation": """iaa.CropToSquare()"""},
        {"no": 23, "augmentation": """iaa.PadToSquare()"""},
        {
            "no": 24,
            "augmentation": """iaa.CenterPadToFixedSize(height=20, width=30)""",
        },
        {
            "no": 25,
            "augmentation": """iaa.CenterCropToFixedSize(height=20, width=10)""",
        },
        {
            "no": 26,
            "augmentation": """iaa.CenterCropToMultiplesOf(height_multiple=10, width_multiple=6)""",
        },
        {
            "no": 27,
            "augmentation": """iaa.CenterPadToMultiplesOf(height_multiple=10, width_multiple=6)""",
        },
        {
            "no": 28,
            "augmentation": """iaa.CropToPowersOf(height_base=3, width_base=2)""",
        },
        {
            "no": 29,
            "augmentation": """iaa.CenterPadToPowersOf(height_base=3, width_base=2)""",
        },
        {"no": 30, "augmentation": """iaa.CenterCropToAspectRatio(2.0)"""},
        {"no": 31, "augmentation": """iaa.PadToAspectRatio(2.0)"""},
        {"no": 32, "augmentation": """iaa.CenterCropToSquare()"""},
        {"no": 33, "augmentation": """iaa.CenterPadToSquare()"""},
        {
            "no": 34,
            "augmentation": """iaa.KeepSizeByResize(iaa.Crop((20, 40), keep_size=False))""",
        },
        {
            "no": 35,
            "augmentation": """iaa.KeepSizeByResize(iaa.Crop((20, 40), keep_size=False), interpolation="nearest")""",
        },
        {
            "no": 36,
            "augmentation": """iaa.KeepSizeByResize(
            iaa.Crop((20, 40), keep_size=False), interpolation=["nearest", "cubic"], 
            interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES,
            interpolation_segmaps=iaa.KeepSizeByResize.NO_RESIZE)""",
        },
    ],
}

preset_1 = [
    [
        [
            {"sequence_group": "meta", "no": 7},
            {"sequence_group": "meta", "no": 9},
            {"sequence_group": "arithmetic", "no": 25},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "arithmetic", "no": 6},
            {"sequence_group": "arithmetic", "no": 10},
            {"sequence_group": "arithmetic", "no": 14},
            {"sequence_group": "arithmetic", "no": 27},
            {"sequence_group": "arithmetic", "no": 28},
            {"sequence_group": "arithmetic", "no": 36},
        ],
    ],
    [
        [
            {"sequence_group": "arithmetic", "no": 26},
            {"sequence_group": "arithmetic", "no": 29},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "arithmetic", "no": 40},
            {"sequence_group": "artistic", "no": 1},
            {"sequence_group": "artistic", "no": 2},
            {"sequence_group": "blend", "no": 2},
            {"sequence_group": "blend", "no": 4},
            {"sequence_group": "blend", "no": 5},
        ],
    ],
    [
        [
            {"sequence_group": "blend", "no": 8},
            {"sequence_group": "blend", "no": 18},
            {"sequence_group": "blend", "no": 19},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "blend", "no": 11},
            {"sequence_group": "blend", "no": 12},
            {"sequence_group": "blend", "no": 13},
            {"sequence_group": "blend", "no": 31},
            {"sequence_group": "gaussian_blur", "no": 3},
        ],
    ],
    [
        [
            {"sequence_group": "gaussian_blur", "no": 5},
            {"sequence_group": "color", "no": 1},
            {"sequence_group": "color", "no": 5},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "color", "no": 9},
            {"sequence_group": "color", "no": 10},
            {"sequence_group": "color", "no": 11},
            {"sequence_group": "color", "no": 15},
            {"sequence_group": "color", "no": 16},
            {"sequence_group": "blend", "no": 18},
        ],
    ],
    [
        [
            {"sequence_group": "color", "no": 21},
            {"sequence_group": "contrast", "no": 1},
            {"sequence_group": "gaussian_blur", "no": 1},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "color", "no": 23},
            {"sequence_group": "color", "no": 29},
            {"sequence_group": "contrast", "no": 2},
            {"sequence_group": "contrast", "no": 4},
            {"sequence_group": "contrast", "no": 6},
            {"sequence_group": "contrast", "no": 7},
        ],
    ],
    [
        [
            {"sequence_group": "contrast", "no": 20},
            {"sequence_group": "convolution", "no": 4},
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
        ],
        [
            {"sequence_group": "contrast", "no": 9},
            {"sequence_group": "contrast", "no": 10},
            {"sequence_group": "contrast", "no": 14},
            {"sequence_group": "contrast", "no": 18},
            {"sequence_group": "convolution", "no": 6},
        ],
    ],
    [
        [
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
            {"sequence_group": "corrupt_like", "no": 3},
        ],
        [
            {"sequence_group": "edges", "no": 2},
            {"sequence_group": "edges", "no": 5},
            {"sequence_group": "pi_like", "no": 13},
        ],
    ],
    [
        [
            {"sequence_group": "flip", "no": 1},
            {"sequence_group": "flip", "no": 2},
            {"sequence_group": "pi_like", "no": 5},
        ],
        [
            {"sequence_group": "corrupt_like", "no": 10},
            {"sequence_group": "corrupt_like", "no": 11},
            {"sequence_group": "corrupt_like", "no": 12},
            {"sequence_group": "corrupt_like", "no": 13},
            {"sequence_group": "pi_like", "no": 1},
            {"sequence_group": "pi_like", "no": 2},
            {"sequence_group": "pi_like", "no": 3},
            {"sequence_group": "pi_like", "no": 5},
            {"sequence_group": "pi_like", "no": 14},
        ],
    ],
]
