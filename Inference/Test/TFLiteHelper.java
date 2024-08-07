void init() {
    try {
        Interpreter.Options opt = new Interpreter.Options();
        tflite = new Interpreter(loadmodelfile(context), opt);
    } catch (Exception e) {
        e.printStackTrace();
    }
}

private TensorImage loadImage(final Bitmap bitmap) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
}

private TensorOperator getPreprocessNormalizeOp() {
    return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
}

private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
    String MODEL_NAME = "model.tflite";
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startoffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
}